"""Standalone 24/7 paper-trading runner — the bot's heartbeat.

Independent of Claude: run `python runner.py` and it loops forever, or let the
cockpit start/stop it in a thread. Each tick it:
  1. reads market-wide news sentiment (risk-off can pause new entries),
  2. for every symbol: manages any open paper position (stop / target / BE+trail),
  3. looks for a fresh rule signal, then gates it through ML win-prob + news +
     the risk manager before opening a paper position,
  4. logs every fill + post-mortem to the journal and writes status JSON for the UI.

Paper fills use the latest price (no real orders). Bybit testnet execution is a
drop-in swap of `PaperBroker.market()` later.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone

import config
from data.bybit import get_klines_cached
from features.indicators import feature_frame, atr, trend_label
from strategies import multi_angle, opportunity_snapshot
from strategies.snapback import generate_signals as snapback
from strategies.continuation import combined_signals
from strategies.range_fade import generate_signals as range_fade
from ml.meta import predict_proba, FEATURE_COLS, expected_value
from features.context import bar_context, signal_context, BAR_CONTEXT_COLS
from ml.forecast import forecast as next_move, MODEL_INTERVAL
from features.microstructure import snapshot as ms_snapshot
from brokers import TradingViewBroker
from brokers.bybit import BybitBroker
import news
import journal
import market
import conviction
import playbook
from risk import RiskManager

STATE_PATH = config.ROOT / "paper_state.json"
LOG_PATH = config.ROOT / "runner.log"

START_EQUITY = 1000.0
# Threshold is set relative to the strategy's reward:risk, NOT an arbitrary 50%.
# At RR 3:1 the breakeven win-rate is only 25%, so even a 0.35 win-prob setup has
# strongly positive expectancy (0.35*3 - 0.65*1 = +0.40R). Walk-forward sweep
# (15m, multi_angle): thr 0.35 -> PF 2.34 / 66% coverage; 0.38 -> PF 2.45 / 61%.
# Picked 0.35 (user wants more activity) — still ~2x the take-all PF (1.83), just
# trades more of the +EV signals. See train_ml.py for the sweep.
# --- Active strategy: SNAPBACK + CONTINUATION (+ ML) on 1h ---
# Snapback (mean-reversion dip/flush) covers pullbacks; Continuation (with-trend
# breakout) covers grinds where price never dips. ML meta-label filters both.
# Tuned (walk-forward, 24 coins, combined pool): ML thr 0.45 => ~4 trades/day,
# 57% win, PF ~2.0. Snapback alone sat idle in steady grind-up markets.
STRATEGY = "snapback+cont+range"
SNAPBACK_PARAMS = dict(adx_min=15.0, stretch_pct=0.5, rsi_os=48.0, rsi_ob=52.0, atr_stop=1.0)
CONT_PARAMS = dict(adx_min=25.0, lookback=20, atr_stop=1.5, rr=2.0)  # higher natural rr => higher conviction score
# RANGE-FADE fills the FLAT-market gap (ADX<18): fade Bollinger-band REJECTIONS (wick
# pokes the band, bar closes back inside) to the mean when snapback/continuation idle.
# This fixes the dead-market zero-trade drought. Runs on its OWN timeframe (RANGE_INTERVAL).
# TF study (24 coins): higher TF = cleaner edge (2h ML PF 1.43) but far too rare — 2h gave
# ZERO signals through the current dead coil, i.e. it would NOT fix the drought. 30m is the
# balance: ML PF 1.32 / 56% win full-sample AND it actually fires in flat markets (~17 raw
# signals in the last 40h here). Honest caveat: the fade's edge is regime-dependent (recent
# chop favours it; an older window was breakeven) — the ML + news + market-bias + risk gates
# keep it selective, and the journal self-learns. A modest, frequent chop edge, not a printer.
RANGE_INTERVAL = "30m"
RANGE_BARS = 600
# Tuned 2026-06-10 (user: "chop/flat market me bhi achha trade kare"): backtest sweep (24 coins, 30m,
# 1500 bars, 2-window robust) — dropping wick_reject + widening atr_stop 0.8->1.2 + min_rr 0.8->0.6
# lifted chop edge from 81 trd/57% win/PF 1.10/$0.49 to 134 trd/66% win/PF 1.29/$1.11, robust in BOTH
# halves (1.25/1.32). adx_max kept at 18 — raising it to 20-22 added trades but killed the edge (fades
# mild trends = knife-catch). ~65% more chop trades with a BETTER win-rate.
RANGE_PARAMS = dict(adx_max=18.0, atr_stop=1.2, min_rr=0.6, min_stretch=0.3, wick_reject=False, rsi_turn=False)
USE_TRAIL = True        # trailing stop ON (user 2026-06-12): once a trade is +1R, stop moves to
                        # breakeven then trails by ATR — locks profit, lets winners run, turns
                        # would-be losers into break-even. The conviction target is the ceiling.
                        # Measured neutral (±0.002R) once the partial below is in place; kept
                        # because it costs nothing and protects a runner that stalls short of
                        # its target.
# --- PARTIAL TAKE-PROFIT: bank a slice early, then risk nothing ---
# See the long note in _manage for the measurements. Short version: a third off at
# +0.5R with the stop to breakeven takes the win rate from 32% to 70% on identical
# entries, at the same expectancy — and it is the change that turns "three losses
# today" into "one loss and two small wins".
USE_PARTIAL = os.environ.get("QUANT_PARTIAL", "1") != "0"
PARTIAL_AT_R = float(os.environ.get("QUANT_PARTIAL_R", "0.5"))
PARTIAL_FRAC = float(os.environ.get("QUANT_PARTIAL_FRAC", "0.34"))
# THE ENTRY GATE — expected value, not win probability (2026-08-10).
#
# The old gate was `P(win) >= 0.60`, chosen from a walk-forward that reported 55%
# win / PF 1.93. Re-measured properly, those numbers were not real. Two flaws:
#   1. the folds were split by ROW ORDER, and rows were stacked coin after coin, so
#      "train on the past" was really "train on the first N coins" — the training
#      set contained bars from later in time than the test rows;
#   2. fees and slippage were not subtracted, and they cost ~0.1R per trade.
# Fixed both (chronological split, 48-bar embargo for overlapping labels, costs in)
# and re-ran on 9,652 signals across 374 days: P(win)>=0.60 earns +0.037R/trade at
# 0.9 trades a day. Essentially nothing.
#
# What does work is gating on EXPECTED VALUE, EV = p*RR - (1-p), with the model fed
# the CONTEXT of each setup (features/context.py). Same data, same discipline:
#     EV >= 0.15   41.0% win  PF 1.06  +0.037R   4.7 trades/day
#     EV >= 0.20   42.1% win  PF 1.11  +0.071R   3.6 trades/day
#     EV >= 0.25   43.0% win  PF 1.16  +0.099R   2.8 trades/day   <- chosen
#     EV >= 0.30   43.1% win  PF 1.18  +0.114R   2.2 trades/day
#     EV >= 0.40   39.5% win  PF 1.04  +0.028R   1.4 trades/day
# 0.25 has the best R/day and is positive in BOTH halves of the period (PF 1.32 in
# the first, 1.11 in the second) — it is not one lucky regime. The same EV gate on
# the OLD feature set fails the second half, so the context features are the edge,
# not the gate alone.
#
# Why EV beats a probability threshold: a 40% shot paying 3:1 is a better trade than
# a 55% shot paying 1:1, and a flat P(win) cut cannot tell them apart — it discards
# exactly the trades whose payoff covers the losers.
#
# RAISED 0.25 -> 0.40 (2026-08-16). The 0.25 figure above was measured on the gate
# ALONE, with every signal taken and held to its target. Re-measured with the whole
# stack that now runs — the playbook's quality filters, a third of the position
# banked at +0.5R, and the daily loss budget — the marginal trades between EV 0.25
# and 0.40 are the ones that bleed: raising the gate lifts expectancy from +0.269R
# to +0.321R a trade and the win rate from 78% to 80%, in both halves of the period
# (+0.379 / +0.257). It costs frequency (1.7 -> 1.2 trades a day from this
# timeframe), which is the trade the operator asked for. See research_manage.py.
EV_MIN = float(os.environ.get("QUANT_EV_MIN", "0.40"))
ML_MIN_PROB = 0.0       # superseded by EV_MIN; kept so old configs don't break
# Sanity bounds on a setup that has aged since it fired (see _score). Not tuned for
# edge — a stop-distance floor WAS tested as a filter and helped one half of the
# year while hurting the other, so it is deliberately set low enough to reject only
# the degenerate cases, not to select trades.
MIN_STOP_PCT = 0.004    # 0.4% — below the 1st percentile of a year of real setups
MIN_STOP_ATR = 0.30     # a stop closer than a third of an ATR is inside the noise
RR_CAP = 3.0            # the widest reward:risk the model has ever been trained on
STALE_EV = -9.0         # sentinel: "this setup is no longer tradeable"
CHART_BARS = 120        # how much history the bot draws on its own chart
DECISION_INTERVAL = "1h"
SIGNAL_BARS = 500
SIGNAL_FRESH_BARS = 2   # a setup from the last N closed bars still counts if price is still in play
BE_AFTER_R = 1.0        # move stop to breakeven once +1R in profit (then trail)
TRAIL_ATR_MULT = 2.5    # trail the stop this many ATRs behind price

# --- FORWARD-LOOKING LAYER (what does the next move look like?) ---
# The next-move forecaster (ml/forecast.py) and live microstructure (open interest,
# funding, order book, taker flow) do NOT pick trades — the rule strategies still do.
# They act on a setup that already exists in two narrow ways:
#   1. VETO, only on CLEAR disagreement (thresholds below are intentionally strict,
#      because this bot's historical failure mode was gating itself into zero trades),
#   2. NUDGE conviction, which moves size and target a little.
# Set USE_FORECAST=False to switch the whole layer off and trade exactly as before.
USE_FORECAST = True
FORECAST_VETO = 0.56    # block only if the CALIBRATED prob of the opposite direction >= this.
                        # Calibrated output tops out near 0.57, so this fires rarely and
                        # only on the model's strongest disagreements.
MS_VETO_BIAS = 0.35     # block only if microstructure leans this hard the other way.
FORECAST_TOP_N = 8      # how many watch-list coins get a full forecast each tick (API budget)


def _gen_signals(df, feats, a):
    """TREND lenses (snapback + continuation) on the primary DECISION_INTERVAL (1h).
    The FLAT/CHOP lens (range-fade) is scanned separately on its own higher timeframe
    via `_range_signal` — different regimes want different timeframes — so this returns
    only the trend signals. Range-fade is merged in `tick()` as a fallback."""
    return combined_signals(df, feats, a, snapback_params=SNAPBACK_PARAMS, cont_params=CONT_PARAMS)


def _range_signal(sym):
    """The FLAT-market lens on its own (higher, robust) timeframe. Returns
    (df, feats, atr, fresh_signal) so the caller can gate/enter using the SAME
    timeframe's features as the signal. None signal if no fresh fade right now."""
    rdf = get_klines_cached(sym, RANGE_INTERVAL, bars=RANGE_BARS, max_age_min=20)
    rfeats = feature_frame(rdf)
    ra = atr(rdf, 14)
    rsigs = range_fade(rdf, rfeats, ra, **RANGE_PARAMS)
    rsig = _fresh_signal(rsigs, float(rdf["close"].iloc[-1]))
    return rdf, rfeats, ra, rsig


def _fresh_signal(sigs, price: float, lookback: int = SIGNAL_FRESH_BARS):
    """Most-recent signal within the last `lookback` closed bars that is STILL
    actionable — price hasn't yet reached its stop or target. Lets the bot take a
    setup that formed a bar or two ago (a real trader does this) instead of only
    the exact current bar, which made fleeting setups slip through."""
    for k in range(1, lookback + 1):
        if len(sigs) >= k and sigs[-k] is not None:
            s = sigs[-k]
            if s.side == 1 and s.stop < price < s.target:
                return s
            if s.side == -1 and s.target < price < s.stop:
                return s
    return None

# Risk profile for the live TradingView paper account. Used only in mode="tradingview".
# The per-trade dollar risk is now chosen by the conviction engine (conviction.py,
# RISK_MIN..RISK_MAX) — these are the OUTER guardrails it runs inside: a % ceiling, a
# concurrent cap, the daily kill-switch, and a hard 10%-of-equity clamp on any one trade.
TV_RISK = {
    "max_risk_per_trade_pct": 6.0,    # % ceiling (fallback when conviction doesn't size)
    "max_concurrent_positions": 6,    # let it take as many qualifying setups as it finds
    "max_daily_drawdown_pct": 15.0,   # daily kill-switch still caps total daily loss
    "default_leverage": 10,           # notional cap headroom (conviction sizing keeps notional small)
    "max_loss_usd": 0.80,             # fallback loss cap; conviction risk (<=0.80) is the live driver
}

# Risk profile for the REAL Bybit account (mode="bybit"). Deliberately tighter than
# the TradingView paper profile: on a paper account a bug costs a screenshot, here it
# costs money. Leverage is halved and the daily kill-switch is far stricter.
# NOTE ON SMALL ACCOUNTS: Bybit enforces a minimum order size per contract (e.g. BTC
# 0.001 ~ $100 notional). On a ~$12 account the expensive majors will be REJECTED with
# a clear message and the bot simply moves on to coins it can actually size.
# TRADING CAPITAL vs ACCOUNT BALANCE.
# Bybit hands a demo account ~$166k. Sizing against that would risk $0.55 on a
# $166,000 book — 0.0003% a trade — so the demo would prove the plumbing works but
# would rehearse nothing about how the bot behaves on the money actually at stake.
# Bybit's UI can't set the demo balance to an arbitrary figure (a reset just restores
# the default), so the cap lives here instead: the bot trades a NOTIONAL book of
# QUANT_CAPITAL that starts at this value and then moves with the account's real PnL.
# Set it to whatever you intend to trade for real, so the demo is a true rehearsal.
# QUANT_CAPITAL="full" (or unset) => the bot manages the WHOLE account balance and
# sizes off it directly. A number => it trades a notional book of that size instead,
# which is what you want to rehearse a smaller account on a big demo balance.
_cap = os.environ.get("QUANT_CAPITAL", "full").strip().lower()
BYBIT_CAPITAL = None if _cap in ("full", "", "0", "auto") else float(_cap)


def _saved_capital():
    """A capital figure the operator set from the cockpit (chat command) outlives the
    process — otherwise a restart would silently put the bot back on the env value and
    it would resume trading a size the operator had explicitly changed.

    Returns the sentinel "__none__" when nothing was ever set by hand.
    """
    if not STATE_PATH.exists():
        return "__none__"
    try:
        return json.loads(STATE_PATH.read_text()).get("capital_override", "__none__")
    except Exception:  # noqa: BLE001
        return "__none__"


_saved = _saved_capital()
if _saved != "__none__":
    BYBIT_CAPITAL = None if _saved in (None, "full") else float(_saved)

# WHEN THIS BOOK STARTED — the anchor the $100 account's PnL is measured from.
#
# The old anchor was a BALANCE (`bybit_real_start`): the exchange balance seen at
# boot, with book equity = capital + (balance now - balance then). On a host whose
# disk is ephemeral that is re-read as "the balance right now" after every restart,
# so the book snapped back to exactly $100 and every trade the bot had ever taken
# vanished from the equity. That is the "balance still says $100" bug.
#
# A TIMESTAMP cannot go stale the same way. Book equity is recomputed from the
# exchange each tick as:
#       capital + realised PnL since the anchor + unrealised PnL now
# so after any restart, redeploy or wiped disk the number rebuilds itself from
# Bybit's own records. Set QUANT_BOOK_START (ISO date or epoch ms) to move it.
BOOK_START_ENV = os.environ.get("QUANT_BOOK_START", "").strip()

# A closed trade whose position was this many times the book cannot have come FROM
# the book. The same demo account ran in full-account mode before the $100 cap
# existed, and it left four positions of $34k-$81k notional behind (2026-08-10).
# Summing those into a $100 book's PnL would have read -$1,057 of "equity". With
# 5x leverage and 8 concurrent positions the book's own worst-case exposure is 5x
# capital, so 50x is far outside anything it can produce — this rejects the other
# regime's trades without ever touching one of its own.
FOREIGN_NOTIONAL_X = 50

BYBIT_RISK = {
    "max_risk_per_trade_pct": 4.0,
    # Raised 3 -> 8 alongside the 24 -> 59 coin universe: with more markets scanned,
    # good setups cluster, and a cap of 3 would silently drop most of them. At ~1% risk
    # per trade this still caps total risk at roughly 8% of the account.
    "max_concurrent_positions": 8,
    # ...but no more than 5 of them may point the SAME WAY. Alts are highly correlated,
    # so 8 shorts is nearly one 8x short: on 2026-08-11 all eight live trades were shorts
    # and all eight stopped. Measured over 373 days / 1,791 gated signals, capping the
    # same side at 5 leaves per-trade expectancy almost unchanged (+0.660R -> +0.629R)
    # but cuts the worst day from -7.4R to -5.3R and max drawdown from -13.9R to -10.0R.
    # -7.4R at 1% risk per trade is -7.4% in a day, which trips the 6% kill-switch below;
    # the cap keeps a bad day inside the budget instead of ending the trading day.
    "max_same_side_positions": 5,
    "max_daily_drawdown_pct": 6.0,    # real money: halt the day much sooner
    "default_leverage": 5,
    "max_loss_usd": 0.80,
}
# The bot now sizes EACH trade by its own conviction (see conviction.py): the qty,
# stop and target are derived from how good it reads the setup, instead of one fixed
# rule. So there's no static MIN_RR gate any more — conviction sets the reward:risk
# (>= 1.85 by construction, further on high-conviction trades) and the dollar risk
# (limited to RISK_MIN..RISK_MAX). The ML/news/regime/risk gates still apply.


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(msg: str) -> None:
    line = f"{_now()}  {msg}"
    print(line)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass


class PaperBroker:
    """Simulated fills at the given price, with Bybit taker fee + slippage."""
    fee = config.BACKTEST["taker_fee"]
    slip = config.BACKTEST["slippage_bps"] / 10_000.0

    def market(self, price: float, side: int) -> float:
        return price * (1 + self.slip * side)  # slippage against us


class QuantRunner:
    def __init__(self, equity: float = START_EQUITY, mode: str = "paper"):
        journal.init_db()
        # "paper"       internal simulator
        # "tradingview" real orders on the TradingView paper account (via CDP)
        # "bybit"       real orders on the user's Bybit account (demo/testnet/mainnet)
        self.mode = mode
        self.broker = PaperBroker()
        self.tv: TradingViewBroker | None = None
        self.ex = None          # the live execution broker, whichever it is
        self.exchange_info = {}  # what the cockpit shows about the connection
        self.real_equity = 0.0   # the exchange's actual balance (bybit mode)
        self._real_start = 0.0   # balance the notional book was anchored to
        if mode == "tradingview":
            self.tv = TradingViewBroker()
            self.ex = self.tv
            tv_eq = self.tv.equity()
            self.equity = tv_eq if tv_eq else self._load_equity(equity)
            self.risk = RiskManager(self.equity, TV_RISK)
        elif mode == "bybit":
            self.ex = BybitBroker(leverage=BYBIT_RISK["default_leverage"])
            ping = self.ex.ping()
            self.exchange_info = ping
            if not ping.get("authenticated"):
                _log(f"[bybit] NOT CONNECTED: {ping.get('error')} — "
                     f"run `python setup_bybit.py` to configure keys. Bot will not place orders.")
            else:
                _log(f"[bybit] connected to {ping['env'].upper()} "
                     f"equity ${ping.get('equity')} positions {ping.get('open_positions')}"
                     + ("  *** REAL MONEY ***" if self.ex.live_money else ""))
            bb_eq = self.ex.equity()
            self.real_equity = bb_eq if bb_eq else 0.0
            self._real_start = self._load_real_start(self.real_equity)
            self.book_start_ms = self._load_book_start()
            self._realised = 0.0     # PnL banked since the anchor (refreshed from Bybit)
            self._unrealised = 0.0   # open-position PnL right now
            self._book_synced = False   # has the exchange confirmed those two numbers?
            self.activity_log = []      # _sync_book reports into it, before the block below
            self._sync_book(boot=True)
            self.equity = self._book_equity()
            self.risk = RiskManager(self.equity, BYBIT_RISK)
        else:
            self.equity = self._load_equity(equity)
            self.risk = RiskManager(self.equity)
        # Real starting equity for honest PnL — persisted, NOT a hardcoded 1000.
        # On a fresh account (reset to $12) this becomes 12, so pnl reads correctly.
        # In Bybit mode the notional book always starts at BYBIT_CAPITAL — reading the
        # persisted start would inherit the $12 TradingView anchor and report nonsense PnL.
        # In Bybit mode PnL is measured from where this book started: the configured
        # capital, or (running the whole account) the balance we first saw.
        self.start_equity = ((BYBIT_CAPITAL if BYBIT_CAPITAL is not None else self._real_start)
                             if mode == "bybit" else self._load_start_equity(self.equity))
        # The day's loss budget and cooldown. Stateless by design — it reads the
        # journal every time it is asked, so a restart mid-session cannot hand the
        # bot a fresh budget after it has already lost twice today.
        self.dayguard = playbook.DayGuard()
        self._day_block: tuple[bool, str] = (False, "")   # refreshed once per tick
        self.running = False
        self._thread: threading.Thread | None = None
        self.last_tick = None
        self.last_note = f"idle ({mode})"
        self.watching: list[dict] = []      # what the bot is looking at right now
        # Rolling feed of what it's doing (newest last). NOT reset to [] here: the
        # Bybit branch above already reports the history it recovered on boot into it.
        self.activity_log: list[str] = getattr(self, "activity_log", [])
        self.scans = 0                      # how many full market scans done
        self.tv_view: dict = {}             # the live TradingView chart the bot is driving
        self.mkt_bias = 0                   # BTC market-regime gate (+1 up / -1 down / 0 neutral)
        self.forecasts: list[dict] = []     # the bot's read on the NEXT move, per coin
        self._fc_cache: dict[str, object] = {}   # per-tick memo (cleared each tick)
        self._orphans: set[str] = set()     # open positions belonging to a DIFFERENT venue
        self._btc_df = None                 # BTC candles for context features (per tick)
        self.chart: dict = {}               # the annotated chart the bot is reading
        self._last_row: dict = {}           # feature row from the most recent _score()

    # ---- persistence ----
    def _load_equity(self, default: float) -> float:
        if STATE_PATH.exists():
            try:
                return json.loads(STATE_PATH.read_text())["equity"]
            except Exception:  # noqa: BLE001
                pass
        return default

    def _load_real_start(self, current: float) -> float:
        """The exchange balance the notional book was anchored to. Persisted, so a
        restart doesn't re-anchor and wipe out the PnL earned so far."""
        if STATE_PATH.exists():
            try:
                rs = json.loads(STATE_PATH.read_text()).get("bybit_real_start")
                if rs:
                    return float(rs)
            except Exception:  # noqa: BLE001
                pass
        return current

    def _load_book_start(self) -> int:
        """Epoch-ms the notional book started at. Env wins (it survives a wiped disk),
        then the state file, then 'now' for a book that has genuinely just begun."""
        if BOOK_START_ENV:
            try:
                if BOOK_START_ENV.isdigit():
                    return int(BOOK_START_ENV)
                iso = BOOK_START_ENV.replace("Z", "+00:00")
                return int(datetime.fromisoformat(iso).timestamp() * 1000)
            except Exception:  # noqa: BLE001
                _log(f"[book] could not read QUANT_BOOK_START={BOOK_START_ENV!r} — ignoring")
        if STATE_PATH.exists():
            try:
                v = json.loads(STATE_PATH.read_text()).get("book_start_ms")
                if v:
                    return int(v)
            except Exception:  # noqa: BLE001
                pass
        return int(time.time() * 1000)

    @staticmethod
    def _book_trades(closed: list[dict]) -> tuple[list[dict], list[dict]]:
        """Split the exchange's closed trades into this book's and everything else.

        Only meaningful when the bot trades a notional book (QUANT_CAPITAL is a
        number); running the whole account, every trade on it is the account's.
        """
        if BYBIT_CAPITAL is None:
            return closed, []
        limit = FOREIGN_NOTIONAL_X * BYBIT_CAPITAL
        mine = [r for r in closed if abs(r["entry"] * r["qty"]) <= limit]
        return mine, [r for r in closed if abs(r["entry"] * r["qty"]) > limit]

    def _sync_book(self, boot: bool = False) -> None:
        """Re-read the book's PnL — and its trade history — from the EXCHANGE.

        This is the single place that makes the dashboard honest on a host with an
        ephemeral disk. Bybit keeps every closed trade; we ask it for everything since
        the anchor, sum the realised PnL (that is the book's equity), and refill
        journal.db with any trade the local file has lost. Both come from one walk of
        the same endpoint, so it costs one paginated read per tick.
        """
        if self.mode != "bybit" or self.ex is None:
            return
        if not self.exchange_info.get("authenticated"):
            return
        try:
            closed = self.ex.closed_pnl_since(self.book_start_ms)
        except Exception as e:  # noqa: BLE001
            _log(f"[book] closed-pnl read failed, keeping last figures: {e}")
            return
        closed, foreign = self._book_trades(closed)
        exch_realised = round(sum(r["pnl"] for r in closed), 6)
        self._unrealised = self.ex.unrealised()
        self._book_synced = True
        if foreign and boot:
            _log(f"[book] ignored {len(foreign)} trade(s) too large to be this book's "
                 f"(${sum(abs(r['pnl']) for r in foreign):,.0f} of PnL from the account's "
                 f"full-account period)")
        try:
            still_open = {p["symbol"] for p in journal.open_positions()}
            added = journal.import_exchange_trades(closed, skip_symbols=still_open)
        except Exception as e:  # noqa: BLE001
            _log(f"[book] history import failed: {e}")
            added = 0
        if added:
            msg = (f"RECOVERED {added} past trade(s) from Bybit — the local journal had "
                   f"been wiped (ephemeral disk). History and balance are rebuilt from "
                   f"the exchange.")
            _log(msg)
            self._activity(msg)
        # Realised PnL is read back OUT of the journal, not from the sum above, so the
        # equity and the trade table can never tell the user two different stories.
        # The journal was just refilled from the exchange, so it holds those trades
        # plus any the exchange has already aged out of its own closed-PnL window
        # (Bybit's demo history is short — it had already dropped this book's first
        # two trades). The exchange sum is the fallback if the journal is empty.
        self._realised = journal.realised_since(self.book_start_ms) or exch_realised
        if boot:
            anchor = datetime.fromtimestamp(self.book_start_ms / 1000, timezone.utc)
            _log(f"[book] anchored {anchor.isoformat(timespec='minutes')} | "
                 f"{len(closed)} closed trades since | realised ${self._realised:+.2f} "
                 f"| unrealised ${self._unrealised:+.2f}")

    def _book_equity(self) -> float:
        """The account the bot sizes against.

        BYBIT_CAPITAL is None -> the bot runs the WHOLE account: book equity IS the
        exchange balance. This is the normal mode.

        BYBIT_CAPITAL set -> a notional book of that size, whose PnL is the account's
        REALISED PnL since the anchor plus what is open right now. Rebuilt from the
        exchange every tick, so a restart cannot reset it to the starting figure — the
        bug that made the dashboard read a flat $100 no matter what the bot had done.
        """
        if BYBIT_CAPITAL is None:
            return round(self.real_equity, 4)
        if self.mode == "bybit" and getattr(self, "_book_synced", False):
            return round(BYBIT_CAPITAL + self._realised + self._unrealised, 4)
        # Fallback for an unauthenticated/offline start: the old balance-delta anchor.
        return round(BYBIT_CAPITAL + (self.real_equity - self._real_start), 4)

    def _load_start_equity(self, default: float) -> float:
        """The equity the account STARTED at (for true PnL). Read from state if it
        was set (e.g. by reset_account.py → 12.0); otherwise anchor to current."""
        if STATE_PATH.exists():
            try:
                se = json.loads(STATE_PATH.read_text()).get("start_equity")
                if se and se != 1000.0:   # 1000 was the old bogus default — ignore it
                    return float(se)
            except Exception:  # noqa: BLE001
                pass
        return round(default, 2)

    def _write_state(self, sentiment) -> None:
        from ml.meta import MODEL_PATH
        stats = journal.stats()
        st = {
            "updated": _now(),
            "running": self.running,
            "equity": round(self.equity, 2),
            "start_equity": round(self.start_equity, 2),
            "pnl": round(self.equity - self.start_equity, 2),
            "daily_dd_pct": self.risk.daily_drawdown_pct(),
            "halted": self.risk.halted,
            "open_positions": journal.open_positions(),
            "stats": stats,
            "market_sentiment": {"score": sentiment.score, "risk_off": sentiment.risk_off,
                                 "n": sentiment.n_headlines},
            "market_bias": {1: "BTC UP", -1: "BTC DOWN", 0: "BTC flat"}[self.mkt_bias],
            "headlines": news.recent_headlines(8),
            "watching": self.watching[:12],
            "activity": self.activity_log[-15:][::-1],   # newest first
            "scans": self.scans,
            "learning": {
                "live_trades_logged": stats.get("trades", 0),
                "model_active": MODEL_PATH.exists(),
                "lessons_count": len(journal.lessons(99)),
            },
            "lessons": journal.lessons(5),
            "last_note": self.last_note,
            "tv_view": self.tv_view,
            "chart": self.chart,
            "mode": self.mode,
            "exchange": self.exchange_info,
            "forecasts": self.forecasts,
            "orphan_positions": sorted(self._orphans),
            # Keep the two numbers distinct and both visible: what the bot trades as,
            # and what the exchange actually holds.
            "trading_capital": (BYBIT_CAPITAL if BYBIT_CAPITAL is not None
                                else round(self.real_equity, 2)) if self.mode == "bybit" else None,
            "manages_full_account": self.mode == "bybit" and BYBIT_CAPITAL is None,
            "real_balance": round(self.real_equity, 2) if self.mode == "bybit" else None,
            "bybit_real_start": self._real_start if self.mode == "bybit" else None,
            # The book's PnL, split into what is banked and what is still open, both
            # re-read from Bybit every tick. `book_start_ms` is the anchor they are
            # measured from and is what has to survive a restart — not the equity.
            "book_start_ms": getattr(self, "book_start_ms", None) if self.mode == "bybit" else None,
            "realised_pnl": round(getattr(self, "_realised", 0.0), 2) if self.mode == "bybit" else None,
            "unrealised_pnl": round(getattr(self, "_unrealised", 0.0), 2) if self.mode == "bybit" else None,
            "book_synced": getattr(self, "_book_synced", False) if self.mode == "bybit" else None,
            "today": journal.day_summary(),
            "days": journal.daily_history(14),
            "playbook": {
                "max_daily_losses": playbook.MAX_DAILY_LOSSES,
                "cooldown_h": playbook.LOSS_COOLDOWN_H,
                "blocked": self._day_block[0],
                "block_reason": self._day_block[1],
                "ev_min": EV_MIN,
                "min_stop_pct": playbook.MIN_STOP_PCT,
                "min_atr_pctile": playbook.MIN_ATR_PCTILE,
                "partial": f"{PARTIAL_FRAC:.0%} at +{PARTIAL_AT_R}R" if USE_PARTIAL else "off",
                "session_skip_utc": list(playbook.SKIP_HOURS_UTC),
            },
        }
        if self.mode == "bybit":
            # Written on EVERY tick, not just when it changes: this file is rewritten
            # whole, so leaving the key out would erase an operator's setting.
            saved = _saved_capital()
            if saved != "__none__":
                st["capital_override"] = saved
        STATE_PATH.write_text(json.dumps(st, indent=2))

    # ---- position management ----
    def _manage(self, sym: str, df, a) -> None:
        pos = next((p for p in journal.open_positions() if p["symbol"] == sym), None)
        if pos is None:
            return
        # A position can only be managed by the venue it actually lives on. Running in
        # Bybit mode must not send a close for a TradingView position: the Bybit close
        # would return "already flat" (true — it was never there), and the trade would
        # be booked as closed at $0 while still sitting open on the other account.
        if not self._owns(pos):
            self._orphans.add(sym)
            return
        price = float(df["close"].iloc[-1])
        side = pos["side"]
        atr_now = float(a.iloc[-1])
        exit_price = None
        why = ""
        if side == 1 and price <= pos["stop"]:
            exit_price, why = pos["stop"], "stop"
        elif side == -1 and price >= pos["stop"]:
            exit_price, why = pos["stop"], "stop"
        elif side == 1 and price >= pos["target"]:
            exit_price, why = pos["target"], "target"
        elif side == -1 and price <= pos["target"]:
            exit_price, why = pos["target"], "target"

        if exit_price is not None:
            if self.ex is not None:
                # Real broker (TradingView paper or Bybit): close, then take the PnL
                # from the equity the account actually reports — not our own estimate,
                # so fees and real slippage are included honestly.
                eq_before = self.ex.equity() or self.equity
                res = self.ex.close(sym)
                if not res["ok"]:
                    self.last_note = f"WARN: {self.mode} close failed for {sym} - {res['raw']}"
                    _log(self.last_note)
                    return
                eq_after = self.ex.equity() or eq_before
                pnl = eq_after - eq_before      # real PnL, from the real balance
                fill = exit_price
                if self.mode == "bybit":
                    self.real_equity = eq_after
                    self._sync_book()      # the close just changed realised PnL
                    self.equity = self._book_equity()
                else:
                    self.equity = eq_after
            else:
                fill = self.broker.market(exit_price, -side)
                pnl = (fill - pos["entry"]) * pos["qty"] * side
                pnl -= abs(fill * pos["qty"]) * self.broker.fee
                self.equity += pnl
            r = pnl / (abs(pos["entry"] - pos["stop"]) * pos["qty"]) if pos["qty"] else None
            rec = journal.record_close(sym, fill, round(pnl, 4), round(r, 2) if r else None, why)
            self.risk.update_equity(self.equity)
            note = f"CLOSE {sym} {why} pnl ${pnl:+.2f} -> eq ${self.equity:.2f}"
            if rec.get("postmortem"):
                note += f" | lesson: {rec['postmortem']}"
            self.last_note = note
            _log(note)
            return

        # --- bank a third at +0.5R, then make the trade free ---
        # THE SINGLE BIGGEST CHANGE TO HOW OFTEN THIS BOT LOSES. Holding every trade
        # whole to a 2R target means the market has to be right about the setup twice
        # over; taking a slice at +0.5R and pulling the stop to breakeven means it only
        # has to be right once, and the trade that then reverses costs nothing instead
        # of a full R. Measured over 197 days on walk-forward-gated signals
        # (research_manage.py), on the same entries:
        #     hold to target, stop at BE after 1R (what ran before)   32% win  +0.063R
        #     bank 1/2 at +1R, stop to BE                             55% win  +0.055R
        #     bank 1/3 at +0.5R, stop to BE   <- this                 70% win  +0.062R
        # and with the daily loss budget on top, that last line is 80% win / +0.321R.
        # A third, not a half: banking half caps the winners hard enough to cost
        # +0.05R a trade (+0.185 vs +0.217 in the same test), and two thirds is worse
        # again (+0.153). Trailing the remainder and time-stopping it were both tested
        # here and are worth nothing either way (±0.002R), so the runner is left alone
        # to reach the target the model was scored against.
        risk_unit = abs(pos["entry"] - pos["stop"])
        move = (price - pos["entry"]) * side
        if (USE_PARTIAL and risk_unit and not pos.get("partial")
                and move >= PARTIAL_AT_R * risk_unit and pos["qty"]):
            self._take_partial(sym, pos, price, risk_unit)
            return

        # --- trailing stop ---  (only ONCE the trade is solidly in profit)
        # Until +BE_AFTER_R the original stop holds (don't choke a fresh trade). After
        # that: lock breakeven, then trail TRAIL_ATR_MULT ATRs behind price. The stop is
        # only ever moved in the FAVOURABLE direction — it never loosens — so this turns
        # a would-be loser into break-even and lets a runner keep running.
        if not USE_TRAIL:
            return
        if not risk_unit or move < BE_AFTER_R * risk_unit:
            return
        new_stop = pos["entry"]                                  # breakeven floor
        trail = price - side * TRAIL_ATR_MULT * atr_now
        new_stop = max(new_stop, trail) if side == 1 else min(new_stop, trail)
        improved = (new_stop > pos["stop"]) if side == 1 else (new_stop < pos["stop"])
        if improved:
            # On a real exchange the stop LIVES on Bybit's servers, so move it there
            # FIRST. If that call fails, leave the journal alone — a journal that
            # disagrees with the exchange about where the stop is, is worse than a
            # stop that didn't move.
            if self.mode == "bybit" and self.ex is not None:
                if not self.ex.set_stops(sym, stop=new_stop):
                    self._activity(f"TRAIL {sym} failed to move exchange stop - keeping {pos['stop']:.4f}")
                    return
            with journal._conn() as c:
                c.execute("UPDATE positions SET stop=? WHERE symbol=?", (new_stop, sym))
            self._activity(f"TRAIL {sym} stop -> {new_stop:.4f} (locking profit)")

    def _take_partial(self, sym: str, pos: dict, price: float, risk_unit: float) -> None:
        """Sell PARTIAL_FRAC of the position at +PARTIAL_AT_R and pull the stop to
        breakeven, so what is left cannot turn into a loss.

        Order matters: the slice is sold FIRST and the stop moved after. If the stop
        move fails, the bot still banked the profit and the trade keeps its original
        (valid) stop — the reverse order could leave the exchange protecting a
        quantity that no longer exists.
        """
        side = pos["side"]
        want = pos["qty"] * PARTIAL_FRAC
        fill, closed_qty = price, want
        if self.mode == "bybit" and self.ex is not None:
            res = self.ex.reduce(sym, want)
            if not res["ok"]:
                # Almost always the exchange's minimum order size on a small book:
                # a third of a $60 position can be below it. Not an error — the trade
                # simply runs whole, and the breakeven stop below still protects it.
                self._activity(f"PARTIAL {sym.replace('USDT','')} skipped - {res['raw']}")
                with journal._conn() as c:
                    c.execute("UPDATE positions SET partial=1 WHERE symbol=?", (sym,))
                self._move_stop_to_be(sym, pos)
                return
            closed_qty = res["qty"] or want
            fill = self.ex.last_price(sym) or price
        pnl = (fill - pos["entry"]) * closed_qty * side
        r = pnl / (risk_unit * closed_qty) if closed_qty and risk_unit else None
        journal.record_partial(sym, fill, closed_qty, round(pnl, 4),
                               round(r, 2) if r is not None else None,
                               f"partial +{PARTIAL_AT_R}R ({PARTIAL_FRAC:.0%} banked)")
        self._move_stop_to_be(sym, pos)
        note = (f"PARTIAL {sym.replace('USDT','')} banked {PARTIAL_FRAC:.0%} at "
                f"{fill:.4f} (+${pnl:.2f}) - stop to breakeven, rest runs free")
        self.last_note = note
        self._activity(note)
        _log(note)

    def _move_stop_to_be(self, sym: str, pos: dict) -> None:
        """Breakeven stop on the remainder. On Bybit the stop lives on the exchange,
        so it is moved there first; the journal only follows a move that landed."""
        be = pos["entry"]
        improved = (be > pos["stop"]) if pos["side"] == 1 else (be < pos["stop"])
        if not improved:
            return
        if self.mode == "bybit" and self.ex is not None:
            if not self.ex.set_stops(sym, stop=be):
                self._activity(f"BE {sym.replace('USDT','')} - exchange stop move failed, "
                               f"keeping {pos['stop']:.4f}")
                return
        with journal._conn() as c:
            c.execute("UPDATE positions SET stop=? WHERE symbol=?", (be, sym))

    # ---- the chart the bot is reading ----
    def _build_chart(self, plans: list[dict], watching: list[dict]) -> None:
        """Draw what the bot is looking at, the way a trader would.

        The bot used to be able to drive a real TradingView chart over CDP, but that
        needs TradingView running on a desktop — impossible on a cloud host. So it
        draws its own: candles, the moving averages it trades around, and the trend
        lines / range / support-resistance it has actually detected (features/
        structure.py), plus the entry, stop and target of the plan.

        Focus order — open position first, then the best live setup, then the coin
        closest to firing. That is the same order a person would look in.
        """
        try:
            from features.structure import analyse, to_drawings
            pos = journal.open_positions()
            mine = [p for p in pos if self._owns(p)]
            focus, kind, plan = None, "WATCH", None
            if mine:
                focus, kind = mine[0]["symbol"], "TRADE"
                plan = {"entry": mine[0]["entry"], "stop": mine[0]["stop"],
                        "target": mine[0]["target"],
                        "side": "LONG" if mine[0]["side"] == 1 else "SHORT"}
            elif plans:
                best = max(plans, key=lambda p: p.get("ev", -9))
                # Only call it a SETUP if it would actually be traded. A signal the
                # EV gate rejects is something the bot is watching, not planning.
                focus = best["sym"]
                kind = "SETUP" if best.get("ev", -9) >= EV_MIN else "WATCH"
                plan = {"entry": best["entry"], "stop": best["stop"],
                        "target": best["target"],
                        "side": "LONG" if best["side"] == 1 else "SHORT"}
            elif watching:
                focus = watching[0]["symbol"] + "USDT"
            if not focus:
                return

            df = get_klines_cached(focus, DECISION_INTERVAL, bars=SIGNAL_BARS, max_age_min=5)
            feats = feature_frame(df)
            st = analyse(df, lookback=CHART_BARS)
            d = df.iloc[-CHART_BARS:]
            f = feats.iloc[-CHART_BARS:]
            self.chart = {
                "symbol": focus.replace("USDT", ""),
                "interval": DECISION_INTERVAL,
                "kind": kind,
                "plan": plan,
                "candles": [
                    {"t": str(ix)[5:16], "o": float(r.open), "h": float(r.high),
                     "l": float(r.low), "c": float(r.close)}
                    for ix, r in zip(d.index, d.itertuples())
                ],
                "ema20": [None if v != v else round(float(v), 8) for v in f["ema20"]],
                "ema50": [None if v != v else round(float(v), 8) for v in f["ema50"]],
                "drawings": to_drawings(st, d),
                "notes": st.notes,
                "updated": _now(),
            }
        except Exception as e:  # noqa: BLE001
            _log(f"[chart] {e}")

    # ---- operator commands ----------------------------------------------------
    # Everything below is ONLY ever called from an explicit operator instruction
    # (cockpit chat / API). The trading loop never calls these — it keeps deciding
    # entries and exits on its own. That separation is the point: the bot trades
    # autonomously, the operator can still intervene.

    def close_now(self, sym: str, why: str = "manual close (operator)") -> dict:
        """Flatten one position because the operator said so.

        Deliberately shares the accounting of `_manage`'s exit path — PnL comes from
        the exchange (realised PnL if it reports one, otherwise the balance delta),
        never from our own estimate, and the journal is only written once the close
        is VERIFIED. A close that failed must not be recorded as a closed trade.
        """
        sym = sym.upper()
        if not sym.endswith("USDT"):
            sym += "USDT"
        pos = next((p for p in journal.open_positions() if p["symbol"] == sym), None)
        if pos is None:
            return {"ok": False, "symbol": sym, "msg": f"{sym}: koi open position nahi hai"}
        if not self._owns(pos):
            return {"ok": False, "symbol": sym,
                    "msg": f"{sym} is on venue '{pos.get('venue') or 'legacy'}', "
                           f"bot abhi '{self.mode}' pe hai — isko close nahi kar sakta"}

        if self.ex is None:                       # internal simulator
            price = pos["entry"]
            try:
                price = float(get_klines_cached(sym, DECISION_INTERVAL, bars=50,
                                                max_age_min=1)["close"].iloc[-1])
            except Exception:  # noqa: BLE001
                pass
            fill = self.broker.market(price, -pos["side"])
            pnl = (fill - pos["entry"]) * pos["qty"] * pos["side"]
            pnl -= abs(fill * pos["qty"]) * self.broker.fee
            self.equity += pnl
        else:
            eq_before = self.ex.equity() or self.equity
            res = self.ex.close(sym)
            if not res["ok"]:
                msg = f"{sym} close FAILED: {res['raw']}"
                _log(msg)
                self._activity(msg)
                return {"ok": False, "symbol": sym, "msg": msg}
            eq_after = self.ex.equity() or eq_before
            pnl = eq_after - eq_before
            fill = pos["entry"]
            if self.mode == "bybit":
                # The exchange's own realised figure is the honest one (fees included);
                # the balance delta is only a fallback if it hasn't settled yet.
                realised = self.ex.closed_pnl(sym, limit=3)
                if realised:
                    pnl, fill = realised[0]["pnl"], realised[0]["exit"]
                else:
                    fill = self.ex.last_price(sym) or pos["entry"]
                self.real_equity = eq_after
                self.equity = self._book_equity()
            else:
                self.equity = eq_after

        risk_unit = abs(pos["entry"] - pos["stop"]) * pos["qty"]
        r = pnl / risk_unit if risk_unit else None
        rec = journal.record_close(sym, fill, round(pnl, 4), round(r, 2) if r else None, why)
        self.risk.update_equity(self.equity)
        note = f"MANUAL CLOSE {sym} pnl ${pnl:+.2f} -> eq ${self.equity:.2f}"
        if rec.get("postmortem"):
            note += f" | lesson: {rec['postmortem']}"
        self.last_note = note
        _log(note)
        self._activity(note)
        return {"ok": True, "symbol": sym, "pnl": round(pnl, 4), "msg": note}

    def close_all(self, why: str = "manual close-all (operator)") -> list[dict]:
        """Flatten every position this venue owns. Positions belonging to another
        venue are reported, never touched."""
        out = []
        for pos in journal.open_positions():
            out.append(self.close_now(pos["symbol"], why))
        return out

    def set_capital(self, value: float | None) -> dict:
        """Change the book the bot sizes against.

        IMPORTANT and easy to misread: this does NOT move money on the exchange.
        Bybit gives a demo account a fixed balance and its API has no "set balance
        to X" call, so `set_capital(100)` makes the bot trade AS IF it had $100 —
        every position size, the exposure cap and the daily kill-switch run off that
        figure — while the exchange balance stays whatever Bybit says it is.
        `None` = go back to managing the whole account balance.
        """
        global BYBIT_CAPITAL
        if self.mode != "bybit":
            return {"ok": False, "msg": f"capital sirf bybit mode me set hota hai (abhi: {self.mode})"}
        if value is not None and value <= 0:
            return {"ok": False, "msg": "capital 0 se bada hona chahiye"}

        BYBIT_CAPITAL = None if value is None else float(value)
        if self.ex is not None:
            self.real_equity = self.ex.equity() or self.real_equity
        # Re-anchor: the new book starts NOW, so past PnL isn't double-counted into it.
        self._real_start = self.real_equity
        self.equity = self._book_equity()
        self.start_equity = BYBIT_CAPITAL if BYBIT_CAPITAL is not None else self._real_start
        was_halted = self.risk.halted          # a resize must not quietly cancel a kill-switch
        self.risk.start_equity = self.start_equity
        self.risk.day_start_equity = self.equity
        self.risk.update_equity(self.equity)
        self.risk.halted = was_halted
        self._save_capital()

        if BYBIT_CAPITAL is None:
            msg = f"Ab bot PURA account manage karega (balance ${self.real_equity:,.2f})"
        else:
            msg = (f"Trading capital set to ${BYBIT_CAPITAL:,.2f}. "
                   f"Exchange balance ${self.real_equity:,.2f} waisa hi hai (Bybit demo "
                   f"balance API se badla nahi ja sakta) — bot ab ${BYBIT_CAPITAL:,.0f} "
                   f"ke account jaisa size, risk aur kill-switch use karega.")
        _log(f"CAPITAL -> {BYBIT_CAPITAL if BYBIT_CAPITAL is not None else 'full account'} (operator)")
        self._activity(f"OPERATOR set trading capital -> "
                       f"{'full account' if BYBIT_CAPITAL is None else f'${BYBIT_CAPITAL:,.0f}'}")
        return {"ok": True, "capital": BYBIT_CAPITAL, "real_balance": round(self.real_equity, 2),
                "msg": msg}

    def _save_capital(self) -> None:
        """Persist the override immediately (not only on the next tick) so a crash or
        redeploy right after the command doesn't lose it."""
        st = {}
        if STATE_PATH.exists():
            try:
                st = json.loads(STATE_PATH.read_text())
            except Exception:  # noqa: BLE001
                st = {}
        st["capital_override"] = "full" if BYBIT_CAPITAL is None else BYBIT_CAPITAL
        st["bybit_real_start"] = self._real_start
        try:
            STATE_PATH.write_text(json.dumps(st, indent=2))
        except Exception as e:  # noqa: BLE001
            _log(f"[capital] could not persist override: {e}")

    # ---- venue ownership ----
    def _owns(self, pos: dict) -> bool:
        """Is this journal position on the venue this runner is trading?

        Positions carry a `venue` since 2026-08-06. Rows written before that are
        tagged 'legacy'/None; those pre-date Bybit mode entirely, so they belong to
        whichever non-Bybit venue is running and are never claimed by Bybit mode.
        """
        venue = pos.get("venue") or "legacy"
        if venue == self.mode:
            return True
        return venue == "legacy" and self.mode != "bybit"

    # ---- exchange reconciliation ----
    def _reconcile(self) -> None:
        """Close, in our journal, any position the EXCHANGE has already closed.

        This is the piece paper trading never needs. On Bybit the stop-loss and
        take-profit sit on the exchange, so a wick at 3am closes the trade while this
        process is asleep between ticks. Without reconciliation the journal would
        still think the position is open — it would refuse to re-enter that symbol,
        report a fantasy equity, and never learn from the trade. The exchange is the
        source of truth; we make our books match it, and use ITS realised PnL.
        """
        if self.mode != "bybit" or self.ex is None:
            return
        # Guard 1: never reconcile against an account we aren't even logged into.
        if not self.exchange_info.get("authenticated"):
            return
        # Guard 2: `fetch_positions` RAISES on a failed read. The convenience
        # `positions()` would hand back [] instead, and an empty list here means
        # "close everything" — an auth hiccup or a network blip would wipe the book.
        try:
            live = {p["symbol"] for p in self.ex.fetch_positions()}
        except Exception as e:  # noqa: BLE001
            _log(f"[reconcile] could not read positions, skipping: {e}")
            return
        # --- ADOPT: positions the exchange has that our journal does not ---
        # On a cloud host the disk is ephemeral, so a redeploy or restart wipes
        # journal.db while the position keeps living on Bybit. Without this the bot
        # would never manage or close that trade, and would happily open a second one
        # in the same symbol. The exchange is the source of truth in BOTH directions.
        known = {p["symbol"] for p in journal.open_positions()}
        for p in self.ex.positions():
            if p["symbol"] in known:
                continue
            side = 1 if p["side"] == "long" else -1
            stop = p.get("stop") or p["avgFill"] * (1 - 0.02 * side)
            target = p.get("target") or p["avgFill"] * (1 + 0.04 * side)
            journal.record_open(p["symbol"], side, p["avgFill"], p["qty"], stop, target,
                                "ADOPTED from exchange after restart", {}, venue="bybit")
            self._activity(f"ADOPTED {p['symbol'].replace('USDT','')} "
                           f"{'LONG' if side == 1 else 'SHORT'} from Bybit "
                           f"(stop {stop:.4f}) - journal was out of date")
            _log(f"ADOPTED {p['symbol']} from exchange (qty {p['qty']} @ {p['avgFill']})")

        for pos in journal.open_positions():
            sym = pos["symbol"]
            # Guard 3: only positions that actually LIVE on Bybit. A TradingView or
            # simulator position is obviously not on Bybit; closing it because Bybit
            # has never heard of it destroys the record of a real trade elsewhere.
            if not self._owns(pos):
                continue
            if sym in live:
                continue
            realised = self.ex.closed_pnl(sym, limit=5)
            if realised:
                r0 = realised[0]
                pnl, exit_px = r0["pnl"], r0["exit"]
                why = "exchange stop/target"
            else:
                pnl, exit_px, why = 0.0, pos["entry"], "closed on exchange (pnl unknown)"
            risk_unit = abs(pos["entry"] - pos["stop"]) * pos["qty"]
            r = pnl / risk_unit if risk_unit else None
            ext = f"{sym}:{realised[0]['closed_ms']}" if realised else None
            rec = journal.record_close(sym, exit_px, round(pnl, 4),
                                       round(r, 2) if r else None, why, ext_id=ext)
            self.real_equity = self.ex.equity() or self.real_equity
            self._sync_book()
            self.equity = self._book_equity()
            self.risk.update_equity(self.equity)
            note = f"SYNC {sym} closed on Bybit ({why}) pnl ${pnl:+.2f} -> eq ${self.equity:.2f}"
            if rec.get("postmortem"):
                note += f" | lesson: {rec['postmortem']}"
            self.last_note = note
            _log(note)

    # ---- forward-looking read ----
    def _forecast_for(self, sym: str):
        """The bot's next-move read for one symbol, memoised for this tick.

        Runs on the forecaster's OWN timeframe (15m), which is where the measured
        edge lives — not on the strategy's decision timeframe. Returns None if the
        layer is off or the data call fails; callers must treat None as 'no opinion',
        never as a reason to block a trade.
        """
        if not USE_FORECAST:
            return None
        if sym in self._fc_cache:
            return self._fc_cache[sym]
        try:
            fdf = get_klines_cached(sym, MODEL_INTERVAL, bars=300, max_age_min=10)
            fc = next_move(sym, fdf, feature_frame(fdf))
        except Exception as e:  # noqa: BLE001
            _log(f"[forecast {sym}] {e}")
            fc = None
        self._fc_cache[sym] = fc
        return fc

    @staticmethod
    def _fwd_agreement(fc, side: int) -> float:
        """How strongly the forward evidence backs THIS side, in [-1, +1].
        Blends the trained forecast with the live microstructure bias it carries."""
        if fc is None:
            return 0.0
        model_term = (fc.prob_up - 0.5) * 2 * side     # +-1 at the model's extremes
        ms_term = fc.ms_bias * side
        return float(max(-1.0, min(1.0, 0.6 * model_term + 0.4 * ms_term)))

    # ---- entry ----
    def _score(self, df, feats, a, sig) -> tuple[float, float, float]:
        """(win probability, natural reward:risk, expected value in R) for a setup.

        One function so the number shown on the dashboard is exactly the number the
        entry gate uses — the "it showed a signal but didn't trade" confusion came
        from those two being computed in different places.

        STALE SETUPS. A signal from an earlier bar is still valid while price sits
        between its stop and target, but if price has drifted almost onto the stop,
        the remaining risk is a rounding error and the reward:risk explodes — a live
        ETH setup priced out at 32:1, while the widest RR in a year of training data
        is 3:1. Trading that is not "high reward", it is a stop sitting inside the
        spread, and fees alone (2*cost/stop_pct) would cost several R. Those are
        rejected, and RR is capped at what the model was actually trained on.

        The exact feature row that produced the score is left in `self._last_row`
        so the caller can journal it as the trade's entry context. It is the same row
        the model saw — a post-mortem written from a re-derived row would be
        describing a slightly different trade.
        """
        price = float(df["close"].iloc[-1])
        stop_d = abs(price - sig.stop)
        atr_now = float(a.iloc[-1]) if len(a) else 0.0
        rr = abs(sig.target - price) / stop_d if stop_d else 0.0
        if (stop_d <= 0 or stop_d / price < MIN_STOP_PCT
                or (atr_now and stop_d / atr_now < MIN_STOP_ATR)):
            self._last_row = {}
            return 0.0, rr, STALE_EV
        row = feats.iloc[-1][FEATURE_COLS].to_dict()
        try:
            ctx = bar_context(df, feats, self._btc_df).iloc[-1]
            row.update({k: ctx[k] for k in BAR_CONTEXT_COLS})
            row.update(signal_context(sig, price, float(a.iloc[-1]), ctx["btc_above_ema50"]))
        except Exception as e:  # noqa: BLE001
            # Context is what gives the model its edge, but a failure here must not
            # stop trading: the missing columns arrive as NaN, which the model handles.
            _log(f"[context] {e}")
        prob = predict_proba(row, sig.side)
        self._last_row = row
        return prob, rr, expected_value(prob, min(rr, RR_CAP))

    def _try_enter(self, sym: str, df, feats, a, sentiment, sig) -> None:
        if not config.is_crypto_symbol(sym):   # CRYPTO ONLY (locked per user)
            return
        if any(p["symbol"] == sym for p in journal.open_positions()):
            return
        if sig is None:  # signal on the most-recent CLOSED bar
            return
        price = float(df["close"].iloc[-1])
        if market.blocks(self.mkt_bias, sig.side):   # don't fight BTC's decisive trend
            return

        # THE ENTRY GATE: expected value, not raw win probability. Measured over a
        # year of signals (chronological split, costs included): gating on P(win)
        # is break-even at best, gating on EV is positive in both halves of the
        # period. See research_edge.py / research_gate.py.
        prob, natural_rr, ev = self._score(df, feats, a, sig)
        # The row the model actually scored — journalled below as the trade's entry
        # context, which is what the loss post-mortems and the next ML retrain read.
        feat_row = dict(self._last_row)
        if ev < EV_MIN:
            return

        # THE PLAYBOOK. Two gates the EV model cannot see: whether this SETUP is one
        # of the structurally poor kinds (stop inside the noise, against the higher
        # timeframe, dead volatility, dead session), and whether the DAY has already
        # spent its loss budget. Both were measured on 197 days of walk-forward-gated
        # signals — see playbook.py for the numbers behind each one.
        veto = playbook.entry_veto(entry=price, stop=sig.stop, side=sig.side, ctx=feat_row)
        if veto:
            self._activity(f"SKIP {sym.replace('USDT','')} - {veto}")
            return
        # Announced once per tick in tick(), not once per coin — 59 identical lines
        # would bury everything else on the activity feed.
        if self._day_block[0]:
            return
        ok_news, news_reason = news.agrees_with(sym, sig.side)
        if not ok_news or sentiment.risk_off:
            return

        # --- forward-looking check: does the next move look like it fights us? ---
        # Only a CLEAR disagreement blocks. A neutral or missing read never does —
        # this bot's old failure mode was stacking gates until nothing ever traded.
        fc = self._forecast_for(sym)
        fwd_agree = self._fwd_agreement(fc, sig.side)
        fwd_reason = "no forward read"
        if fc is not None:
            prob_against = fc.prob_up if sig.side == -1 else 1 - fc.prob_up
            if prob_against >= FORECAST_VETO:
                self._activity(f"SKIP {sym.replace('USDT','')} - forecast says "
                               f"{fc.direction} {fc.prob*100:.0f}% next {fc.horizon_text}")
                return
            if fc.ms_bias * sig.side < -MS_VETO_BIAS:
                worst = min(fc.reasons[1:], key=len) if len(fc.reasons) > 1 else "order flow against"
                self._activity(f"SKIP {sym.replace('USDT','')} - {worst}")
                return
            fwd_reason = (f"fwd {fc.direction} {fc.prob*100:.0f}%/{fc.horizon_text} "
                          f"(agree {fwd_agree:+.2f})")

        # --- the bot reads THIS setup's potential and builds its own plan ---
        # Conviction (ML win-prob + trend strength + natural RR) sets the dollar risk
        # (-> quantity), a slightly tighter stop, and a further-out target on the trades
        # it rates highly. Loss stays limited; the planned win is always >= $1.
        plan = conviction.assess(prob=prob, adx=float(feats.iloc[-1]["adx14"]),
                                 natural_rr=natural_rr, side=sig.side,
                                 entry=price, stop=sig.stop, fwd_agree=fwd_agree,
                                 equity=self.equity)
        # CONVICTION SIZES THE TRADE; THE STRATEGY STILL SETS THE LEVELS.
        # It used to also tighten the stop and stretch the target on high-conviction
        # setups. That quietly broke the EV gate: EV is p x RR for the setup's OWN
        # stop and target, and those are the levels the model's training labels were
        # measured against. Moving the target further out lowers the true probability
        # of reaching it, so the trade taken was no longer the trade that was scored.
        # Size varies with conviction; where to get out does not.
        plan.stop, plan.target = sig.stop, sig.target

        decision = self.risk.evaluate(side=sig.side, entry=price, stop=plan.stop,
                                      open_positions=journal.open_positions(),
                                      risk_usd=plan.risk_usd)
        if not decision.approved:
            # Say WHY on the feed. A setup that cleared every other gate and then
            # vanished silently is exactly the "it showed a signal but didn't trade"
            # confusion; the risk manager's reason is the answer to it.
            self._activity(f"SKIP {sym.replace('USDT','')} - {decision.reason}")
            return

        protected = False
        if self.mode == "bybit" and self.ex is not None:
            if not self.exchange_info.get("authenticated"):
                # Keep scanning and showing signals, but never pretend an order went
                # out. A silent no-op here would look exactly like "the bot won't trade".
                self._activity(f"WOULD TRADE {sym.replace('USDT','')} "
                               f"{'LONG' if sig.side == 1 else 'SHORT'} - "
                               f"Bybit not connected (run setup_bybit.py)")
                return
            # Send the stop and target WITH the order so the position is protected
            # server-side from the very first second — if this bot, the PC or the
            # internet dies, Bybit still closes the trade at the planned levels.
            res = self.ex.market(sym, sig.side, decision.qty,
                                 stop=plan.stop, target=plan.target)
            if not res["ok"]:
                self.last_note = f"WARN: Bybit order failed for {sym} - {res['raw']}"
                _log(self.last_note)
                self._activity(f"ORDER REJECTED {sym.replace('USDT','')}: {str(res['raw'])[:90]}")
                return
            fill = res["fill"] or price
            qty = res["qty"] or decision.qty
            protected = res.get("protected", False)
            if not protected:
                _log(f"WARN: {sym} opened but exchange stop NOT set - managing in software only")
        elif self.mode == "tradingview" and self.tv is not None:
            res = self.tv.market(sym, sig.side, decision.qty)
            if not res["ok"]:
                self.last_note = f"WARN: TV order failed for {sym} - {res['raw']}"
                _log(self.last_note)
                return
            fill = res["fill"] or price
            qty = res["qty"] or decision.qty
        else:
            fill = self.broker.market(price, sig.side)
            qty = decision.qty
        # Re-anchor the conviction stop/target to the ACTUAL fill, preserving their
        # distances (so a long can't open already below its own stop — the DOGE bug —
        # and the planned reward:risk is kept intact).
        stop_dist = abs(price - plan.stop)
        tgt_dist = abs(plan.target - price)
        f_stop = fill - sig.side * stop_dist
        f_target = fill + sig.side * tgt_dist
        # The SL/TP we sent were anchored to the SIGNAL price; the fill is usually a
        # little different. Push the re-anchored levels so the exchange protects the
        # trade at the same RISK DISTANCE we actually planned, not a drifted one.
        if self.mode == "bybit" and self.ex is not None and protected:
            self.ex.set_stops(sym, stop=f_stop, target=f_target)
        journal.record_open(sym, sig.side, fill, qty, f_stop, f_target,
                            f"{sig.reason} | conv {plan.conviction:.2f} rr {plan.rr} "
                            f"| ML p={prob:.2f} | {fwd_reason} | {news_reason}", feat_row,
                            venue=self.mode)
        side_txt = "LONG" if sig.side == 1 else "SHORT"
        plan_win = plan.risk_usd * plan.rr
        self.last_note = (f"OPEN {side_txt} {sym} @ {fill:.4f} qty {qty} "
                          f"stop {f_stop:.4f} tgt {f_target:.4f} "
                          f"(conv {plan.conviction:.2f}, risk ${plan.risk_usd:.2f} "
                          f"-> ~${plan_win:.2f} win, ML {prob:.2f})")
        _log(self.last_note)

    def _visualize(self, plans: list[dict], watching: list[dict]) -> None:
        """Drive the real TradingView chart to what the bot cares about most, draw
        its plan (entry/stop/target), and snapshot it — so the user SEES it work.
        Priority: an OPEN position > the best live signal > the top opportunity."""
        try:
            focus = None
            open_pos = journal.open_positions()
            if open_pos:
                p = open_pos[0]
                focus = {"sym": p["symbol"], "side": p["side"], "entry": p["entry"],
                         "stop": p["stop"], "target": p["target"],
                         "label": (p.get("reason") or "").split("|")[0].strip()[:24] or "OPEN",
                         "kind": "TRADE"}
            elif plans:
                best = max(plans, key=lambda x: x["score"])
                focus = {**best, "kind": "SETUP"}
            elif watching:
                top = watching[0]
                focus = {"sym": top["symbol"] + "USDT", "kind": "WATCH",
                         "setup": top.get("setup"), "score": top.get("score")}

            if not focus:
                return
            self.tv.set_chart(focus["sym"], "15")
            if focus["kind"] in ("TRADE", "SETUP"):
                self.tv.draw_levels(focus["sym"], focus["entry"], focus["stop"],
                                    focus["target"], focus["side"], focus["label"])
            else:
                self.tv.clear_drawings()
            shot = self.tv.screenshot("bot_view")
            self.tv_view = {
                "symbol": focus["sym"].replace("USDT", ""),
                "kind": focus["kind"],
                "side": ("LONG" if focus.get("side") == 1 else "SHORT") if focus.get("side") else None,
                "label": focus.get("label") or focus.get("setup") or "scanning",
                "shot": shot,
                "updated": _now(),
            }
        except Exception as e:  # noqa: BLE001
            _log(f"[viz] {e}")

    def _build_forecast_board(self, watching: list[dict]) -> None:
        """The 'next move' panel: a forward read on the coins that matter this tick.

        Only the top FORECAST_TOP_N are forecast, not all 24 — each one costs several
        API calls (open interest, funding, book, tape, three timeframes) and there is
        no point spending that budget on coins the bot would never trade right now.
        Anything already carrying a live signal is included regardless of rank.
        """
        if not USE_FORECAST:
            self.forecasts = []
            return
        picks, seen = [], set()
        for w in watching:
            sym = w["symbol"] + "USDT"
            if w.get("signal") or len(picks) < FORECAST_TOP_N:
                if sym not in seen:
                    picks.append(sym)
                    seen.add(sym)
        board = []
        for sym in picks:
            fc = self._forecast_for(sym)
            if fc is None:
                continue
            d = fc.as_dict()
            if not fc.model_ready:
                self.forecasts = [d]
                return
            board.append(d)
        # strongest conviction first — that is the one worth a human's attention
        board.sort(key=lambda x: -x["confidence"])
        self.forecasts = board

    def _activity(self, msg: str) -> None:
        self.activity_log.append(f"{_now()[11:19]}  {msg}")
        self.activity_log = self.activity_log[-25:]  # keep last 25

    # ---- one full pass ----
    def tick(self) -> None:
        self._fc_cache = {}          # forecasts are per-tick fresh
        self._orphans = set()
        if self.ex is not None:
            eq = self.ex.equity()
            if eq:
                if self.mode == "bybit":
                    self.real_equity = eq
                    self._sync_book()      # realised + unrealised, straight from Bybit
                    self.equity = self._book_equity()
                else:
                    self.equity = eq
        self._reconcile()            # exchange is the source of truth (bybit mode)
        self.risk.update_equity(self.equity)
        # The day's loss budget — evaluated once, after reconciliation has booked
        # anything the exchange closed while this process was between ticks.
        self._day_block = self.dayguard.blocks()
        if self._day_block[0]:
            self._activity(f"NO NEW TRADES - {self._day_block[1]}")
        sentiment = news.market_sentiment()
        self.mkt_bias = market.current_bias(DECISION_INTERVAL)   # don't fight BTC's trend
        # BTC candles once per tick: every coin's context features are measured
        # against the same market backdrop, and it is one fetch instead of 59.
        try:
            self._btc_df = get_klines_cached("BTCUSDT", DECISION_INTERVAL,
                                             bars=SIGNAL_BARS, max_age_min=4)
        except Exception as e:  # noqa: BLE001
            self._btc_df = None
            _log(f"[btc context] {e}")
        watching = []
        plans: list[dict] = []     # candidate trade plans to visualise on the TV chart
        for sym in config.UNIVERSE:
            try:
                df = get_klines_cached(sym, DECISION_INTERVAL, bars=SIGNAL_BARS, max_age_min=4)
                feats = feature_frame(df)
                a = atr(df, 14)
                self._manage(sym, df, a)
                # --- build the live "what am I looking at" view ---
                regime = trend_label(df)
                row = feats.iloc[-1]
                # TREND lenses on 1h (priority); if none, fall back to the FLAT-market
                # RANGE-FADE lens on its own higher timeframe. Whichever fires carries its
                # OWN dataframe/features so ML + price levels use the right timeframe.
                sigs = _gen_signals(df, feats, a)
                sig = _fresh_signal(sigs, float(df["close"].iloc[-1]))
                e_df, e_feats, e_a, e_price = df, feats, a, float(df["close"].iloc[-1])
                if sig is None:
                    try:
                        rdf, rfeats, ra, rsig = _range_signal(sym)
                        if rsig is not None:
                            sig, e_df, e_feats, e_a, e_price = rsig, rdf, rfeats, ra, float(rdf["close"].iloc[-1])
                    except Exception as re:  # noqa: BLE001
                        _log(f"[range {sym}] {re}")
                snap = opportunity_snapshot(df, feats, a)
                entry = {"symbol": sym.replace("USDT", ""), "regime": regime,
                         "rsi": round(float(row["rsi14"]), 0), "adx": round(float(row["adx14"]), 0),
                         "signal": None, "ml": None, "ev": None, "rr": None,
                         "setup": snap["setup"], "score": snap["score"]}
                if sig is not None:
                    entry["signal"] = "LONG" if sig.side == 1 else "SHORT"
                    # Same call the entry gate makes, so the board can never show a
                    # number the decision didn't use.
                    p_, rr_, ev_ = self._score(e_df, e_feats, e_a, sig)
                    entry["ml"] = round(p_, 2)
                    entry["rr"] = round(rr_, 2)
                    entry["ev"] = round(ev_, 2)
                    setup_tag = sig.reason.split(":")[0]
                    entry["setup"] = setup_tag           # show the ACTUAL firing setup, not the multi_angle proxy
                    if ev_ == STALE_EV:
                        why = "price already at its stop - setup expired"
                    elif ev_ >= EV_MIN:
                        why = f"win {p_*100:.0f}% x {min(rr_, RR_CAP):.1f}R -> EV {ev_:+.2f}"
                    else:
                        why = (f"win {p_*100:.0f}% x {min(rr_, RR_CAP):.1f}R -> "
                               f"EV {ev_:+.2f} < {EV_MIN} skip")
                    self._activity(f"SIGNAL {setup_tag} {entry['symbol']} "
                                   f"(ADX {entry['adx']:.0f}, RSI {entry['rsi']:.0f}, {why})")
                    plans.append({"sym": sym, "side": sig.side, "entry": e_price,
                                  "stop": sig.stop, "target": sig.target, "label": setup_tag,
                                  "score": snap["score"], "ev": ev_})
                watching.append(entry)
                if not self.risk.halted:
                    self._try_enter(sym, e_df, e_feats, e_a, sentiment, sig)
            except Exception as e:  # noqa: BLE001
                _log(f"[warn] {sym}: {e}")
        # rank: live signals first, then by opportunity score (best setup potential)
        watching.sort(key=lambda w: (w["signal"] is None, -w.get("score", 0)))
        self.watching = watching
        self._build_forecast_board(watching)
        # --- drive the real TradingView chart so the bot trades VISIBLY ---
        if self.mode == "tradingview" and self.tv is not None:
            self._visualize(plans, watching)
        self._build_chart(plans, watching)
        self.scans += 1
        bias_txt = {1: "BTC UP", -1: "BTC DOWN", 0: "BTC flat"}[self.mkt_bias]
        if self._orphans:
            # Don't hide these. They are real open trades on another account that this
            # runner deliberately will not touch — the user needs to know they exist.
            self._activity(f"NOT MANAGING {len(self._orphans)} position(s) from another venue: "
                           f"{', '.join(sorted(s.replace('USDT', '') for s in self._orphans))} "
                           f"- switch mode to manage them")
        fc_txt = ""
        if self.forecasts and self.forecasts[0].get("model_ready", True):
            top = self.forecasts[0]
            fc_txt = (f" | next move: {top['symbol']} {top['direction']} "
                      f"{top['prob']}% in {top['horizon']}")
        self._activity(f"scanned {len(watching)} crypto coins | {bias_txt} | news {sentiment.score} "
                       f"({sentiment.n_headlines} headlines) | "
                       f"{'no A+ setup' if not any(w['signal'] for w in watching) else 'evaluating signals'}"
                       f"{fc_txt}")
        self.last_tick = _now()
        self._write_state(sentiment)

    # ---- loop control (used by cockpit) ----
    def _loop(self, period_sec: int) -> None:
        self.running = True
        _log(f"runner STARTED, period {period_sec}s, equity ${self.equity:.2f}")
        while self.running:
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001
                _log(f"[tick-error] {e}")
            for _ in range(period_sec):
                if not self.running:
                    break
                time.sleep(1)
        _log("runner STOPPED")

    def start(self, period_sec: int = 120) -> bool:
        if self.running:
            return False
        self._thread = threading.Thread(target=self._loop, args=(period_sec,), daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self.running = False


if __name__ == "__main__":
    import sys
    period = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    mode = sys.argv[2] if len(sys.argv) > 2 else "paper"  # "paper" | "tradingview" | "bybit"
    r = QuantRunner(mode=mode)
    print(f"runner mode={mode} equity=${r.equity}")
    try:
        r._loop(period)
    except KeyboardInterrupt:
        r.stop()
        _log("interrupted by user")
