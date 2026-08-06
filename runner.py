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
from ml.meta import predict_proba, FEATURE_COLS
from ml.forecast import forecast as next_move, MODEL_INTERVAL
from features.microstructure import snapshot as ms_snapshot
from brokers import TradingViewBroker
from brokers.bybit import BybitBroker
import news
import journal
import market
import conviction
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
# ML gate RESTORED 2026-08-06 (user: "profit chahiye, loss minimum, accuracy high").
# Measured walk-forward on the 59-coin universe, 3,045 out-of-sample signals
# (tune_accuracy.py) — every loss is exactly -1.00R by construction, so the only
# things that move are win-rate and payoff:
#     thr 0.00 (the old "trade everything")  36.8% win  PF 0.98  -0.011R  <- LOSES
#     thr 0.45                               49.1% win  PF 1.42  +0.216R
#     thr 0.50                               51.1% win  PF 1.53  +0.258R
#     thr 0.60  <- chosen                    55.3% win  PF 1.93  +0.414R  1.7 trades/day
#     thr 0.70                               56.8% win  PF 2.29  +0.556R  but 0.35/day
#                                                                          (44 trades = too
#                                                                           thin to trust)
# 0.60 is the best point that still clears the user's "at least one trade a day".
# Going higher buys a little more edge and loses most of the activity.
ML_MIN_PROB = 0.60
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

BYBIT_RISK = {
    "max_risk_per_trade_pct": 4.0,
    # Raised 3 -> 8 alongside the 24 -> 59 coin universe: with more markets scanned,
    # good setups cluster, and a cap of 3 would silently drop most of them. At ~1% risk
    # per trade this still caps total risk at roughly 8% of the account.
    "max_concurrent_positions": 8,
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
        self.running = False
        self._thread: threading.Thread | None = None
        self.last_tick = None
        self.last_note = f"idle ({mode})"
        self.watching: list[dict] = []      # what the bot is looking at right now
        self.activity_log: list[str] = []   # rolling feed of what it's doing (newest last)
        self.scans = 0                      # how many full market scans done
        self.tv_view: dict = {}             # the live TradingView chart the bot is driving
        self.mkt_bias = 0                   # BTC market-regime gate (+1 up / -1 down / 0 neutral)
        self.forecasts: list[dict] = []     # the bot's read on the NEXT move, per coin
        self._fc_cache: dict[str, object] = {}   # per-tick memo (cleared each tick)
        self._orphans: set[str] = set()     # open positions belonging to a DIFFERENT venue

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

    def _book_equity(self) -> float:
        """The account the bot sizes against.

        BYBIT_CAPITAL is None -> the bot runs the WHOLE account: book equity IS the
        exchange balance. This is the normal mode.

        BYBIT_CAPITAL set -> a notional book of that size that starts there and then
        moves one-for-one with the account's real PnL. Used to rehearse a small account
        on a large demo balance; the exchange balance stays the untouched truth.
        """
        if BYBIT_CAPITAL is None:
            return round(self.real_equity, 4)
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
        }
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

        # --- trailing stop ---  (only ONCE the trade is solidly in profit)
        # Until +BE_AFTER_R the original stop holds (don't choke a fresh trade). After
        # that: lock breakeven, then trail TRAIL_ATR_MULT ATRs behind price. The stop is
        # only ever moved in the FAVOURABLE direction — it never loosens — so this turns
        # a would-be loser into break-even and lets a runner keep running.
        if not USE_TRAIL:
            return
        move = (price - pos["entry"]) * side
        risk_unit = abs(pos["entry"] - pos["stop"])
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
            rec = journal.record_close(sym, exit_px, round(pnl, 4),
                                       round(r, 2) if r else None, why)
            self.real_equity = self.ex.equity() or self.real_equity
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
        feat_row = feats.iloc[-1][FEATURE_COLS].to_dict()

        prob = predict_proba(feat_row, sig.side)
        if prob < ML_MIN_PROB:
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
        stop_d0 = abs(price - sig.stop)
        natural_rr = abs(sig.target - price) / stop_d0 if stop_d0 else 0.0
        plan = conviction.assess(prob=prob, adx=float(feats.iloc[-1]["adx14"]),
                                 natural_rr=natural_rr, side=sig.side,
                                 entry=price, stop=sig.stop, fwd_agree=fwd_agree,
                                 equity=self.equity)

        decision = self.risk.evaluate(side=sig.side, entry=price, stop=plan.stop,
                                      open_positions=journal.open_positions(),
                                      risk_usd=plan.risk_usd)
        if not decision.approved:
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
                    self.equity = self._book_equity()
                else:
                    self.equity = eq
        self._reconcile()            # exchange is the source of truth (bybit mode)
        self.risk.update_equity(self.equity)
        sentiment = news.market_sentiment()
        self.mkt_bias = market.current_bias(DECISION_INTERVAL)   # don't fight BTC's trend
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
                         "signal": None, "ml": None,
                         "setup": snap["setup"], "score": snap["score"]}
                if sig is not None:
                    entry["signal"] = "LONG" if sig.side == 1 else "SHORT"
                    entry["ml"] = round(predict_proba(e_feats.iloc[-1][FEATURE_COLS].to_dict(), sig.side), 2)
                    setup_tag = sig.reason.split(":")[0]
                    entry["setup"] = setup_tag           # show the ACTUAL firing setup, not the multi_angle proxy
                    self._activity(f"SIGNAL {setup_tag} {entry['symbol']} "
                                   f"(ADX {entry['adx']:.0f}, RSI {entry['rsi']:.0f}, ML {entry['ml']})")
                    plans.append({"sym": sym, "side": sig.side, "entry": e_price,
                                  "stop": sig.stop, "target": sig.target, "label": setup_tag,
                                  "score": snap["score"]})
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
