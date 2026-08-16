"""How to MANAGE a trade so that losing DAYS are rare — measured, not guessed.

Run:  python research_manage.py [bars] [interval]

THE QUESTION THIS ANSWERS. The operator's ask is not "make more R", it is "stop
having days with three losses on them". Those are different objectives and the
usual metrics (PF, expectancy) do not measure the second one at all. So this
script reports, for every management scheme, the distribution of LOSING TRADES
PER DAY — what fraction of days had 0 losses, at most 1, at most 2 — alongside
the R it costs to buy that calm.

WHAT IS HELD CONSTANT. Every scheme sees the SAME entries: the same strategies,
the same EV gate, the same bars. Only what happens AFTER the fill changes. That
is the whole point — entry selection is already tuned (research_edge.py); this
measures the part nobody measured.

THE SCHEMES:
  fixed        stop and target, no interference (what a backtest usually assumes)
  be1          stop -> breakeven once +1R  (what the live bot does today)
  be1_trail    breakeven at +1R, then trail 2.5 ATR   (live bot with USE_TRAIL)
  half1_be     take 50% off at +1R, stop -> breakeven, rest runs to target
  half1_trail  take 50% off at +1R, stop -> breakeven, rest trails 2.5 ATR
  half1_time   half1_be plus a time stop: out at market after N bars under +0.3R

BAR-ORDERING IS DELIBERATELY PESSIMISTIC. When one bar's range contains both the
stop and a favourable level, the stop is assumed to fill FIRST. That penalises
exactly the schemes being advocated for (partials, breakeven stops), so any
improvement they show is real and not an artefact of optimistic fill ordering.

COSTS ARE IN. Every fill pays taker fee + slippage, converted to R via the
setup's own stop distance — a tight stop pays proportionally more, which is why
scalping tiny stops looks better than it trades.

THE GATE IS WALK-FORWARD, NOT THE SHIPPED MODEL. The first version of this script
gated with ml/meta_model.pkl, which was fitted on this very window: it reported a
68% win rate and +0.9R a trade, roughly ten times the edge research_edge.py
measures out-of-sample. That is the model recognising bars it was trained on, and
it does not merely inflate the totals — it changes the RANKING of the schemes,
because a gate that already knows which trades reach target makes "just hold to
target" look free. So probabilities here are produced chronologically, training
only on signals that closed before each test fold, with a 48-bar embargo — the
same discipline as research_edge.py.

HONEST LIMITS. The news gate and the BTC-regime gate are live-only inputs and
cannot be replayed here, so the trade pool is slightly wider than production's.
"""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

import config
import runner
from data.bybit import get_klines_cached
from features.indicators import feature_frame, atr as atr_series
from features.context import bar_context, signal_context, BAR_CONTEXT_COLS, CONTEXT_COLS
from ml.meta import FEATURE_COLS, _new_model, SKLEARN_OK

warnings.filterwarnings("ignore")

HORIZON = 72            # bars a trade is allowed to live (72h on 1h bars)
COST_PER_FILL = config.BACKTEST["taker_fee"] + config.BACKTEST["slippage_bps"] / 1e4
MODEL_COLS = FEATURE_COLS + CONTEXT_COLS + ["side"]   # exactly what the live model sees


# --------------------------------------------------------------------- schemes
SCHEMES: dict[str, dict] = {
    "fixed":       dict(),
    "be1":         dict(be_at=1.0),
    "be1_trail":   dict(be_at=1.0, trail_atr=2.5),
    "half1_be":    dict(be_at=1.0, partial_at=1.0, partial_frac=0.5),
    "half1_trail": dict(be_at=1.0, partial_at=1.0, partial_frac=0.5, trail_atr=2.5),
    "half1_time":  dict(be_at=1.0, partial_at=1.0, partial_frac=0.5,
                        time_stop=24, time_min_r=0.3),
    "half08_be":   dict(be_at=0.8, partial_at=0.8, partial_frac=0.5),
    "half05_be":   dict(be_at=0.5, partial_at=0.5, partial_frac=0.5),
    "half05_trail": dict(be_at=0.5, partial_at=0.5, partial_frac=0.5, trail_atr=2.5),
    "half05_time": dict(be_at=0.5, partial_at=0.5, partial_frac=0.5,
                        time_stop=24, time_min_r=0.3),
    "third05_be":  dict(be_at=0.5, partial_at=0.5, partial_frac=0.34),
    "two05_be":    dict(be_at=0.5, partial_at=0.5, partial_frac=0.66),
}


def simulate(h, l, c, a, i0: int, side: int, entry: float, stop: float, target: float,
             *, be_at: float | None = None, be_offset_r: float = 0.02,
             partial_at: float | None = None, partial_frac: float = 0.5,
             trail_atr: float | None = None, trail_from: float = 1.0,
             time_stop: int | None = None, time_min_r: float = 0.3) -> tuple[float, int]:
    """Walk the trade forward bar by bar. Returns (net R, exit bar index)."""
    n = len(c)
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0, i0
    stop_pct = risk / entry
    cost_r = COST_PER_FILL / stop_pct        # cost of ONE full-size fill, in R
    booked = -cost_r                          # the entry fill is already paid for
    rem = 1.0                                 # fraction of the position still open
    cur_stop = stop
    partial_done = False

    def r_at(px: float) -> float:
        return (px - entry) * side / risk

    for k in range(i0, min(i0 + HORIZON, n)):
        hi, lo = h[k], l[k]
        adverse = lo if side == 1 else hi
        favour = hi if side == 1 else lo

        # --- stop first: pessimistic ordering inside the bar ---
        if (adverse <= cur_stop) if side == 1 else (adverse >= cur_stop):
            booked += rem * (r_at(cur_stop) - cost_r)
            return booked, k

        # --- partial take-profit ---
        if partial_at is not None and not partial_done:
            plevel = entry + side * partial_at * risk
            if (favour >= plevel) if side == 1 else (favour <= plevel):
                booked += partial_frac * (partial_at - cost_r)
                rem -= partial_frac
                partial_done = True

        # --- target ---
        if (favour >= target) if side == 1 else (favour <= target):
            booked += rem * (r_at(target) - cost_r)
            return booked, k

        # --- move the stop (only ever in the favourable direction) ---
        mfe_r = r_at(favour)
        if be_at is not None and mfe_r >= be_at:
            be = entry + side * be_offset_r * risk
            cur_stop = max(cur_stop, be) if side == 1 else min(cur_stop, be)
        if trail_atr is not None and mfe_r >= trail_from:
            t = c[k] - side * trail_atr * a[k]
            cur_stop = max(cur_stop, t) if side == 1 else min(cur_stop, t)

        # --- time stop: a trade that is going nowhere is a loss that hasn't happened yet ---
        if time_stop is not None and (k - i0) >= time_stop and r_at(c[k]) < time_min_r:
            booked += rem * (r_at(c[k]) - cost_r)
            return booked, k

    # ran out of horizon — mark out at the last close we saw
    kend = min(i0 + HORIZON, n) - 1
    booked += rem * (r_at(c[kend]) - cost_r)
    return booked, kend


# --------------------------------------------------------------------- collect
def collect(interval: str, bars: int) -> pd.DataFrame:
    """One row per signal — features, context, and its outcome under EVERY scheme.

    NOTHING is filtered by the model here. The gate is applied afterwards, from
    walk-forward probabilities, so the pool this function returns is the raw
    strategy output and the same collection can be re-gated without refetching.
    """
    btc = get_klines_cached("BTCUSDT", interval, bars=bars, max_age_min=100000)
    rows = []
    for n_done, sym in enumerate(config.UNIVERSE, 1):
        try:
            df = get_klines_cached(sym, interval, bars=bars, max_age_min=100000)
            if len(df) < 300:
                continue
            feats = feature_frame(df)
            a = atr_series(df, 14)
            sigs = runner._gen_signals(df, feats, a)
            ctx = bar_context(df, feats, btc)
            h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
            av = a.to_numpy(float)
            ts = df.index
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {sym}: {e}")
            continue

        kept = 0
        for i, sig in enumerate(sigs):
            if sig is None or i + 1 >= len(df):
                continue
            entry = float(c[i])
            risk = abs(entry - sig.stop)
            if risk <= 0:
                continue
            stop_pct = risk / entry
            atr_now = float(av[i]) if not np.isnan(av[i]) else 0.0
            # the same degenerate-setup guards the live runner applies
            if stop_pct < runner.MIN_STOP_PCT or (atr_now and risk / atr_now < runner.MIN_STOP_ATR):
                continue
            f = feats.iloc[i]
            if f[FEATURE_COLS].isna().any():
                continue
            row = {k: f[k] for k in FEATURE_COLS}
            try:
                crow = ctx.iloc[i]
                row.update({k: crow[k] for k in BAR_CONTEXT_COLS})
                row.update(signal_context(sig, entry, atr_now, crow["btc_above_ema50"]))
            except Exception:  # noqa: BLE001
                pass
            nat_rr = abs(sig.target - entry) / risk
            rec = {k: row.get(k, np.nan) for k in MODEL_COLS}
            rec["side"] = sig.side
            rec.update({"ts": ts[i], "symbol": sym, "rr": nat_rr, "stop_pct": stop_pct,
                        "adx": float(f["adx14"]), "hour": int(ts[i].hour)})
            for name, kw in SCHEMES.items():
                r, kexit = simulate(h, l, c, av, i + 1, sig.side, entry,
                                    sig.stop, sig.target, **kw)
                rec[f"R_{name}"] = r
                if name == "fixed":
                    rec["exit_ts"] = ts[kexit]
                    rec["bars_held"] = kexit - i
                    # the label the model is trained on: did the fixed target land?
                    rec["y"] = int(r > 0)
            rows.append(rec)
            kept += 1
        print(f"  [{n_done}/{len(config.UNIVERSE)}] {sym}: {kept} signals")
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


def walk_forward_probs(d: pd.DataFrame, folds: int = 5, embargo_bars: int = 48) -> np.ndarray:
    """Out-of-sample win probability per signal — trained only on the past.

    Same discipline as research_edge.wf_probs: chronological folds, and signals that
    were still resolving when a fold began are dropped from that fold's training set
    (their outcome lives inside the test window).
    """
    X, y = d[MODEL_COLS], d["y"]
    probs = np.full(len(d), np.nan)
    bounds = np.linspace(0, len(d), folds + 1, dtype=int)
    for k in range(1, folds):
        tr, te = bounds[k], bounds[k + 1]
        if tr < 200 or te <= tr:
            continue
        cutoff = d["ts"].iloc[tr] - pd.Timedelta(hours=embargo_bars)
        idx = np.flatnonzero((d["ts"].iloc[:tr] <= cutoff).to_numpy())
        if len(idx) < 200 or y.iloc[idx].nunique() < 2:
            continue
        m = _new_model().fit(X.iloc[idx], y.iloc[idx])
        probs[tr:te] = m.predict_proba(X.iloc[tr:te])[:, 1]
    return probs


# ----------------------------------------------------------------- day metrics
def day_report(d: pd.DataFrame, col: str, *, max_daily_losses: int | None = None,
               cooldown_h: float = 0.0, lock_day_at_r: float | None = None,
               max_open: int | None = None) -> dict:
    """Replay the trades in time order under a set of DAY rules and report how the
    losing days are distributed. This is the metric the operator actually asked for."""
    taken, day_r, day_losses = [], defaultdict(float), defaultdict(int)
    last_loss_ts: pd.Timestamp | None = None
    cur_day = None
    open_until: list[pd.Timestamp] = []

    for _, t in d.iterrows():
        day = t["ts"].date()
        if day != cur_day:
            cur_day, last_loss_ts, open_until = day, None, []
        if max_daily_losses is not None and day_losses[day] >= max_daily_losses:
            continue
        if lock_day_at_r is not None and day_r[day] >= lock_day_at_r:
            continue
        if last_loss_ts is not None and cooldown_h:
            if (t["ts"] - last_loss_ts).total_seconds() / 3600.0 < cooldown_h:
                continue
        if max_open is not None:
            open_until = [x for x in open_until if x > t["ts"]]
            if len(open_until) >= max_open:
                continue
            open_until.append(t["exit_ts"])
        r = t[col]
        taken.append((day, r, t["ts"], t["exit_ts"]))
        day_r[day] += r
        if r < 0:
            day_losses[day] += 1
            last_loss_ts = t["exit_ts"]

    if not taken:
        return {"trades": 0}
    rs = np.array([x[1] for x in taken])
    days = sorted(day_r)
    dr = np.array([day_r[x] for x in days])
    dl = np.array([day_losses[x] for x in days])
    eq = np.cumsum(dr)
    dd = float((eq - np.maximum.accumulate(eq)).min())
    wins, losses = rs[rs > 0], rs[rs < 0]
    span_days = max(1, (d["ts"].iloc[-1] - d["ts"].iloc[0]).days)
    return {
        "trades": len(rs),
        "trd_day": round(len(rs) / span_days, 2),
        "win%": round(len(wins) / len(rs) * 100, 1),
        "R/trd": round(float(rs.mean()), 3),
        "totR": round(float(rs.sum()), 1),
        "PF": round(float(wins.sum() / -losses.sum()), 2) if len(losses) else 99.0,
        "days": len(days),
        "0loss%": round(float((dl == 0).mean() * 100), 1),
        "<=1loss%": round(float((dl <= 1).mean() * 100), 1),
        "<=2loss%": round(float((dl <= 2).mean() * 100), 1),
        "green%": round(float((dr > 0).mean() * 100), 1),
        "worstday": round(float(dr.min()), 2),
        "maxDD": round(dd, 1),
        "R/day": round(float(dr.sum() / len(days)), 3),
    }


def show(title: str, rep: dict) -> None:
    if not rep.get("trades"):
        print(f"{title:<34} (no trades)")
        return
    print(f"{title:<34} {rep['trades']:>5} {rep['trd_day']:>5} {rep['win%']:>6} "
          f"{rep['R/trd']:>7} {rep['totR']:>7} {rep['PF']:>6} {rep['0loss%']:>7} "
          f"{rep['<=1loss%']:>8} {rep['green%']:>7} {rep['worstday']:>9} {rep['maxDD']:>7}")


HEAD = (f"{'scheme / rules':<34} {'trd':>5} {'t/day':>5} {'win%':>6} {'R/trd':>7} "
        f"{'totR':>7} {'PF':>6} {'0loss%':>7} {'<=1los%':>8} {'green%':>7} "
        f"{'worstday':>9} {'maxDD':>7}")


def main() -> None:
    bars = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    interval = sys.argv[2] if len(sys.argv) > 2 else "1h"
    if not SKLEARN_OK:
        print("sklearn unavailable — the walk-forward gate cannot be built")
        return
    cache = config.CACHE_DIR / f"manage_all_{interval}_{bars}.pkl"
    if cache.exists():
        d = pd.read_pickle(cache)
        print(f"loaded {len(d)} signals from cache")
    else:
        print(f"collecting {interval} x {bars} bars over {len(config.UNIVERSE)} coins...")
        d = collect(interval, bars)
        d.to_pickle(cache)
    if d.empty:
        print("no signals")
        return
    d = d.sort_values("ts").reset_index(drop=True)
    raw = len(d)
    probs_cache = config.CACHE_DIR / f"manage_probs_{interval}_{bars}.pkl"
    if probs_cache.exists():
        d["prob"] = pd.read_pickle(probs_cache)
        print("loaded walk-forward probabilities from cache")
    else:
        print("fitting the walk-forward gate (chronological, 48-bar embargo)...")
        d["prob"] = walk_forward_probs(d)
        pd.to_pickle(d["prob"], probs_cache)
    d = d[~d["prob"].isna()].reset_index(drop=True)
    d["ev"] = d["prob"] * d["rr"].clip(upper=runner.RR_CAP) - (1 - d["prob"])
    scored = len(d)
    pool = d.copy()          # every scored signal — section D re-gates this itself
    d = d[d["ev"] >= runner.EV_MIN].reset_index(drop=True)
    span = (d["ts"].iloc[-1] - d["ts"].iloc[0]).days
    print(f"\n{raw} raw signals -> {scored} scored out-of-sample -> {len(d)} pass "
          f"EV >= {runner.EV_MIN}, over {span} days "
          f"({d['ts'].iloc[0].date()} -> {d['ts'].iloc[-1].date()})\n")

    print("=" * 128)
    print("A) MANAGEMENT ONLY — no day rules, every gated signal taken")
    print("=" * 128)
    print(HEAD)
    for name in SCHEMES:
        show(name, day_report(d, f"R_{name}"))

    best = max(SCHEMES, key=lambda s: day_report(d, f"R_{s}").get("R/day", -9))
    print(f"\nbest by R/day: {best}")

    print("\n" + "=" * 128)
    print(f"B) DAY RULES on top of the two best schemes (loss budget + cooldown + lock-in)")
    print("=" * 128)
    print(HEAD)
    for scheme in dict.fromkeys([best, "half1_be", "be1_trail"]):
        col = f"R_{scheme}"
        show(f"{scheme} | no rules", day_report(d, col))
        for mdl in (3, 2, 1):
            show(f"{scheme} | max {mdl} loss/day",
                 day_report(d, col, max_daily_losses=mdl))
        show(f"{scheme} | max 2 loss + 4h cooldown",
             day_report(d, col, max_daily_losses=2, cooldown_h=4))
        show(f"{scheme} | max 2 loss + lock +2R",
             day_report(d, col, max_daily_losses=2, lock_day_at_r=2.0))
        show(f"{scheme} | max 2 loss + 4h cd + lock 2R",
             day_report(d, col, max_daily_losses=2, cooldown_h=4, lock_day_at_r=2.0))
        show(f"{scheme} | max 1 loss + lock +2R",
             day_report(d, col, max_daily_losses=1, lock_day_at_r=2.0))
        print("-" * 128)

    print("\n" + "=" * 128)
    print("C) WHICH SETUPS LOSE — where the losing trades actually come from")
    print("=" * 128)
    col = f"R_{best}"
    for key, buckets in [
        ("EV",        pd.cut(d["ev"], [0.24, 0.35, 0.5, 0.8, 9])),
        ("ADX",       pd.cut(d["adx"], [0, 15, 20, 25, 100])),
        ("ATR%ile",   pd.cut(d["atr_pctile"], [0, 0.3, 0.6, 0.85, 1.01])),
        ("with_BTC",  d["with_btc"]),
        ("HTF align", (d["htf_trend"] * d["side"]).clip(-1, 1)),
        ("setup",     d["setup_code"]),
        ("stop %",    pd.cut(d["stop_pct"] * 100, [0, 0.75, 1.25, 2.0, 99])),
        ("nat RR",    pd.cut(d["rr"], [0, 1.5, 2.0, 2.5, 99])),
        ("UTC hour",  pd.cut(d["hour"], [-1, 5, 11, 17, 23])),
    ]:
        print(f"\n  by {key}:")
        g = d.groupby(buckets, observed=True)[col]
        for k, v in g:
            if len(v) < 25:
                continue
            print(f"    {str(k):<18} n={len(v):>5}  win {(v > 0).mean() * 100:>5.1f}%  "
                  f"R/trd {v.mean():>+7.3f}  totR {v.sum():>+8.1f}")

    # ---------------------------------------------------------------- section D
    # A filter that only works on the whole sample is a filter that was fitted to it.
    # Every candidate below is therefore also measured on the FIRST half and the
    # SECOND half of the period separately: one that flips sign between them is
    # noise, however good the total looks.
    print("\n" + "=" * 128)
    print("D) FILTER STACK — each rule measured on the whole period AND on both halves")
    print("=" * 128)

    def apply(p: pd.DataFrame, names: list[str]) -> pd.DataFrame:
        m = pd.Series(True, index=p.index)
        for nm in names:
            m &= FILTERS[nm](p)
        return p[m].reset_index(drop=True)

    def line(label: str, sel: pd.DataFrame, scheme: str, **rules) -> None:
        if sel.empty:
            print(f"{label:<40} (nothing left)")
            return
        mid = sel["ts"].iloc[len(sel) // 2]
        full = day_report(sel, f"R_{scheme}", **rules)
        h1 = day_report(sel[sel["ts"] < mid].reset_index(drop=True), f"R_{scheme}", **rules)
        h2 = day_report(sel[sel["ts"] >= mid].reset_index(drop=True), f"R_{scheme}", **rules)
        print(f"{label:<40} {full['trades']:>5} {full['trd_day']:>5} {full['win%']:>6} "
              f"{full['R/trd']:>+7.3f} {full['totR']:>7} {full['PF']:>6} "
              f"{full['<=1loss%']:>7} {full['green%']:>7} {full['maxDD']:>7} "
              f"| {h1.get('R/trd', 0):>+6.3f} {h2.get('R/trd', 0):>+6.3f}")

    head_d = (f"{'filter stack':<40} {'trd':>5} {'t/day':>5} {'win%':>6} {'R/trd':>7} "
              f"{'totR':>7} {'PF':>6} {'<=1los':>7} {'green%':>7} {'maxDD':>7} "
              f"| {'1stH':>6} {'2ndH':>6}")

    scheme, rules = "half1_be", dict(max_daily_losses=2, cooldown_h=4)
    print(f"\nmanagement = {scheme}, day rules = max 2 losses + 4h cooldown\n")
    print(head_d)
    print("-" * 128)
    line("live today (EV>=0.25)", apply(pool, ["ev25"]), scheme, **rules)
    for f in ["ev40", "ev50", "stop125", "stop200", "htf", "vol50", "hours0_17", "hours0_11"]:
        line(f"EV>=0.25 + {f}", apply(pool, ["ev25", f]), scheme, **rules)
    print("-" * 128)
    stack: list[str] = ["ev25"]
    for f in ["stop125", "htf", "vol50", "ev40", "hours0_17"]:
        stack = stack + [f]
        line(" + ".join(stack[1:]) or "none", apply(pool, stack), scheme, **rules)
    print("-" * 128)
    best_stack = ["ev25", "stop125", "htf", "vol50"]
    for sch in ("fixed", "be1_trail", "half1_be", "half05_be", "half05_trail",
                "half05_time", "third05_be", "two05_be"):
        for rl, txt in [(dict(), "no day rules"),
                        (dict(max_daily_losses=2), "max 2 loss"),
                        (dict(max_daily_losses=2, cooldown_h=4), "max 2 loss + 4h cd"),
                        (dict(max_daily_losses=1), "max 1 loss"),
                        (dict(max_daily_losses=2, cooldown_h=4, max_open=3), "max 2 loss + cd + 3 open")]:
            line(f"{sch} | {txt}", apply(pool, best_stack), sch, **rl)
        print("-" * 128)


FILTERS = {
    "ev25":      lambda p: p["ev"] >= 0.25,
    "ev40":      lambda p: p["ev"] >= 0.40,
    "ev50":      lambda p: p["ev"] >= 0.50,
    # Fees and noise are paid in R: a 0.75% stop pays three times the R-cost of a
    # 2.25% one, and sits inside the bar's own noise.
    "stop125":   lambda p: p["stop_pct"] >= 0.0125,
    "stop200":   lambda p: p["stop_pct"] >= 0.02,
    # Never trade against the higher-timeframe trend.
    "htf":       lambda p: (p["htf_trend"] * p["side"]) >= 0,
    # Dead tape doesn't reach targets.
    "vol50":     lambda p: p["atr_pctile"] >= 0.5,
    "hours0_17": lambda p: p["hour"] < 17,
    "hours0_11": lambda p: p["hour"] < 12,
}


if __name__ == "__main__":
    main()
