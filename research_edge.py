"""Where the next point of edge actually is — measured, not guessed.

Run:  python research_edge.py [bars]

This compares, on IDENTICAL data and with the same walk-forward discipline:

  A) the model as it runs today          (12 generic features, gate on P(win))
  B) the same model, but split by TIME   (see the ordering bug below)
  C) a richer feature set                (context the bot already has but never fed
                                          the model: setup type, higher-timeframe
                                          trend, BTC regime, volatility percentile,
                                          hour of day, the setup's own reward:risk)
  D) gating on EXPECTED VALUE            (p*RR - (1-p)) instead of on P(win)

THE ORDERING BUG (why B exists). `tune_accuracy.py` concatenates each coin's
signals one coin after another and then splits the rows by position. So "train on
the past, test on the future" is really "train on the first N coins, test on the
rest" — and the training set contains bars from LATER in time than the test rows.
That is leakage, and it flatters every number it produces. Everything below sorts
all signals by timestamp first and splits on time, which is the only split that
answers the question the bot actually faces: given only the past, what happens next?

Nothing here touches the live bot. It reads cached candles and prints a table.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

import config
import runner
from data.bybit import get_klines_cached
from features.indicators import feature_frame, atr
from ml.meta import FEATURE_COLS, _signal_outcome, SKLEARN_OK
from features.context import (bar_context, signal_context, BAR_CONTEXT_COLS,
                              CONTEXT_COLS)

warnings.filterwarnings("ignore")

BASE_COLS = FEATURE_COLS + ["side"]
# Context the bot already computes for its own decisions but has never given the
# model. Each one is information available AT the signal bar — no forward peeking.
EXTRA_COLS = CONTEXT_COLS


def collect(interval: str, bars: int) -> pd.DataFrame:
    """One row per fired signal: its features, its context, and how it resolved.

    Context comes from features/context.py — the SAME module the live runner uses,
    so a model trained here is scoring identical inputs in production.
    """
    btc_df = get_klines_cached("BTCUSDT", interval, bars=bars, max_age_min=600)
    rows = []
    for sym in config.UNIVERSE:
        try:
            df = get_klines_cached(sym, interval, bars=bars, max_age_min=600)
            feats = feature_frame(df)
            a = atr(df, 14)
            sigs = runner._gen_signals(df, feats, a)
            h, low, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
            ctx = bar_context(df, feats, btc_df)

            for i, sig in enumerate(sigs):
                if sig is None or i + 1 >= len(df):
                    continue
                y = _signal_outcome(h, low, c, i + 1, sig.side, sig.stop, sig.target, 48)
                if y is None:
                    continue
                f = feats.iloc[i]
                if f[FEATURE_COLS].isna().any():
                    continue
                entry = float(c[i])
                risk = abs(entry - sig.stop)
                if risk <= 0:
                    continue
                crow = ctx.iloc[i]
                row = {k: f[k] for k in FEATURE_COLS}
                row.update({k: crow[k] for k in BAR_CONTEXT_COLS})
                row.update(signal_context(sig, entry, float(a.iloc[i]),
                                          crow["btc_above_ema50"]))
                row.update({
                    "side": sig.side,
                    "y": y,
                    "rr": abs(sig.target - entry) / risk,
                    "stop_pct": risk / entry,     # for the fee model, below
                    "ts": df.index[i],
                    "symbol": sym,
                })
                rows.append(row)
        except Exception as e:  # noqa: BLE001
            print(f"  {sym}: {e}")
    return pd.DataFrame(rows)


def net_R(d: pd.DataFrame, y=None, rr=None) -> np.ndarray:
    """Result per trade in R, AFTER costs.

    A trade pays taker fee + slippage twice (in and out) on the full notional. In
    R units that is (2*cost)/stop_pct, because R is the stop distance. This is why
    a tight stop is expensive: a 0.5% stop pays four times the R-cost of a 2% one.
    Ignoring it is how a backtest shows +0.06R and a live account still bleeds.
    """
    yy = d["y"].to_numpy() if y is None else y
    rrv = d["rr"].to_numpy() if rr is None else rr
    cost = config.BACKTEST["taker_fee"] + config.BACKTEST["slippage_bps"] / 10_000.0
    fee_R = 2 * cost / d["stop_pct"].to_numpy()
    return np.where(yy == 1, rrv, -1.0) - fee_R


def wf_probs(data: pd.DataFrame, cols: list[str], folds: int = 5,
             by_time: bool = True, embargo_bars: int = 48, model=None):
    """Out-of-sample probability per signal.

    by_time=True splits chronologically (honest); False reproduces the row-order
    split used today.

    EMBARGO: a label looks up to 48 bars forward, so signals fired just before the
    split boundary are still resolving inside the test window. Training on them
    leaks the test period's outcome. Those rows are dropped from the training set.
    """
    from ml.meta import _new_model
    d = data.sort_values("ts").reset_index(drop=True) if by_time else data.reset_index(drop=True)
    X, y = d[cols], d["y"]
    n = len(d)
    probs = np.full(n, np.nan)
    bounds = np.linspace(0, n, folds + 1, dtype=int)
    for k in range(1, folds):
        tr, te = bounds[k], bounds[k + 1]
        if tr < 100 or te <= tr:
            continue
        if by_time:
            cutoff = d["ts"].iloc[tr] - pd.Timedelta(hours=embargo_bars)
            keep = d["ts"].iloc[:tr] <= cutoff
            idx = np.flatnonzero(keep.to_numpy())
        else:
            idx = np.arange(tr)
        if len(idx) < 100 or y.iloc[idx].nunique() < 2:
            continue
        m = (model or _new_model)().fit(X.iloc[idx], y.iloc[idx])
        probs[tr:te] = m.predict_proba(X.iloc[tr:te])[:, 1]
    return probs, d


def report(name: str, d: pd.DataFrame, probs: np.ndarray, days: float,
           gate: str = "prob", net: bool = True) -> list[tuple]:
    ok = ~np.isnan(probs)
    p, yy, rr = probs[ok], d["y"].to_numpy()[ok], d["rr"].to_numpy()[ok]
    R = (net_R(d)[ok] if net else np.where(yy == 1, rr, -1.0))
    # EV in R units: what this signal is worth if the model's probability is honest.
    ev = p * rr - (1 - p)
    grid = ((0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7) if gate == "prob"
            else (-0.2, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0))
    score = p if gate == "prob" else ev
    print(f"\n=== {name}  ({ok.sum():,} scored signals) ===")
    print(f"{'gate':>7} {'trades':>7} {'/day':>6} {'win%':>7} {'payoff':>7} "
          f"{'PF':>6} {'expR':>7} {'R/day':>7}")
    print("-" * 62)
    out = []
    for g in grid:
        m = score >= g
        if m.sum() < 30:
            continue
        r = R[m]
        w, l = r[r > 0], r[r < 0]
        win = len(w) / len(r) * 100
        pf = w.sum() / -l.sum() if len(l) else float("inf")
        exp = r.mean()
        tpd = m.sum() / days
        print(f"{g:>7.2f} {m.sum():>7} {tpd:>6.2f} {win:>6.1f}% "
              f"{(w.mean()/-l.mean()) if len(w) and len(l) else 0:>7.2f} "
              f"{pf:>6.2f} {exp:>+7.3f} {exp*tpd:>+7.3f}")
        out.append((exp * tpd, exp, pf, win, g, tpd, name, gate))
    return out


def main() -> None:
    if not SKLEARN_OK:
        print("scikit-learn unavailable — cannot run.")
        return
    bars = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    interval = runner.DECISION_INTERVAL
    per_day = {"15m": 96, "30m": 48, "1h": 24, "4h": 6}.get(interval, 24)

    # Collecting is the slow part (59 coins x indicators x signal replay), and every
    # experiment below wants the SAME rows. Cache them so ideas can be tested in
    # seconds instead of re-deriving the dataset each time.
    # pickle, not parquet: pyarrow is installed on the server but not on this
    # machine's Python 3.14, and a research cache that only works on one of them
    # is worse than none.
    cache = config.ROOT / "cache" / f"research_{interval}_{bars}.pkl"
    if cache.exists() and "--fresh" not in sys.argv:
        data = pd.read_pickle(cache)
        print(f"loaded {len(data):,} cached signals from {cache.name}")
    else:
        print(f"collecting: {len(config.UNIVERSE)} coins, {interval}, {bars} bars")
        data = collect(interval, bars)
        if not data.empty:
            data.to_pickle(cache)
    if data.empty:
        print("no signals")
        return
    span_days = (data["ts"].max() - data["ts"].min()).total_seconds() / 86400
    print(f"\n{len(data):,} signals over {span_days:.0f} days | "
          f"raw win {data['y'].mean()*100:.1f}% | median RR {data['rr'].median():.2f}")
    print(f"setups: {data['setup_code'].map({0:'snapback',1:'continuation',2:'range'}).value_counts().to_dict()}")

    fee_R = 2 * (config.BACKTEST["taker_fee"] + config.BACKTEST["slippage_bps"] / 1e4) \
        / data["stop_pct"].median()
    print(f"cost per trade: {fee_R:.3f}R at the median stop width "
          f"({data['stop_pct'].median()*100:.2f}% stop) — subtracted everywhere below")

    results = []
    # A) exactly what runs today, including the row-order split, costs off:
    #    the numbers the project has been steering by.
    p_a, d_a = wf_probs(data, BASE_COLS, by_time=False)
    results += report("A) today: row-order split, no costs", d_a, p_a, span_days, net=False)
    # B) same model and features, honest chronological split + costs
    p_b, d_b = wf_probs(data, BASE_COLS, by_time=True)
    results += report("B) today's model, TIME split + costs", d_b, p_b, span_days)
    # C) richer features
    cols_c = BASE_COLS + EXTRA_COLS
    p_c, d_c = wf_probs(data, cols_c, by_time=True)
    results += report("C) + context features", d_c, p_c, span_days)
    # D) same model as C, gated on expected value rather than raw probability
    results += report("D) context features, gate on EV", d_c, p_c, span_days, gate="ev")

    if "--save" in sys.argv:
        from ml.meta import MODEL_COLS, train_final
        missing = [c for c in MODEL_COLS if c not in data.columns]
        if missing:
            print(f"cannot save — dataset is missing {missing}")
        else:
            train_final(data[MODEL_COLS], data["y"])
            print(f"\nlive model retrained on {len(data):,} signals x {len(MODEL_COLS)} "
                  f"features and saved to ml/meta_model.pkl")

    print("\n" + "=" * 66)
    print("Ranked by R/day — expectancy x frequency, the number that compounds:")
    for rpd, exp, pf, win, g, tpd, name, gate in sorted(results, reverse=True)[:6]:
        print(f"  {rpd:+.3f} R/day | {name[:32]:<32} {gate}>={g:.2f} | "
              f"{win:.1f}% win, PF {pf:.2f}, {tpd:.2f}/day, {exp:+.3f}R each")
    return data, d_c, p_c, cols_c, span_days


if __name__ == "__main__":
    main()
