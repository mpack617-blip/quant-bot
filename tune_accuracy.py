"""What does each ML threshold actually buy you? Win rate, payoff, frequency.

The ask is always "more accuracy, tiny losses, big wins". You cannot max all three
at once — every point of extra win-rate is paid for in trades skipped, and every
dollar of extra payoff is paid for in win-rate. This script measures that curve on
the CURRENT universe and the CURRENT live strategy, so the choice is made on numbers
rather than hope.

    python tune_accuracy.py [bars]

Everything is walk-forward: the model that scores a signal was trained only on
signals that came before it. In-sample accuracy is worthless here and is not shown.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config
import runner
from data.bybit import get_klines_cached
from features.indicators import feature_frame, atr
from ml.meta import build_dataset, FEATURE_COLS, _new_model, train_final, SKLEARN_OK


def collect(interval: str, bars: int):
    """(X, y, rr) for every signal the LIVE strategy fires across the universe."""
    Xs, ys, rrs = [], [], []
    for sym in config.UNIVERSE:
        try:
            df = get_klines_cached(sym, interval, bars=bars, max_age_min=600)
            feats = feature_frame(df)
            a = atr(df, 14)
            sigs = runner._gen_signals(df, feats, a)
            X, y, idxs = build_dataset(df, feats, sigs)
            if len(X) == 0:
                continue
            for k, i in enumerate(idxs):
                s = sigs[i]
                entry = float(df["close"].iloc[i])
                risk = abs(entry - s.stop)
                rrs.append(abs(s.target - entry) / risk if risk > 0 else 1.0)
            Xs.append(X)
            ys.append(y)
        except Exception as e:  # noqa: BLE001
            print(f"  {sym}: {e}")
    if not Xs:
        return None, None, None
    return pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True), np.array(rrs)


def walk_forward_probs(X, y, folds: int = 5):
    """Out-of-sample probability for each signal (NaN for the first fold, which has
    no past to train on)."""
    n = len(X)
    probs = np.full(n, np.nan)
    bounds = np.linspace(0, n, folds + 1, dtype=int)
    for k in range(1, folds):
        tr, te = bounds[k], bounds[k + 1]
        if tr < 100 or te <= tr or y.iloc[:tr].nunique() < 2:
            continue
        m = _new_model().fit(X.iloc[:tr], y.iloc[:tr])
        probs[tr:te] = m.predict_proba(X.iloc[tr:te])[:, 1]
    return probs


def main() -> None:
    if not SKLEARN_OK:
        print("scikit-learn unavailable — cannot tune.")
        return
    bars = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    interval = runner.DECISION_INTERVAL
    days = bars / {"15m": 96, "30m": 48, "1h": 24, "4h": 6}.get(interval, 24)

    print(f"collecting signals: {len(config.UNIVERSE)} coins, {interval}, {bars} bars "
          f"(~{days:.0f} days)\n")
    X, y, rr = collect(interval, bars)
    if X is None:
        print("no data")
        return
    print(f"\n{len(X):,} signals | raw win-rate {y.mean()*100:.1f}% | median RR {np.median(rr):.2f}\n")

    probs = walk_forward_probs(X, y)
    ok = ~np.isnan(probs)
    p, yy, rrv = probs[ok], y.to_numpy()[ok], rr[:len(y)][ok]
    print(f"scored out-of-sample: {ok.sum():,} signals\n")

    # Per-trade result in R (risk units): a win pays its reward:risk, a loss costs 1R.
    R = np.where(yy == 1, rrv, -1.0)

    print(f"{'ML thr':>7} {'trades':>8} {'/day':>6} {'win%':>7} {'avg win':>9} "
          f"{'avg loss':>9} {'payoff':>7} {'PF':>6} {'exp R':>7}")
    print("-" * 76)
    best = []
    for thr in (0.00, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        m = p >= thr
        if m.sum() < 30:
            continue
        r = R[m]
        w = r[r > 0]
        l = r[r < 0]
        win = len(w) / len(r) * 100
        pf = w.sum() / -l.sum() if len(l) and l.sum() else float("inf")
        exp = r.mean()
        print(f"{thr:>7.2f} {m.sum():>8} {m.sum()/days:>6.2f} {win:>6.1f}% "
              f"{w.mean() if len(w) else 0:>8.2f}R {l.mean() if len(l) else 0:>8.2f}R "
              f"{(w.mean()/-l.mean()) if len(w) and len(l) else 0:>6.2f} {pf:>6.2f} {exp:>+7.3f}")
        best.append((exp, pf, win, thr, m.sum() / days))

    print("\nHow to read this: 'exp R' is the average profit per trade in RISK units.")
    print("+0.20R means each trade earns 0.2x what it risks, on average. That is the")
    print("number that compounds — a high win-rate with a terrible payoff still loses.")
    if best:
        best.sort(reverse=True)
        e, pf, win, thr, tpd = best[0]
        print(f"\nBest expectancy: threshold {thr:.2f} -> {win:.1f}% win, PF {pf:.2f}, "
              f"{tpd:.2f} trades/day, {e:+.3f}R per trade")
        hi = max(best, key=lambda b: (b[2], b[0]))
        print(f"Highest win-rate: threshold {hi[3]:.2f} -> {hi[2]:.1f}% win, PF {hi[1]:.2f}, "
              f"{hi[4]:.2f} trades/day, {hi[0]:+.3f}R per trade")

    if "--save" in sys.argv:
        train_final(X, y)
        print(f"\nmodel retrained on all {len(X):,} signals and saved")


if __name__ == "__main__":
    main()
