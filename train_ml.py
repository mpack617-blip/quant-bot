"""Train + walk-forward-evaluate the meta-labeling model across the universe.

Usage: python train_ml.py [bars] [threshold]
Trains on the live mix: trend (snapback+cont)@DECISION_INTERVAL + range-fade@RANGE_INTERVAL.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config
from data.bybit import get_klines
from features.indicators import feature_frame, atr
import runner
from strategies.continuation import combined_signals
from strategies.range_fade import generate_signals as range_fade
from ml.meta import build_dataset, walk_forward_eval, train_final, FEATURE_COLS


def _collect(interval, bars, signal_fn):
    """Build (X, y, pnl) from a lens over the universe on a given timeframe."""
    X_all, y_all, pnl_all = [], [], []
    for sym in config.UNIVERSE:
        try:
            df = get_klines(sym, interval, bars=bars)
            feats = feature_frame(df)
            a = atr(df, 14)
            sigs = signal_fn(df, feats, a)
            X, y, idxs = build_dataset(df, feats, sigs)
            if len(X) == 0:
                continue
            for k, i in enumerate(idxs):
                s = sigs[i]
                entry = float(df["close"].iloc[i])
                risk = abs(entry - s.stop); reward = abs(s.target - entry)
                rr = (reward / risk) if risk > 0 else 1.0
                pnl_all.append(rr if y.iloc[k] == 1 else -1.0)
            X_all.append(X); y_all.append(y)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {sym}: {e}")
    return X_all, y_all, pnl_all


def main() -> None:
    # The model must see the SAME signal distribution the runner filters. The runner
    # trades TWO lenses on TWO timeframes: trend (snapback+cont) on DECISION_INTERVAL,
    # and range-fade on RANGE_INTERVAL. We pool BOTH so the meta-label is calibrated to
    # the full live mix. Pulls the exact runner params so train/live never drift.
    bars = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else runner.ML_MIN_PROB

    trend = lambda df, feats, a: combined_signals(
        df, feats, a, snapback_params=runner.SNAPBACK_PARAMS, cont_params=runner.CONT_PARAMS)
    rfade = lambda df, feats, a: range_fade(df, feats, a, **runner.RANGE_PARAMS)

    # RANGE_BARS is just the live scanning window; for TRAINING use full history so the
    # (rarer, higher-TF) range-fade lens contributes enough examples to calibrate on.
    Xt, yt, pt = _collect(runner.DECISION_INTERVAL, bars, trend)
    Xr, yr, pr = _collect(runner.RANGE_INTERVAL, bars, rfade)
    X_all, y_all, pnl_all = Xt + Xr, yt + yr, pt + pr

    if not X_all:
        print("No signals to train on.")
        return

    X = pd.concat(X_all, ignore_index=True)
    y = pd.concat(y_all, ignore_index=True)
    pnls = np.array(pnl_all, float)
    n_trend = sum(len(x) for x in Xt)

    print(f"\n=== META-LABEL ML TRAINING — trend@{runner.DECISION_INTERVAL} + range@{runner.RANGE_INTERVAL} ===")
    print(f"  trend signals: {n_trend}  |  range-fade signals: {len(X) - n_trend}")
    print(f"dataset: {len(X)} signals  |  raw win-rate: {y.mean()*100:.1f}%  |  features: {len(FEATURE_COLS)}")

    res = walk_forward_eval(X, y, pnls, folds=5, threshold=threshold)
    print(f"\n--- WALK-FORWARD (out-of-sample, threshold {res.threshold}) ---")
    print(f"  take ALL signals : win {res.base_winrate}%   PF {res.base_pf}")
    print(f"  ML-approved only : win {res.model_winrate}%   PF {res.model_pf}   (took {res.coverage}% of signals)")
    verdict = "ML FILTER ADDS EDGE [+]" if res.model_pf > res.base_pf else "no improvement at this threshold [-]"
    print(f"  verdict: {verdict}")

    model = train_final(X, y)
    # feature importance via permutation-free proxy: not built-in for HGB; show class balance
    print(f"\nfinal model trained on all {len(X)} signals and saved -> ml/meta_model.pkl")
    print("live use: ml.predict_proba(feature_row, side) -> win probability filter")


if __name__ == "__main__":
    main()
