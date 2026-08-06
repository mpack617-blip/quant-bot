"""Train the next-move forecaster on the whole universe.

    python train_forecast.py [interval] [bars] [horizon]
    python train_forecast.py 1h 5000 6        # default

Pools every coin in config.UNIVERSE into one dataset (crypto pairs share far more
behaviour than they differ, and one pooled model beats 24 thin per-coin models),
then reports a strictly walk-forward score: train on the past, predict an untouched
future, never shuffled.

Read the output like a sceptic:
  accuracy 50%  = coin flip, the model knows nothing. Do not deploy.
  AUC     0.50  = no ranking power at all. 0.55+ is a genuine, tradeable tilt.
  calibration   = the honest test. In the "0.60-0.70" bucket, did price really rise
                  ~60-70% of the time? If predicted and realised match, the numbers
                  the cockpit shows mean what they say.
"""
from __future__ import annotations

import sys
import time

import pandas as pd

import config
from data.bybit import get_klines_cached
from features.indicators import feature_frame
from ml import forecast as F


def main() -> None:
    interval = sys.argv[1] if len(sys.argv) > 1 else "1h"
    bars = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else F.HORIZON

    if not F.SKLEARN_OK:
        print("scikit-learn unavailable — cannot train (Smart App Control may be blocking it).")
        return

    print(f"training next-move forecaster: {interval}, {bars} bars, horizon {horizon} bars "
          f"({F._horizon_text(interval, horizon)} ahead), {len(config.UNIVERSE)} coins\n")

    Xs = []
    for sym in config.UNIVERSE:
        try:
            df = get_klines_cached(sym, interval, bars=bars, max_age_min=60)
            feats = feature_frame(df)
            X, y, mv = F.build_dataset(df, feats, horizon=horizon)
            if len(X) < 100:
                print(f"  {sym:10} skipped (only {len(X)} usable rows)")
                continue
            # Carry the label + forward move as COLUMNS of one frame. Aligning them
            # later by timestamp index would blow up: 24 coins share the same bar
            # timestamps, so a .loc reindex cross-joins into millions of bogus rows.
            block = X.copy()
            block["_y"] = y.to_numpy()
            block["_mv"] = mv.to_numpy()
            Xs.append(block)
            print(f"  {sym:10} {len(X):5} rows   up-rate {y.mean()*100:.1f}%")
        except Exception as e:  # noqa: BLE001
            print(f"  {sym:10} FAILED: {e}")
        time.sleep(0.05)

    if not Xs:
        print("\nno data collected — aborting.")
        return

    # Interleave by TIME, not by coin: concatenating coin-blocks would let the
    # walk-forward split train on one coin and test on another instead of training
    # on the past and testing on the future.
    pool = pd.concat(Xs).sort_index().reset_index(drop=True)
    y = pool.pop("_y")
    mv = pool.pop("_mv")
    X = pool

    print(f"\npooled dataset: {len(X):,} rows, {X.shape[1]} features, "
          f"base up-rate {y.mean()*100:.1f}%")

    print("\nwalk-forward evaluation (out-of-sample)...")
    rep, oos_p, oos_t = F.walk_forward(X, y, mv, return_raw=True)
    print(f"  scored rows      {rep.n:,}")
    print(f"  accuracy         {rep.accuracy}%      (50% = coin flip)")
    print(f"  AUC              {rep.auc}        (0.50 = no skill, 0.55+ = real tilt)")
    print(f"  confident calls  {rep.edge_hi}% right on {rep.coverage_hi}% of bars")
    print(f"  magnitude MAE    {rep.mae_move}%")
    print("\n  calibration (does the stated probability mean what it says?)")
    for band, d in rep.by_bucket.items():
        print(f"    predicted {band}: realised UP {d['realised_up_pct']}%  (n={d['n']})")

    # Learn the overconfidence squash from the OUT-OF-SAMPLE predictions only —
    # calibrating on in-sample output would just re-learn the model's own optimism.
    cal = F.fit_calibration(oos_p, oos_t)
    print(f"\n  calibration fitted: logit slope {cal[0]:.3f}, bias {cal[1]:+.3f}")
    print("    raw -> shown:  " + "  ".join(
        f"{r:.2f}->{F.apply_calibration(r, cal):.2f}" for r in (0.30, 0.40, 0.50, 0.60, 0.70)))

    meta = {"interval": interval, "bars": bars, "horizon": horizon,
            "trained_utc": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
            "rows": len(X), "accuracy": rep.accuracy, "auc": rep.auc,
            "edge_hi": rep.edge_hi, "mae_move": rep.mae_move}
    F.train_final(X, y, mv, meta, calibration=cal)
    print(f"\nsaved -> {F.MODEL_PATH}")

    if rep.auc < 0.52:
        print("\nHONEST VERDICT: AUC below 0.52 — this model has little to no directional")
        print("edge on this timeframe. The bot should keep trading its rule setups and")
        print("treat the forecast as commentary, not as a gate.")
    elif rep.auc < 0.56:
        print("\nVERDICT: a small but real tilt. Useful as a filter/confirmation on top")
        print("of the rule strategies — not strong enough to trade on its own.")
    else:
        print("\nVERDICT: a solid directional tilt for crypto. Still a probability, not a")
        print("prophecy — position sizing and stops remain what keep the account alive.")


if __name__ == "__main__":
    main()
