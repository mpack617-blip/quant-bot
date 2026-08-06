"""A MECHANICAL approximation of Craig Percoco's day-trading method, so we can
backtest it honestly instead of guessing if it's profitable.

His actual method (from his video transcript) is largely DISCRETIONARY price
action — so this can never be a perfect 1:1. What IS mechanically definable, and
is the core of his edge, is captured here:

  - Market structure trend filter  (HH/HL = up, LH/LL = down)  -> via EMA20 vs EMA50
  - Fair Value Gap (FVG): the 3-candle imbalance Lux Algo draws
       bullish FVG: high[k-2] < low[k]   (gap zone = [high[k-2], low[k]])
       bearish FVG: low[k-2]  > high[k]  (gap zone = [high[k], low[k-2]])
  - ENTRY: price pulls back and RETESTS the FVG midpoint with the trend, then
    holds ("reclaim the underside of that level")  -> his "fair value gap entry"
  - STOP: "comfortably outside that level" -> just beyond the gap, +0.5 ATR buffer
  - TARGET: "let winners run" -> RR target + the engine's move-to-BE + ATR trail
  - RSI reaction filter (not into the opposite extreme)

Signals are emitted on the retest bar; the backtest engine fills at next open and
manages stop/BE/trail itself. No look-ahead: bar i uses only data up to close i.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import Signal


def generate_signals(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    atr: pd.Series,
    *,
    rr: float = 2.0,
    stop_buffer_atr: float = 0.5,
    fvg_max_age: int = 30,      # an FVG stays "live" for this many bars
    min_gap_atr: float = 0.25,  # ignore trivially small gaps (noise)
    rsi_long_max: float = 72.0,
    rsi_short_min: float = 28.0,
    allow_long: bool = True,
    allow_short: bool = True,
) -> list[Signal | None]:
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    e20 = feats["ema20"].to_numpy(float)
    e50 = feats["ema50"].to_numpy(float)
    rsi = feats["rsi14"].to_numpy(float)
    a = atr.to_numpy(float)
    n = len(df)
    out: list[Signal | None] = [None] * n

    # active FVG zones: list of dicts {dir, lo, hi, mid, born, used}
    bull: list[dict] = []
    bear: list[dict] = []

    for i in range(2, n):
        if np.isnan(e50[i]) or np.isnan(a[i]) or a[i] <= 0:
            continue

        # --- detect a new FVG formed by bars (i-2, i-1, i) ---
        if high[i - 2] < low[i] and (low[i] - high[i - 2]) >= min_gap_atr * a[i]:
            bull.append({"lo": high[i - 2], "hi": low[i],
                         "mid": (high[i - 2] + low[i]) / 2, "born": i, "used": False})
        if low[i - 2] > high[i] and (low[i - 2] - high[i]) >= min_gap_atr * a[i]:
            bear.append({"lo": high[i], "hi": low[i - 2],
                         "mid": (high[i] + low[i - 2]) / 2, "born": i, "used": False})

        # expire old / filled zones
        bull = [z for z in bull if i - z["born"] <= fvg_max_age and close[i] > z["lo"]]
        bear = [z for z in bear if i - z["born"] <= fvg_max_age and close[i] < z["hi"]]

        up = e20[i] > e50[i]
        dn = e20[i] < e50[i]
        sig: Signal | None = None

        # --- LONG: uptrend, price retests a bullish FVG midpoint and holds ---
        if allow_long and up and rsi[i] <= rsi_long_max:
            for z in bull:
                if z["used"] or z["born"] == i:
                    continue
                if low[i] <= z["mid"] and close[i] >= z["lo"]:   # tagged midpoint, held above the gap floor
                    entry = close[i]
                    stop = z["lo"] - stop_buffer_atr * a[i]
                    risk = entry - stop
                    if risk <= 0:
                        continue
                    sig = Signal(1, stop, entry + rr * risk,
                                 f"FVG LONG: retest gap [{z['lo']:.4g}-{z['hi']:.4g}], RSI {rsi[i]:.0f}")
                    z["used"] = True
                    break

        # --- SHORT: downtrend, price retests a bearish FVG midpoint and holds ---
        if sig is None and allow_short and dn and rsi[i] >= rsi_short_min:
            for z in bear:
                if z["used"] or z["born"] == i:
                    continue
                if high[i] >= z["mid"] and close[i] <= z["hi"]:
                    entry = close[i]
                    stop = z["hi"] + stop_buffer_atr * a[i]
                    risk = stop - entry
                    if risk <= 0:
                        continue
                    sig = Signal(-1, stop, entry - rr * risk,
                                 f"FVG SHORT: retest gap [{z['lo']:.4g}-{z['hi']:.4g}], RSI {rsi[i]:.0f}")
                    z["used"] = True
                    break

        out[i] = sig
    return out
