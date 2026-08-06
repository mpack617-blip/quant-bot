"""Baseline strategy v1 — trend-aligned pullback continuation.

The same edge that won trade T2 on the discretionary bot (short the weakest
laggard *with* the trend on a failed bounce), expressed as hard rules:

  LONG  when EMA stack is up (20>50>200), ADX shows a real trend, price pulls
        back to/under EMA20 and then RECLAIMS it with RSI in a constructive
        (not overbought) zone.
  SHORT is the mirror image — trend down, price pops to EMA20 and rejects.

Stops are ATR-based; the target is R:R-driven and the engine then trails
winners. No look-ahead: every value used at bar i is known at the close of i.
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
    adx_min: float = 22.0,
    atr_mult: float = 1.5,
    rr: float = 3.0,
    atr_pct_max: float = 6.0,
    slope_lookback: int = 5,
    allow_long: bool = True,
    allow_short: bool = True,
) -> list[Signal | None]:
    close = df["close"].to_numpy(float)
    e20 = feats["ema20"].to_numpy(float)
    e50 = feats["ema50"].to_numpy(float)
    e200 = feats["ema200"].to_numpy(float)
    rsi = feats["rsi14"].to_numpy(float)
    adx = feats["adx14"].to_numpy(float)
    atr_pct = feats["atr_pct"].to_numpy(float)
    macd_hist = feats["macd_hist"].to_numpy(float)
    a = atr.to_numpy(float)
    n = len(df)

    out: list[Signal | None] = [None] * n
    L = slope_lookback
    for i in range(L, n):
        if np.isnan(e200[i]) or np.isnan(a[i]) or a[i] <= 0 or np.isnan(adx[i]):
            continue
        if adx[i] < adx_min:           # only trade real trends
            continue
        if atr_pct[i] > atr_pct_max:   # skip when too wild/choppy
            continue
        up_trend = e20[i] > e50[i] > e200[i]
        dn_trend = e20[i] < e50[i] < e200[i]
        e50_slope = e50[i] - e50[i - L]   # the medium trend must point our way

        # LONG: EMA20 reclaim, trend up + sloping up, momentum turning up (MACD>0)
        if allow_long and up_trend and e50_slope > 0 and macd_hist[i] > 0 \
                and close[i] > e20[i] and close[i - 1] <= e20[i - 1] \
                and 45.0 <= rsi[i] <= 65.0:
            stop = close[i] - atr_mult * a[i]
            target = close[i] + rr * atr_mult * a[i]
            out[i] = Signal(side=1, stop=stop, target=target,
                            reason=f"LONG trend-pullback: EMA20 reclaim, ADX {adx[i]:.0f}, RSI {rsi[i]:.0f}, MACD+")
            continue

        # SHORT: EMA20 loss, trend down + sloping down, momentum turning down (MACD<0)
        if allow_short and dn_trend and e50_slope < 0 and macd_hist[i] < 0 \
                and close[i] < e20[i] and close[i - 1] >= e20[i - 1] \
                and 35.0 <= rsi[i] <= 55.0:
            stop = close[i] + atr_mult * a[i]
            target = close[i] - rr * atr_mult * a[i]
            out[i] = Signal(side=-1, stop=stop, target=target,
                            reason=f"SHORT trend-rejection: EMA20 loss, ADX {adx[i]:.0f}, RSI {rsi[i]:.0f}, MACD-")
    return out
