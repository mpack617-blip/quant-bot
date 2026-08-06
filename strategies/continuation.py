"""CONTINUATION — a with-trend breakout angle to complement SNAPBACK.

Snapback only fires on a counter-trend FLUSH (a dip in an uptrend / pop in a
downtrend). In a steady GRIND where price never pulls back to the mean, snapback
sits idle. This angle covers that case: when a strong trend makes a fresh N-bar
high/low with momentum, ride the continuation.

  LONG  : EMA20>EMA50>EMA200, ADX strong, close breaks the prior-N-bar HIGH,
          momentum (MACD) up, RSI not already blown off  -> ride the grind up
  SHORT : mirror for downtrends

Fixed RR target (no trailing — matches the runner's USE_TRAIL=False). ML filters
quality. No look-ahead: bar i uses only data up to close i.
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
    adx_min: float = 25.0,
    lookback: int = 20,        # breakout channel length
    atr_stop: float = 1.5,
    rr: float = 1.8,
    rsi_long_max: float = 78.0,
    rsi_short_min: float = 22.0,
    allow_long: bool = True,
    allow_short: bool = True,
) -> list[Signal | None]:
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    e20 = feats["ema20"].to_numpy(float)
    e50 = feats["ema50"].to_numpy(float)
    e200 = feats["ema200"].to_numpy(float)
    rsi = feats["rsi14"].to_numpy(float)
    adx = feats["adx14"].to_numpy(float)
    mh = feats["macd_hist"].to_numpy(float)
    a = atr.to_numpy(float)
    n = len(df)

    hh = pd.Series(high).rolling(lookback).max().shift(1).to_numpy(float)
    ll = pd.Series(low).rolling(lookback).min().shift(1).to_numpy(float)

    out: list[Signal | None] = [None] * n
    for i in range(lookback + 1, n):
        if np.isnan(e200[i]) or np.isnan(a[i]) or a[i] <= 0 or np.isnan(adx[i]) or np.isnan(hh[i]):
            continue
        if adx[i] < adx_min:
            continue
        up = e20[i] > e50[i] > e200[i]
        dn = e20[i] < e50[i] < e200[i]
        sig: Signal | None = None

        if allow_long and up and mh[i] > 0 and close[i] > hh[i] and rsi[i] <= rsi_long_max:
            entry = close[i]; stop = entry - atr_stop * a[i]
            sig = Signal(1, stop, entry + rr * (entry - stop),
                         f"CONT LONG: {lookback}-bar high, ADX {adx[i]:.0f}, RSI {rsi[i]:.0f}")
        elif allow_short and dn and mh[i] < 0 and close[i] < ll[i] and rsi[i] >= rsi_short_min:
            entry = close[i]; stop = entry + atr_stop * a[i]
            sig = Signal(-1, stop, entry - rr * (stop - entry),
                         f"CONT SHORT: {lookback}-bar low, ADX {adx[i]:.0f}, RSI {rsi[i]:.0f}")

        out[i] = sig
    return out


def combined_signals(df, feats, atr, snapback_params: dict | None = None,
                     cont_params: dict | None = None,
                     range_params: dict | None = None) -> list[Signal | None]:
    """Merge SNAPBACK (priority — higher win-rate) with CONTINUATION, and (optionally)
    RANGE-FADE per bar. The three fire in near-disjoint regimes: snapback/continuation
    need a trend (ADX up); range-fade only fires in a flat range (ADX down). So this
    covers ALL regimes — including the dead, choppy market that left the bot idle.
    Priority where they ever overlap: snapback > continuation > range-fade."""
    from strategies.snapback import generate_signals as snap
    sp = snapback_params or {}
    cp = cont_params or {}
    s = snap(df, feats, atr, **sp)
    c = generate_signals(df, feats, atr, **cp)
    r = [None] * len(df)
    if range_params is not None:
        from strategies.range_fade import generate_signals as rfade
        r = rfade(df, feats, atr, **range_params)
    return [s[i] or c[i] or r[i] for i in range(len(df))]
