"""Multi-angle opportunity scanner — look at the SAME timeframe from several
angles so the bot actually finds setups instead of waiting for one rare pattern.

Why this exists: the original `trend_pullback` only fired on the exact bar where
price closed back above EMA20 (a single-bar "reclaim"). In a strong, grinding
trend price rides ABOVE EMA20 and never gives that bar — so 0 signals fired even
though the market was full of opportunity. This module keeps that edge but adds
two more lenses, and picks the best one per bar.

Angles (each long + short symmetric, all gated by ADX trend strength + ATR%):
  1. PULLBACK  — with-trend shallow dip near EMA20 in an established trend,
                 momentum already turning our way. No cross-below required.
  2. BREAKOUT  — price breaks the prior N-bar high/low with the trend and
                 positive momentum: rides STRONG UP / STRONG DOWN grinds.
  3. MEANREV   — counter-trend fade only at a stretched extreme (RSI + Bollinger
                 + distance from EMA20). Lower priority, tighter target.

Priority when several fire on a bar: PULLBACK > BREAKOUT > MEANREV (proven edge
first, counter-trend last). No look-ahead: bar i uses only data up to close i.
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
    adx_min: float = 30.0,        # tuned on 4h: PF 0.85(15m)->1.27(4h) at ADX>=30
    adx_breakout: float = 30.0,
    atr_mult: float = 2.5,        # wider stop = far fewer premature stop-outs
    rr: float = 2.0,              # RR 2 (was 3): the 3:1 target almost never filled
    atr_pct_max: float = 6.0,
    slope_lookback: int = 5,
    pullback_band: float = 2.5,    # |dist from EMA20| (%) that still counts as "at" EMA20
    breakout_lookback: int = 20,   # bars for the high/low channel breakout
    mr_rsi_hi: float = 80.0,
    mr_rsi_lo: float = 20.0,
    mr_stretch: float = 4.0,       # min |dist from EMA20| (%) to fade
    mr_rr: float = 1.5,
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
    atr_pct = feats["atr_pct"].to_numpy(float)
    macd_hist = feats["macd_hist"].to_numpy(float)
    dist20 = feats["dist_ema20_pct"].to_numpy(float)
    bb_pos = feats["bb_pos"].to_numpy(float)
    a = atr.to_numpy(float)
    n = len(df)

    # prior-N-bar channel (shifted by 1 so bar i compares to bars before it)
    s = pd.Series(high)
    hh = s.rolling(breakout_lookback).max().shift(1).to_numpy(float)
    ll = pd.Series(low).rolling(breakout_lookback).min().shift(1).to_numpy(float)

    out: list[Signal | None] = [None] * n
    L = slope_lookback
    start = max(L, breakout_lookback + 1)
    for i in range(start, n):
        if np.isnan(e200[i]) or np.isnan(a[i]) or a[i] <= 0 or np.isnan(adx[i]):
            continue
        if atr_pct[i] > atr_pct_max:          # too wild — skip every angle
            continue
        up = e20[i] > e50[i] > e200[i]
        dn = e20[i] < e50[i] < e200[i]
        slope = e50[i] - e50[i - L]

        sig: Signal | None = None

        # --- Angle 1: PULLBACK (with-trend shallow dip near EMA20) ---
        # Momentum only has to still point our way (macd_hist sign); during a
        # pullback the histogram is naturally easing off, so we don't demand it rise.
        if adx[i] >= adx_min:
            if allow_long and up and slope > 0 and macd_hist[i] > 0 \
                    and -0.3 <= dist20[i] <= pullback_band and 45.0 <= rsi[i] <= 72.0:
                stop = close[i] - atr_mult * a[i]
                sig = Signal(1, stop, close[i] + rr * atr_mult * a[i],
                             f"PULLBACK LONG: dip to EMA20, ADX {adx[i]:.0f}, RSI {rsi[i]:.0f}, MACD+")
            elif allow_short and dn and slope < 0 and macd_hist[i] < 0 \
                    and -pullback_band <= dist20[i] <= 0.3 and 28.0 <= rsi[i] <= 55.0:
                stop = close[i] + atr_mult * a[i]
                sig = Signal(-1, stop, close[i] - rr * atr_mult * a[i],
                             f"PULLBACK SHORT: pop to EMA20, ADX {adx[i]:.0f}, RSI {rsi[i]:.0f}, MACD-")

        # --- Angle 2: BREAKOUT (channel break with the trend) ---
        if sig is None and adx[i] >= adx_breakout and not np.isnan(hh[i]):
            if allow_long and up and macd_hist[i] > 0 and close[i] > hh[i] and rsi[i] <= 78.0:
                stop = close[i] - atr_mult * a[i]
                sig = Signal(1, stop, close[i] + rr * atr_mult * a[i],
                             f"BREAKOUT LONG: {breakout_lookback}-bar high, ADX {adx[i]:.0f}, RSI {rsi[i]:.0f}")
            elif allow_short and dn and macd_hist[i] < 0 and close[i] < ll[i] and rsi[i] >= 22.0:
                stop = close[i] + atr_mult * a[i]
                sig = Signal(-1, stop, close[i] - rr * atr_mult * a[i],
                             f"BREAKOUT SHORT: {breakout_lookback}-bar low, ADX {adx[i]:.0f}, RSI {rsi[i]:.0f}")

        # --- Angle 3: MEANREV (counter-trend fade at a stretched extreme) ---
        if sig is None:
            if allow_short and rsi[i] >= mr_rsi_hi and bb_pos[i] >= 1.0 and dist20[i] >= mr_stretch:
                stop = high[i] + atr_mult * a[i]
                sig = Signal(-1, stop, close[i] - mr_rr * atr_mult * a[i],
                             f"MEANREV SHORT: overbought RSI {rsi[i]:.0f}, +{dist20[i]:.1f}% from EMA20")
            elif allow_long and rsi[i] <= mr_rsi_lo and bb_pos[i] <= 0.0 and dist20[i] <= -mr_stretch:
                stop = low[i] - atr_mult * a[i]
                sig = Signal(1, stop, close[i] + mr_rr * atr_mult * a[i],
                             f"MEANREV LONG: oversold RSI {rsi[i]:.0f}, {dist20[i]:.1f}% from EMA20")

        out[i] = sig
    return out


def opportunity_snapshot(df: pd.DataFrame, feats: pd.DataFrame, atr: pd.Series) -> dict:
    """Latest-bar, human-readable readout of where the opportunity is, from every
    angle — for the cockpit. Returns {setup, side, score(0-100), note}.

    `score` blends trend strength (ADX), trend alignment, momentum and how close
    price is to a trigger, so the operator can see *potential* even on bars where
    no full signal has fired yet.
    """
    i = len(df) - 1
    f = feats.iloc[i]
    adx = float(f["adx14"]); rsi = float(f["rsi14"]); d20 = float(f["dist_ema20_pct"])
    mh = float(f["macd_hist"]); e20 = float(f["ema20"]); e50 = float(f["ema50"])
    e200 = float(f["ema200"])
    up = e20 > e50 > e200
    dn = e20 < e50 < e200

    sig = generate_signals(df, feats, atr)[-1]
    setup = side = None
    note = ""
    if sig is not None:
        setup = sig.reason.split(":")[0]            # e.g. "BREAKOUT LONG"
        side = "LONG" if sig.side == 1 else "SHORT"
        note = sig.reason

    # opportunity score: how attractive is *this* coin to act on right now
    score = 0.0
    aligned = up or dn
    if aligned:
        score += min(adx, 60) / 60 * 45             # trend strength up to 45
        score += 15 if ((up and mh > 0) or (dn and mh < 0)) else 0   # momentum agrees
        prox = max(0.0, 1 - abs(d20) / 2.0)         # near EMA20 = closer to a dip entry
        score += prox * 20
    # extreme = mean-reversion opportunity
    if rsi >= 78 or rsi <= 22:
        score = max(score, 55 + min(abs(rsi - 50) - 28, 20))
    if setup:
        score = max(score, 80)                      # a live signal is the real thing
    return {"setup": setup, "side": side, "score": round(min(score, 100)), "note": note}
