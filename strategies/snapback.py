"""SNAPBACK — a high-win-rate strategy designed for the small $12 account.

Idea (combines everything the backtests proved today):
  - Trade WITH the dominant trend only (trend-following direction)         <- direction edge
  - ENTER only on a sharp COUNTER-trend flush to an oversold/overbought
    extreme (RSI + Bollinger + stretch from EMA20)                          <- mean-reversion timing
  - TARGET the mean (EMA20) — price reverting to its average is FREQUENT,
    so the target hits often => high win rate (small wins, many of them)    <- high hit-rate
  - STOP just beyond the extreme (the flush low/high) + ATR buffer
  - Accept that some trades lose — the edge is win-rate * payoff > 0

Why high win rate: snapping back to the moving average after a stretched flush
is one of the most repeatable short-term behaviours in trending crypto. We give
up the huge 3R runners (that almost never filled anyway) in exchange for lots of
small, high-probability mean-reversion wins — exactly what a small account that
wants frequent green trades needs.

No look-ahead: bar i uses only data up to close i. Emits Signal(side, stop,
target, reason); the engine fills next open and manages the exit.
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
    adx_min: float = 18.0,        # need a trend to lean on (but not too strict — we want frequency)
    rsi_os: float = 38.0,         # "oversold-ish" in an uptrend dip
    rsi_ob: float = 62.0,         # "overbought-ish" in a downtrend pop
    stretch_pct: float = 1.2,     # min |dist from EMA20| (%) — price must be flushed away from the mean
    bb_long: float = 0.25,        # near/below lower band
    bb_short: float = 0.75,       # near/above upper band
    atr_stop: float = 1.5,        # stop buffer beyond the flush extreme
    min_rr: float = 0.8,          # skip if target(mean) too close vs stop (keeps payoff sane)
    max_rr_cap: float = 3.0,      # don't let an abnormally far mean inflate the target
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
    dist20 = feats["dist_ema20_pct"].to_numpy(float)
    bb = feats["bb_pos"].to_numpy(float)
    a = atr.to_numpy(float)
    n = len(df)
    out: list[Signal | None] = [None] * n

    for i in range(2, n):
        if np.isnan(e200[i]) or np.isnan(a[i]) or a[i] <= 0 or np.isnan(adx[i]):
            continue
        if adx[i] < adx_min:
            continue
        up = e20[i] > e50[i] > e200[i]
        dn = e20[i] < e50[i] < e200[i]
        turning_up = close[i] > close[i - 1]
        turning_dn = close[i] < close[i - 1]
        sig: Signal | None = None

        # --- LONG: uptrend, flushed BELOW the mean to an oversold extreme, turning up ---
        if allow_long and up and turning_up \
                and dist20[i] <= -stretch_pct and (rsi[i] <= rsi_os or bb[i] <= bb_long):
            entry = close[i]
            stop = min(low[i], low[i - 1]) - atr_stop * a[i]
            target = e20[i]                      # revert to the mean
            risk = entry - stop
            reward = target - entry
            if risk > 0 and reward > 0:
                rr = reward / risk
                if rr >= min_rr:
                    if rr > max_rr_cap:
                        target = entry + max_rr_cap * risk
                    sig = Signal(1, stop, target,
                                 f"SNAPBACK LONG: dip {dist20[i]:.1f}% below EMA20, RSI {rsi[i]:.0f}, ADX {adx[i]:.0f}")

        # --- SHORT: downtrend, popped ABOVE the mean to an overbought extreme, turning down ---
        if sig is None and allow_short and dn and turning_dn \
                and dist20[i] >= stretch_pct and (rsi[i] >= rsi_ob or bb[i] >= bb_short):
            entry = close[i]
            stop = max(high[i], high[i - 1]) + atr_stop * a[i]
            target = e20[i]
            risk = stop - entry
            reward = entry - target
            if risk > 0 and reward > 0:
                rr = reward / risk
                if rr >= min_rr:
                    if rr > max_rr_cap:
                        target = entry - max_rr_cap * risk
                    sig = Signal(-1, stop, target,
                                 f"SNAPBACK SHORT: pop {dist20[i]:.1f}% above EMA20, RSI {rsi[i]:.0f}, ADX {adx[i]:.0f}")

        out[i] = sig
    return out
