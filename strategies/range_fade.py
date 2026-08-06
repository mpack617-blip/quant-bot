"""RANGE-FADE — make money when the market is FLAT (the gap snapback/continuation leave).

Snapback needs a trend + a counter-trend flush; continuation needs a breakout with
ADX strength. Both sit idle in a dead, range-bound market (low ADX, price chopping
between support/resistance, RSI hugging the middle) — which is *exactly* the regime
that produced two straight days of zero trades.

This angle profits from that regime: in a range, price oscillates between the
Bollinger bands and keeps reverting to the middle. So we FADE the band extremes
back to the mean.

  LONG  : NOT trending (ADX low), price pokes the LOWER band / oversold RSI, the bar
          REJECTS (closes back inside) and momentum turns up -> fade to the mid-band
  SHORT : mirror at the UPPER band / overbought RSI in a non-trending market

Quality filters (each toggleable + backtested) that lift a raw-breakeven fade into a
real edge in chop:
  - `wick_reject` : the bar must poke the band with its wick but CLOSE back inside —
                    a rejection of the extreme, not a clean break (which = breakdown).
  - `rsi_turn`    : RSI must be turning back (not just price) — confirms momentum flip.
  - bandwidth gate: only fade when the Bollinger band-width is in a sweet spot —
                    wide enough to have room to revert, not so wide it's really a trend.

Crucial guard: only fade when the market is genuinely range-bound (ADX < adx_max
and no clean EMA stack). Fading a strong trend = catching a falling knife, so the
ADX cap keeps us out of real trends (where snapback/continuation take over instead).

Target = the mid-band (SMA20), the centre of the range — frequent to hit => high
win rate. Stop sits just beyond the band extreme + an ATR buffer. Mean-reversion,
so the runner's trailing stays OFF (fixed target). No look-ahead: bar i uses only
data up to close i.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import Signal
from features.indicators import bollinger


def generate_signals(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    atr: pd.Series,
    *,
    adx_max: float = 18.0,        # ONLY fade when NOT trending (range regime) — tuned
    atr_pct_max: float = 4.0,     # only fade genuinely TIGHT ranges (skip high-vol "ranges")
    rsi_os: float = 38.0,         # oversold-ish for a long fade
    rsi_ob: float = 62.0,         # overbought-ish for a short fade
    bb_long: float = 0.15,        # at/below the lower band (close basis)
    bb_short: float = 0.85,       # at/above the upper band (close basis)
    atr_stop: float = 0.8,        # buffer beyond the band extreme
    min_rr: float = 0.8,          # skip if the mean is too close vs the stop
    max_rr_cap: float = 2.5,
    min_stretch: float = 0.3,     # price must be at least this % from the mid-band
    wick_reject: bool = True,     # require the bar to poke the band then close back inside
    rsi_turn: bool = True,        # require RSI (not just price) to be turning back
    bw_min: float = 0.0,          # min Bollinger band-width % (room to revert)
    bw_max: float = 100.0,        # max band-width % (above this it's a trend, not a range)
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
    bb = feats["bb_pos"].to_numpy(float)
    dist20 = feats["dist_ema20_pct"].to_numpy(float)
    atrp = feats["atr_pct"].to_numpy(float)
    lo_band, mid_band, hi_band = (s.to_numpy(float) for s in bollinger(df["close"], 20, 2.0))
    bw = (hi_band - lo_band) / np.where(mid_band == 0, np.nan, mid_band) * 100  # band-width %
    a = atr.to_numpy(float)
    n = len(df)
    out: list[Signal | None] = [None] * n

    for i in range(2, n):
        if np.isnan(e200[i]) or np.isnan(a[i]) or a[i] <= 0 or np.isnan(adx[i]) or np.isnan(mid_band[i]):
            continue
        if adx[i] >= adx_max:        # trending -> leave it to snapback/continuation
            continue
        if not np.isnan(atrp[i]) and atrp[i] > atr_pct_max:   # too volatile = not a clean range
            continue
        if not np.isnan(bw[i]) and not (bw_min <= bw[i] <= bw_max):  # band-width sweet spot
            continue
        stacked_up = e20[i] > e50[i] > e200[i]
        stacked_dn = e20[i] < e50[i] < e200[i]
        turning_up = close[i] > close[i - 1]
        turning_dn = close[i] < close[i - 1]
        rsi_up = rsi[i] > rsi[i - 1]
        rsi_dn = rsi[i] < rsi[i - 1]
        # rejection: the bar's wick poked the band, but it CLOSED back inside it
        long_reject = (low[i] <= lo_band[i]) and (close[i] > lo_band[i])
        short_reject = (high[i] >= hi_band[i]) and (close[i] < hi_band[i])
        sig: Signal | None = None

        # --- LONG: poked lower band / oversold, range (not a downtrend), rejecting + turning up ---
        if allow_long and not stacked_dn and turning_up \
                and (bb[i] <= bb_long or rsi[i] <= rsi_os) and dist20[i] <= -min_stretch \
                and (not wick_reject or long_reject) and (not rsi_turn or rsi_up):
            entry = close[i]
            stop = min(low[i], low[i - 1]) - atr_stop * a[i]
            target = mid_band[i]
            risk = entry - stop
            reward = target - entry
            if risk > 0 and reward > 0:
                rr = reward / risk
                if rr >= min_rr:
                    if rr > max_rr_cap:
                        target = entry + max_rr_cap * risk
                    sig = Signal(1, stop, target,
                                 f"RANGEFADE LONG: lower-band reject, RSI {rsi[i]:.0f}, ADX {adx[i]:.0f} (range)")

        # --- SHORT: poked upper band / overbought, range (not an uptrend), rejecting + turning down ---
        if sig is None and allow_short and not stacked_up and turning_dn \
                and (bb[i] >= bb_short or rsi[i] >= rsi_ob) and dist20[i] >= min_stretch \
                and (not wick_reject or short_reject) and (not rsi_turn or rsi_dn):
            entry = close[i]
            stop = max(high[i], high[i - 1]) + atr_stop * a[i]
            target = mid_band[i]
            risk = stop - entry
            reward = entry - target
            if risk > 0 and reward > 0:
                rr = reward / risk
                if rr >= min_rr:
                    if rr > max_rr_cap:
                        target = entry - max_rr_cap * risk
                    sig = Signal(-1, stop, target,
                                 f"RANGEFADE SHORT: upper-band reject, RSI {rsi[i]:.0f}, ADX {adx[i]:.0f} (range)")

        out[i] = sig
    return out
