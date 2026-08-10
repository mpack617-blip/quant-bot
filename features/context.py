"""Context features — the situation a setup fires INTO.

The meta-model used to see only the candle in front of it: RSI, ADX, distance from
the EMAs. Measured on a year of signals (see research_edge.py / research_gate.py),
that feature set has no edge left after costs. What it was missing is context the
bot already computes for its own decisions but never fed the model:

  * WHICH lens fired, and what that setup's own reward:risk is
  * the higher timeframe (4h) trend the 1h setup sits inside
  * where BTC is, and whether this trade agrees with it
  * how volatile this coin is right now versus its own recent history
  * volume conviction, momentum slope, time of day

With these, EV-gated selection is positive in BOTH halves of the year; without
them it is not. So this module is the edge, and it lives in ONE place on purpose:
training and the live runner import the same functions. A context feature computed
slightly differently in the live path than in training silently feeds the model
something it never learned on, and the model's output stops meaning anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Bar-level context: everything derivable from this symbol's own candles plus BTC.
BAR_CONTEXT_COLS = [
    "dist_ema200_pct", "atr_pctile", "vol_z", "rsi_slope", "macd_slope",
    "htf_trend", "htf_dist_pct", "hour_sin", "hour_cos",
    "btc_ret24", "btc_above_ema50", "rel_str",
]
# Signal-level context: only knowable once a setup has actually fired.
SIGNAL_CONTEXT_COLS = ["setup_code", "nat_rr", "stop_atr", "with_btc"]

CONTEXT_COLS = BAR_CONTEXT_COLS + SIGNAL_CONTEXT_COLS


def setup_code(reason: str) -> int:
    """0 snapback (mean-revert flush) · 1 continuation (breakout) · 2 range-fade."""
    r = (reason or "").lower()
    if "snap" in r:
        return 0
    if "range" in r or "fade" in r:
        return 2
    return 1


def btc_context(btc_df: pd.DataFrame) -> pd.DataFrame:
    """Market backdrop from BTC's own candles, on the same timeframe."""
    c = btc_df["close"]
    return pd.DataFrame({
        "btc_ret24": c.pct_change(24) * 100,
        "btc_above_ema50": (c > c.ewm(span=50, adjust=False).mean()).astype(float),
    }, index=btc_df.index)


def htf_frame(df: pd.DataFrame, rule: str = "4h") -> pd.DataFrame:
    """Higher-timeframe trend, resampled from the same candles — no extra fetches.

    `.shift(1)` is what keeps it honest: at 10:00 the 08:00-12:00 bar is still
    forming, so only the last CLOSED higher-timeframe bar may be used.
    """
    h = df["close"].resample(rule).last()
    e20 = h.ewm(span=20, adjust=False).mean()
    e50 = h.ewm(span=50, adjust=False).mean()
    out = pd.DataFrame({"htf_trend": np.sign(e20 - e50), "htf_dist_pct": (h / e20 - 1) * 100})
    return out.shift(1).reindex(df.index, method="ffill")


def bar_context(df: pd.DataFrame, feats: pd.DataFrame,
                btc_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-bar context for a whole symbol. Same code path in training and live."""
    out = pd.DataFrame(index=df.index)
    ema200 = feats["ema200"]
    out["dist_ema200_pct"] = (df["close"] / ema200 - 1) * 100
    out["atr_pctile"] = feats["atr_pct"].rolling(200, min_periods=30).rank(pct=True)
    vol = df["volume"]
    out["vol_z"] = ((vol - vol.rolling(50).mean()) / vol.rolling(50).std()) \
        .replace([np.inf, -np.inf], np.nan)
    out["rsi_slope"] = feats["rsi14"].diff(3)
    out["macd_slope"] = feats["macd_hist"].diff(3)
    htf = htf_frame(df)
    out["htf_trend"] = htf["htf_trend"]
    out["htf_dist_pct"] = htf["htf_dist_pct"]
    hour = pd.Series(df.index.hour, index=df.index, dtype=float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    if btc_df is not None and len(btc_df):
        b = btc_context(btc_df).reindex(df.index).ffill()
        out["btc_ret24"] = b["btc_ret24"]
        out["btc_above_ema50"] = b["btc_above_ema50"]
        out["rel_str"] = df["close"].pct_change(24) * 100 - b["btc_ret24"]
    else:
        # No BTC data (a transient fetch failure) must not silently become "BTC is
        # flat and this trade agrees with it" — leave it missing. The model handles
        # NaN natively, and a missing value is honest where a zero would be a lie.
        out["btc_ret24"] = np.nan
        out["btc_above_ema50"] = np.nan
        out["rel_str"] = np.nan
    return out


def signal_context(sig, entry: float, atr_now: float, btc_above: float) -> dict:
    """Context that only exists once a setup has fired."""
    risk = abs(entry - sig.stop)
    return {
        "setup_code": setup_code(getattr(sig, "reason", "")),
        "nat_rr": abs(sig.target - entry) / risk if risk > 0 else np.nan,
        "stop_atr": risk / atr_now if atr_now else np.nan,
        "with_btc": (1.0 if (sig.side == 1) == (btc_above > 0.5) else 0.0)
        if btc_above == btc_above else np.nan,          # NaN-safe: NaN != NaN
    }
