"""Market-regime gate — don't fight the whole market's direction.

BTC leads crypto: when BTC is in a decisive up/down trend, counter-trend trades
on alts get run over. This module reads BTC's own trend and returns a bias the
runner uses to BLOCK against-the-market entries (e.g. no alt longs while BTC is
trending down). Backtest-proven on 4h: lifts expectancy in both a flat and a
trending window (per-trade EV up, fewer-but-better trades) without overfitting —
it's a direction filter, not a parameter curve-fit.
"""
from __future__ import annotations

import pandas as pd

from data.bybit import get_klines_cached
from features.indicators import feature_frame

_LEAD = "BTCUSDT"


def bias_series(df: pd.DataFrame, feats: pd.DataFrame, slope_lookback: int = 10) -> pd.Series:
    """+1 = BTC up-trend, -1 = down-trend, 0 = no decisive trend. Per-bar, no
    look-ahead (each bar uses only its own close + EMA)."""
    e50 = feats["ema50"]
    price = df["close"]
    slope = e50 - e50.shift(slope_lookback)
    b = pd.Series(0, index=df.index)
    b[(price > e50) & (slope > 0)] = 1
    b[(price < e50) & (slope < 0)] = -1
    return b


def current_bias(interval: str = "4h") -> int:
    """Latest BTC bias for the live runner. 0 (no block) if data is unavailable."""
    try:
        df = get_klines_cached(_LEAD, interval, bars=300, max_age_min=240)
        feats = feature_frame(df)
        return int(bias_series(df, feats).iloc[-1])
    except Exception:  # noqa: BLE001
        return 0


def blocks(bias: int, side: int) -> bool:
    """True if an entry should be blocked because it fights a decisive market bias."""
    return (bias == 1 and side == -1) or (bias == -1 and side == 1)
