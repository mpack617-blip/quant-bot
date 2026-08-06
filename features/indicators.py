"""Custom technical-indicator library (no external TA dependency).

All functions take/return pandas Series/DataFrame and are vectorised.
Built by hand for full control and zero version-breakage with pandas 3.x.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def atr_pct(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """ATR as % of price — comparable across symbols."""
    return atr(df, n) / df["close"] * 100.0


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(s, fast) - ema(s, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(s, n)
    sd = s.rolling(n).std()
    return mid - k * sd, mid, mid + k * sd


def stochastic(df: pd.DataFrame, n: int = 14, d: int = 3):
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    k = 100 * (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan)
    return k, k.rolling(d).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average Directional Index — trend strength (0-100)."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df)
    atr_n = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def returns(s: pd.Series, n: int = 1) -> pd.Series:
    return s.pct_change(n)


def rolling_vol(s: pd.Series, n: int = 20) -> pd.Series:
    """Rolling stdev of log returns (volatility)."""
    return np.log(s / s.shift()).rolling(n).std()


def trend_label(df: pd.DataFrame) -> str:
    """Classify the latest bar's regime: STRONG UP / UP / CHOP / DOWN / STRONG DOWN.

    Uses EMA stack (20/50/200) + ADX, mirroring the JS analyze logic.
    """
    c = df["close"]
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    last = len(df) - 1
    px = c.iloc[last]
    adx_v = adx(df).iloc[last]
    stacked_up = e20.iloc[last] > e50.iloc[last] > e200.iloc[last]
    stacked_dn = e20.iloc[last] < e50.iloc[last] < e200.iloc[last]
    if adx_v < 18:
        return "CHOP"
    if stacked_up and px > e20.iloc[last]:
        return "STRONG UP" if adx_v > 28 else "UP"
    if stacked_dn and px < e20.iloc[last]:
        return "STRONG DOWN" if adx_v > 28 else "DOWN"
    return "UP" if px > e50.iloc[last] else "DOWN"


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build a full feature matrix from OHLCV — the input to strategies/ML."""
    c = df["close"]
    out = pd.DataFrame(index=df.index)
    out["close"] = c
    out["ret1"] = returns(c, 1)
    out["ret6"] = returns(c, 6)
    out["ret24"] = returns(c, 24)
    out["ema20"] = ema(c, 20)
    out["ema50"] = ema(c, 50)
    out["ema200"] = ema(c, 200)
    out["dist_ema20_pct"] = (c / out["ema20"] - 1) * 100
    out["dist_ema50_pct"] = (c / out["ema50"] - 1) * 100
    out["rsi14"] = rsi(c, 14)
    out["atr_pct"] = atr_pct(df, 14)
    out["adx14"] = adx(df, 14)
    macd_line, macd_sig, macd_hist = macd(c)
    out["macd_hist"] = macd_hist
    bb_lo, bb_mid, bb_hi = bollinger(c)
    out["bb_pos"] = (c - bb_lo) / (bb_hi - bb_lo).replace(0, np.nan)  # 0=lower band,1=upper
    out["vol20"] = rolling_vol(c, 20)
    k, d = stochastic(df)
    out["stoch_k"] = k
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.bybit import get_klines

    df = get_klines("BTCUSDT", "1h", bars=300)
    feats = feature_frame(df)
    print(feats.tail(3).T)
    print("\nlatest regime:", trend_label(df))
