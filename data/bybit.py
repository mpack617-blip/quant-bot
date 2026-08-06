"""Bybit v5 market-data layer (public endpoints — no API key required).

Fetches historical and recent OHLCV klines, with backward pagination and a
local parquet/csv cache so we never re-download the same bars.
"""
from __future__ import annotations

import time
import sys
from pathlib import Path

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "quant-bot/0.1"})

KLINE_COLS = ["open", "high", "low", "close", "volume", "turnover"]


def _request_klines(symbol: str, interval: str, end_ms: int | None, limit: int = 1000,
                    category: str = config.DEFAULT_CATEGORY) -> list[list]:
    """One raw call to Bybit v5 /market/kline. Returns rows newest-first."""
    params = {"category": category, "symbol": symbol, "interval": interval, "limit": limit}
    if end_ms is not None:
        params["end"] = end_ms
    url = f"{config.DATA_BASE_URL}/v5/market/kline"
    for attempt in range(4):
        try:
            r = _SESSION.get(url, params=params, timeout=15)
            r.raise_for_status()
            body = r.json()
            if body.get("retCode") != 0:
                raise RuntimeError(f"Bybit error: {body.get('retCode')} {body.get('retMsg')}")
            return body["result"]["list"]
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    return []


_INTERVAL_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60,
                     "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440, "1w": 10080}


def _interval_minutes(interval: str) -> int | None:
    """Bar length in minutes for friendly or raw-Bybit interval codes."""
    if interval in _INTERVAL_MINUTES:
        return _INTERVAL_MINUTES[interval]
    if interval.isdigit():
        return int(interval)
    return {"D": 1440, "W": 10080, "M": 43200}.get(interval)


def get_klines(symbol: str, interval: str = "1h", bars: int = 1000,
               category: str = config.DEFAULT_CATEGORY) -> pd.DataFrame:
    """Fetch up to `bars` most-recent klines, paginating backwards as needed.

    `interval` accepts friendly codes ('1m','5m','15m','1h','4h','1d'...) or raw Bybit codes.
    Returns a DataFrame indexed by UTC timestamp, oldest-first, columns = KLINE_COLS.
    """
    iv = config.INTERVALS.get(interval, interval)
    collected: dict[int, list] = {}
    end_ms: int | None = None

    while len(collected) < bars:
        rows = _request_klines(symbol, iv, end_ms, limit=1000, category=category)
        if not rows:
            break
        for row in rows:
            ts = int(row[0])
            collected[ts] = row
        oldest = min(int(r[0]) for r in rows)
        if end_ms is not None and oldest >= end_ms:
            break  # no progress; stop
        end_ms = oldest - 1
        if len(rows) < 1000:
            break  # reached start of history
        time.sleep(0.12)  # gentle on the API

    if not collected:
        raise RuntimeError(f"No kline data returned for {symbol} {interval}")

    rows = sorted(collected.values(), key=lambda r: int(r[0]))[-bars:]
    df = pd.DataFrame(rows, columns=["ts"] + KLINE_COLS)
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    df[KLINE_COLS] = df[KLINE_COLS].astype(float)
    df = df.set_index("ts").sort_index()
    # Drop the still-forming candle: Bybit returns the current incomplete bar as
    # the last row. Strategies must only ever see CLOSED bars, else the signal
    # REPAINTS as the candle develops (looked like signals flickering on/off).
    mins = _interval_minutes(interval)
    if mins and len(df) > 1:
        bar_close = df.index[-1] + pd.Timedelta(minutes=mins)
        if bar_close > pd.Timestamp.now(tz="UTC"):
            df = df.iloc[:-1]
    df.attrs["symbol"] = symbol
    df.attrs["interval"] = interval
    return df


def get_klines_cached(symbol: str, interval: str = "1h", bars: int = 1000,
                      max_age_min: float = 5.0) -> pd.DataFrame:
    """Cache wrapper: re-use a parquet file if it's fresh enough, else refetch."""
    cache = config.CACHE_DIR / f"{symbol}_{interval}_{bars}.parquet"
    if cache.exists():
        age_min = (time.time() - cache.stat().st_mtime) / 60.0
        if age_min <= max_age_min:
            try:
                return pd.read_parquet(cache)
            except Exception:  # noqa: BLE001
                pass
    df = get_klines(symbol, interval, bars)
    try:
        df.to_parquet(cache)
    except Exception:  # noqa: BLE001 — parquet engine optional; cache is best-effort
        pass
    return df


def get_universe(symbols: list[str] | None = None, interval: str = "1h",
                 bars: int = 500) -> dict[str, pd.DataFrame]:
    """Fetch klines for a whole list of symbols. Returns {symbol: DataFrame}."""
    symbols = symbols or config.UNIVERSE
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            out[sym] = get_klines(sym, interval, bars)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {sym}: {e}")
    return out


if __name__ == "__main__":
    df = get_klines("BTCUSDT", "1h", bars=10)
    print(df.tail())
    print(f"\nrows={len(df)}  span: {df.index[0]} -> {df.index[-1]}")
