"""Run the Relative-Strength Rotation backtest over the whole universe.

Usage: python run_rotation.py [interval] [bars]
"""
from __future__ import annotations

import sys

import config
from data.bybit import get_klines
from backtest.rotation import backtest_rotation


def fetch(interval: str, bars: int) -> dict:
    prices = {}
    for sym in config.UNIVERSE:
        try:
            prices[sym] = get_klines(sym, interval, bars=bars)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {sym}: {e}")
    return prices


def show(r: dict) -> None:
    if "error" in r:
        print("  ERROR:", r); return
    print(f"  strategy : ret {r['total_return_pct']:+.1f}%   Sharpe {r['sharpe']}   "
          f"maxDD {r['max_drawdown_pct']}%   bar-win {r['bar_win_rate_pct']}%   "
          f"fees {r['total_fees_pct']}%   ({r['rebalances']} rebalances)")
    print(f"  buy&hold : ret {r['benchmark_return_pct']:+.1f}%   Sharpe {r['benchmark_sharpe']}   "
          f"maxDD {r['benchmark_maxdd_pct']}%")
    edge = "BEATS hold [+]" if r['total_return_pct'] > r['benchmark_return_pct'] else "loses to hold [-]"
    sh = "better risk-adj [+]" if r['sharpe'] > r['benchmark_sharpe'] else "worse risk-adj [-]"
    print(f"  verdict  : {edge} | {sh}")


def main() -> None:
    interval = sys.argv[1] if len(sys.argv) > 1 else "1h"
    bars = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    print(f"fetching {len(config.UNIVERSE)} coins @ {interval}, {bars} bars ...")
    prices = fetch(interval, bars)
    print(f"got {len(prices)} coins\n")

    lb = {"15m": 96, "1h": 168, "4h": 180, "1d": 30}.get(interval, 168)
    rb = {"15m": 16, "1h": 24, "4h": 6, "1d": 1}.get(interval, 24)

    variants = [
        ("L/S market-neutral + trend + vol-adj", dict(market_neutral=True, trend_filter=True, vol_adjust=True)),
        ("L/S market-neutral (raw momentum)",     dict(market_neutral=True, trend_filter=False, vol_adjust=False)),
        ("LONG-only top-N + trend + vol-adj",      dict(market_neutral=False, trend_filter=True, vol_adjust=True)),
    ]
    for name, kw in variants:
        print(f"=== {name} ===")
        r = backtest_rotation(prices, interval=interval, lookback=lb, rebalance=rb,
                              n_long=3, n_short=3, **kw)
        show(r)
        print()


if __name__ == "__main__":
    main()
