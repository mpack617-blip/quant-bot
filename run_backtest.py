"""Run the live strategy backtest over one symbol or the whole universe.

Uses the SAME strategy the live runner trades (`multi_angle`), so the
before/after numbers actually reflect what the bot does. Pass a strategy name to
compare against the old `trend_pullback`.

Usage:
    python run_backtest.py                      # whole UNIVERSE, 15m, 1500 bars, multi_angle
    python run_backtest.py BTCUSDT 15m 2000     # one symbol/interval/bars
    python run_backtest.py ALL 15m 1500 trend_pullback   # old strategy for comparison
"""
from __future__ import annotations

import sys

import pandas as pd

import config
from data.bybit import get_klines
from features.indicators import feature_frame, atr
from strategies import multi_angle, trend_pullback
from backtest import run_backtest, summarize

STRATS = {"multi_angle": multi_angle, "trend_pullback": trend_pullback}


def backtest_symbol(symbol: str, interval: str, bars: int, strat) -> tuple[dict, list]:
    df = get_klines(symbol, interval, bars=bars)
    feats = feature_frame(df)
    a = atr(df, 14)
    signals = strat(df, feats, a)
    res = run_backtest(
        df, signals, symbol=symbol,
        equity0=1000.0,
        risk_per_trade_pct=config.RISK["max_risk_per_trade_pct"],
        taker_fee=config.BACKTEST["taker_fee"],
        slippage_bps=config.BACKTEST["slippage_bps"],
        atr=a,
        move_to_be_after_r=1.3,   # let it earn a cushion before going risk-free
        trail_atr_mult=2.5,        # wider trail so winners breathe
    )
    return summarize(res, interval), res.trades


def main() -> None:
    args = sys.argv[1:]
    interval = args[1] if len(args) > 1 else "4h"
    bars = int(args[2]) if len(args) > 2 else 2000
    strat_name = args[3] if len(args) > 3 else "multi_angle"
    strat = STRATS.get(strat_name, multi_angle)

    if args and args[0].upper() not in ("ALL", "UNIVERSE"):
        symbols = [args[0].upper()]
    else:
        symbols = config.UNIVERSE

    rows = []
    all_trades = []
    for sym in symbols:
        try:
            m, trades = backtest_symbol(sym, interval, bars, strat)
            m["symbol"] = sym
            rows.append(m)
            all_trades.extend(trades)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {sym}: {e}")

    if not rows:
        print("No results.")
        return

    table = pd.DataFrame(rows).set_index("symbol")
    cols = ["num_trades", "win_rate_pct", "profit_factor", "total_return_pct",
            "max_drawdown_pct", "sharpe", "avg_r", "expectancy_usd"]
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    print(f"\n=== {strat_name.upper()} BACKTEST — {interval}, {bars} bars, {len(symbols)} symbols ===\n")
    print(table[cols].sort_values("total_return_pct", ascending=False).to_string())

    # Portfolio-level aggregate across all trades
    pnls = pd.Series([t.pnl for t in all_trades])
    wins = pnls[pnls > 0]
    gross_w = wins.sum()
    gross_l = -pnls[pnls < 0].sum()
    pf = (gross_w / gross_l) if gross_l > 0 else float("inf")
    print("\n--- AGGREGATE (all symbols pooled) ---")
    print(f"  trades: {len(all_trades)}  | win-rate: {len(wins)/len(all_trades)*100:.1f}%"
          f"  | profit-factor: {pf:.2f}  | total net PnL: ${pnls.sum():.2f} on $1000/sym risk model")
    if len(wins) and (pnls < 0).any():
        print(f"  avg win: ${wins.mean():.2f}  | avg loss: ${pnls[pnls<0].mean():.2f}"
              f"  | expectancy: ${pnls.mean():.2f}/trade")


if __name__ == "__main__":
    main()
