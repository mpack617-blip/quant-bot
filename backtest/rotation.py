"""Cross-sectional Relative-Strength Rotation backtester (portfolio-level).

Idea (the strategy I pitched): instead of scanning each coin alone, RANK the whole
universe by momentum every rebalance and hold the strongest (long) / weakest
(short). You're always in what's actually moving, not the laggards.

This is returns-based (not stop/target): at each rebalance bar we pick target
weights; between rebalances we hold them; portfolio return each bar = weighted sum
of the coins' bar returns. Fees are charged on turnover (taker fee per unit traded).

No look-ahead: weights are decided at the close of a rebalance bar and only earn
returns from the NEXT bar (weights are shifted by 1 before multiplying returns).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _close_matrix(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """[time x symbol] close-price frame on the common (intersection) index."""
    cols = {sym: df["close"] for sym, df in prices.items() if len(df)}
    mat = pd.DataFrame(cols).dropna(how="any")
    return mat


def backtest_rotation(
    prices: dict[str, pd.DataFrame],
    *,
    interval: str = "1h",
    lookback: int = 168,        # momentum window in bars (168 1h bars = 7 days)
    rebalance: int = 24,        # rebalance every N bars (24 1h bars = daily)
    n_long: int = 3,
    n_short: int = 3,
    market_neutral: bool = True,  # True = long top + short bottom; False = long-only
    trend_filter: bool = True,    # long only if above EMA200, short only if below
    vol_adjust: bool = True,      # rank by return / volatility (risk-adjusted momentum)
    taker_fee: float = 0.00055,
    equity0: float = 1000.0,
) -> dict:
    mat = _close_matrix(prices)
    if mat.shape[0] < lookback + rebalance + 5 or mat.shape[1] < n_long + n_short:
        return {"error": "not enough aligned data", "bars": mat.shape[0], "syms": mat.shape[1]}

    rets = mat.pct_change()                      # bar-to-bar returns
    mom = mat.pct_change(lookback)               # momentum score
    if vol_adjust:
        vol = rets.rolling(lookback).std()
        mom = mom / vol.replace(0, np.nan)
    ema200 = mat.ewm(span=200, adjust=False).mean()
    above = mat > ema200                          # trend up?

    n = len(mat)
    syms = list(mat.columns)
    target = pd.DataFrame(0.0, index=mat.index, columns=syms)

    for i in range(lookback, n):
        if (i - lookback) % rebalance != 0:
            continue
        scores = mom.iloc[i].dropna()
        if scores.empty:
            continue
        ranked = scores.sort_values(ascending=False)
        longs = [s for s in ranked.index if scores[s] > 0]
        shorts = [s for s in ranked.index[::-1] if scores[s] < 0]
        if trend_filter:
            longs = [s for s in longs if above.iloc[i].get(s, False)]
            shorts = [s for s in shorts if not above.iloc[i].get(s, True)]
        longs = longs[:n_long]
        shorts = shorts[:n_short] if market_neutral else []

        w = pd.Series(0.0, index=syms)
        if longs:
            w[longs] = 1.0 / len(longs)            # long book sums to +1
        if shorts:
            w[shorts] = -1.0 / len(shorts)         # short book sums to -1
        # hold these weights until the next rebalance
        target.iloc[i:] = 0.0
        target.iloc[i:] = w.values

    # no look-ahead: weights earn the NEXT bar's return
    held = target.shift(1).fillna(0.0)
    port_ret = (held * rets).sum(axis=1)
    # turnover cost at each weight change (taker fee per unit traded)
    turnover = target.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * taker_fee
    net_ret = port_ret - cost

    equity = (1.0 + net_ret).cumprod() * equity0
    equity.iloc[0] = equity0

    # benchmark: equal-weight buy & hold the whole universe
    bench = (1.0 + rets.mean(axis=1).fillna(0.0)).cumprod() * equity0

    from backtest.metrics import sharpe, max_drawdown, cagr, _bars_per_year
    bpy = _bars_per_year(interval)
    n_rebals = int((target.diff().abs().sum(axis=1) > 0).sum())
    wins = (net_ret[held.abs().sum(axis=1) > 0] > 0)

    def _ret_pct(eq):
        return round((eq.iloc[-1] / eq.iloc[0] - 1) * 100, 2)

    return {
        "interval": interval, "lookback": lookback, "rebalance": rebalance,
        "n_long": n_long, "n_short": n_short, "market_neutral": market_neutral,
        "trend_filter": trend_filter, "vol_adjust": vol_adjust,
        "bars": n, "symbols": len(syms),
        "rebalances": n_rebals,
        "total_return_pct": _ret_pct(equity),
        "cagr_pct": round(cagr(equity, bpy) * 100, 2),
        "sharpe": round(sharpe(net_ret, bpy), 2),
        "max_drawdown_pct": round(max_drawdown(equity) * 100, 2),
        "bar_win_rate_pct": round(float(wins.mean()) * 100, 1) if len(wins) else 0.0,
        "total_fees_pct": round(float(cost.sum()) * 100, 2),
        "benchmark_return_pct": _ret_pct(bench),
        "benchmark_sharpe": round(sharpe(rets.mean(axis=1), bpy), 2),
        "benchmark_maxdd_pct": round(max_drawdown(bench) * 100, 2),
        "_equity": equity, "_bench": bench,
    }
