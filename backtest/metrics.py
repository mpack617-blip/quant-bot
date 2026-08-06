"""Performance metrics for a backtest — computed from the per-bar equity curve
and the list of closed trades. No external deps beyond numpy/pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _bars_per_year(interval: str) -> float:
    """Approximate number of bars in a calendar year for annualising Sharpe/CAGR."""
    minutes = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
        "1d": 1440, "1w": 10080,
    }.get(interval, 60)
    return (365.0 * 24 * 60) / minutes


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough drop, as a negative fraction (e.g. -0.23 = -23%)."""
    if len(equity) == 0:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def sharpe(returns: pd.Series, bars_per_year: float) -> float:
    """Annualised Sharpe of per-bar returns (risk-free = 0)."""
    r = returns.dropna()
    sd = r.std()
    if sd == 0 or np.isnan(sd) or len(r) < 2:
        return 0.0
    return float(r.mean() / sd * np.sqrt(bars_per_year))


def cagr(equity: pd.Series, bars_per_year: float) -> float:
    """Compound annual growth rate from the equity curve."""
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / bars_per_year
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def summarize(result, interval: str = "1h") -> dict:
    """Build a full metrics dict from a BacktestResult."""
    eq = result.equity_curve
    bpy = _bars_per_year(interval)
    rets = eq.pct_change() if len(eq) else pd.Series(dtype=float)

    trades = result.trades
    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (np.inf if gross_win > 0 else 0.0)
    win_rate = (len(wins) / len(trades)) if trades else 0.0
    expectancy = float(pnls.mean()) if len(pnls) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    rs = np.array([t.r_multiple for t in trades if t.r_multiple is not None], dtype=float)

    start_eq = float(eq.iloc[0]) if len(eq) else 0.0
    end_eq = float(eq.iloc[-1]) if len(eq) else 0.0

    return {
        "start_equity": round(start_eq, 2),
        "end_equity": round(end_eq, 2),
        "total_return_pct": round((end_eq / start_eq - 1) * 100, 2) if start_eq else 0.0,
        "cagr_pct": round(cagr(eq, bpy) * 100, 2),
        "sharpe": round(sharpe(rets, bpy), 2),
        "max_drawdown_pct": round(max_drawdown(eq) * 100, 2),
        "num_trades": len(trades),
        "win_rate_pct": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2) if np.isfinite(profit_factor) else float("inf"),
        "expectancy_usd": round(expectancy, 2),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "avg_r": round(float(rs.mean()), 2) if len(rs) else 0.0,
        "total_fees_usd": round(sum(t.fees for t in trades), 2),
    }
