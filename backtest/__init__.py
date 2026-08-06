"""Backtesting engine + performance metrics."""
from .engine import Signal, Trade, BacktestResult, run_backtest  # noqa: F401
from .metrics import summarize  # noqa: F401
