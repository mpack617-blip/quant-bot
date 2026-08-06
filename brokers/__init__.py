"""Execution brokers. PaperBroker = internal sim; TradingViewBroker = real
orders on the TradingView paper account via the tradingview-mcp scripts (CDP)."""
from .tradingview import TradingViewBroker, to_tv_symbol  # noqa: F401
