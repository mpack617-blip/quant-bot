"""Rule-based (and later ML) trading strategies. Each exposes
`generate_signals(df, feats, atr, **params) -> list[Signal | None]`.
"""
from .trend_pullback import generate_signals as trend_pullback  # noqa: F401
from .multi_angle import generate_signals as multi_angle  # noqa: F401
from .multi_angle import opportunity_snapshot  # noqa: F401
