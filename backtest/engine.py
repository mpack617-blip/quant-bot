"""Vectorised-ish, bar-by-bar backtest engine.

Contract
--------
A *strategy* produces, per bar, an optional entry `Signal(side, stop, target)`.
The engine:
  - enters at the NEXT bar's open (no look-ahead),
  - sizes the position so the loss at `stop` ≈ `risk_per_trade_pct` of equity,
  - applies taker fee + slippage on every fill,
  - manages the open position bar-by-bar against each bar's high/low,
  - supports move-to-breakeven after +1R and ATR trailing stops,
  - assumes STOP-first when a bar's range touches both stop and target (conservative).

It is deliberately single-position-per-symbol — portfolio aggregation is a thin
layer on top (run per symbol, combine equity curves) added later.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Signal:
    side: int          # +1 long, -1 short
    stop: float        # absolute stop price
    target: float      # absolute take-profit price
    reason: str = ""   # why this trade (for the journal / self-learning)


@dataclass
class Trade:
    symbol: str
    side: int
    entry_time: pd.Timestamp
    entry: float
    exit_time: pd.Timestamp
    exit: float
    qty: float
    pnl: float
    fees: float
    r_multiple: float | None
    reason: str
    exit_reason: str


@dataclass
class BacktestResult:
    symbol: str
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def run_backtest(
    df: pd.DataFrame,
    signals: list[Signal | None],
    *,
    symbol: str = "?",
    equity0: float = 1000.0,
    risk_per_trade_pct: float = 1.0,
    taker_fee: float = 0.00055,
    slippage_bps: float = 2.0,
    atr: pd.Series | None = None,
    move_to_be_after_r: float | None = 1.0,
    trail_atr_mult: float | None = 1.5,
    allow_short: bool = True,
) -> BacktestResult:
    """Simulate `signals` over `df` (OHLCV). Returns trades + per-bar equity curve.

    `signals[i]` is the strategy's decision using information available *at the
    close of bar i*; the fill happens at `open[i+1]`.
    """
    n = len(df)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    idx = df.index
    atr_arr = atr.to_numpy(float) if atr is not None else np.full(n, np.nan)
    slip = slippage_bps / 10_000.0

    equity = equity0
    equity_curve = np.empty(n, dtype=float)

    pos = None  # dict when in a trade
    trades: list[Trade] = []

    i = 0
    while i < n:
        # ---- manage an open position on bar i (entry bar included) ----
        if pos is not None:
            side = pos["side"]
            exit_price = None
            exit_reason = ""

            # 1) stop check (worst-case: assume the stop is touched before target)
            if side == 1 and low[i] <= pos["stop"]:
                exit_price, exit_reason = pos["stop"], "stop"
            elif side == -1 and h[i] >= pos["stop"]:
                exit_price, exit_reason = pos["stop"], "stop"
            # 2) target check
            elif side == 1 and h[i] >= pos["target"]:
                exit_price, exit_reason = pos["target"], "target"
            elif side == -1 and low[i] <= pos["target"]:
                exit_price, exit_reason = pos["target"], "target"

            if exit_price is not None:
                fill = exit_price * (1 - slip * side)  # slippage against us
                fee = abs(fill * pos["qty"]) * taker_fee
                pnl = (fill - pos["entry"]) * pos["qty"] * side - fee - pos["entry_fee"]
                equity += pnl
                risk0 = pos["risk_amount"]
                trades.append(Trade(
                    symbol=symbol, side=side, entry_time=pos["entry_time"], entry=pos["entry"],
                    exit_time=idx[i], exit=fill, qty=pos["qty"], pnl=pnl,
                    fees=fee + pos["entry_fee"],
                    r_multiple=(pnl / risk0) if risk0 else None,
                    reason=pos["reason"], exit_reason=exit_reason,
                ))
                pos = None
            else:
                # 3) trailing / breakeven management for FUTURE bars
                fav = (h[i] if side == 1 else low[i])
                pos["extreme"] = max(pos["extreme"], fav) if side == 1 else min(pos["extreme"], fav)
                move = (pos["extreme"] - pos["entry"]) * side
                if move_to_be_after_r and move >= move_to_be_after_r * pos["risk_per_unit"]:
                    be = pos["entry"]
                    pos["stop"] = max(pos["stop"], be) if side == 1 else min(pos["stop"], be)
                if trail_atr_mult and not np.isnan(atr_arr[i]):
                    trail = pos["extreme"] - side * trail_atr_mult * atr_arr[i]
                    pos["stop"] = max(pos["stop"], trail) if side == 1 else min(pos["stop"], trail)

        # ---- look for a new entry when flat ----
        if pos is None:
            sig = signals[i] if i < len(signals) else None
            if sig is not None and i + 1 < n and (sig.side == 1 or allow_short):
                entry_bar = i + 1
                raw = o[entry_bar]
                entry = raw * (1 + slip * sig.side)  # slippage against us
                risk_per_unit = abs(entry - sig.stop)
                if risk_per_unit > 0:
                    risk_amount = equity * risk_per_trade_pct / 100.0
                    qty = risk_amount / risk_per_unit
                    entry_fee = abs(entry * qty) * taker_fee
                    pos = {
                        "side": sig.side, "entry": entry, "entry_time": idx[entry_bar],
                        "stop": sig.stop, "target": sig.target, "qty": qty,
                        "risk_per_unit": risk_per_unit, "risk_amount": risk_amount,
                        "entry_fee": entry_fee, "extreme": entry, "reason": sig.reason,
                    }
                    # mark equity for the skipped bar, then jump to the entry bar
                    equity_curve[i] = equity
                    i = entry_bar
                    continue

        # mark-to-market equity (include open position's unrealised PnL)
        eq = equity
        if pos is not None:
            eq = equity + (c[i] - pos["entry"]) * pos["qty"] * pos["side"]
        equity_curve[i] = eq
        i += 1

    return BacktestResult(symbol=symbol, trades=trades,
                          equity_curve=pd.Series(equity_curve, index=idx))
