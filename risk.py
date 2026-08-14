"""Risk manager — the non-negotiable layer that sits between a signal and an order.

Enforces, from config.RISK:
  - per-trade risk sizing (qty so that loss at stop ≈ risk% of equity),
  - max concurrent positions,
  - total exposure cap (sum notional / equity),
  - daily-drawdown kill-switch (stop trading for the day after -X%).
Every rejection returns a human-readable reason for the journal/cockpit.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import config


@dataclass
class SizingDecision:
    approved: bool
    qty: float
    notional: float
    risk_amount: float
    reason: str


class RiskManager:
    def __init__(self, equity: float, risk_cfg: dict | None = None):
        self.start_equity = equity
        self.equity = equity
        self.cfg = risk_cfg or config.RISK
        self.day = datetime.now(timezone.utc).date()
        self.day_start_equity = equity
        self.halted = False

    # ---- daily kill-switch ----
    def _roll_day(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.day:
            self.day = today
            self.day_start_equity = self.equity
            self.halted = False

    def update_equity(self, equity: float) -> None:
        self.equity = equity
        self._roll_day()
        dd = (self.equity / self.day_start_equity - 1) * 100
        if dd <= -self.cfg["max_daily_drawdown_pct"]:
            self.halted = True

    def daily_drawdown_pct(self) -> float:
        return round((self.equity / self.day_start_equity - 1) * 100, 2)

    # ---- pre-trade gate ----
    def evaluate(self, *, side: int, entry: float, stop: float,
                 open_positions: list[dict], leverage: int | None = None,
                 target: float | None = None, risk_usd: float | None = None) -> SizingDecision:
        self._roll_day()
        lev = leverage or self.cfg["default_leverage"]

        if self.halted:
            return SizingDecision(False, 0, 0, 0,
                                  f"HALTED: daily drawdown {self.daily_drawdown_pct()}% hit kill-switch")
        if len(open_positions) >= self.cfg["max_concurrent_positions"]:
            return SizingDecision(False, 0, 0, 0,
                                  f"max concurrent positions ({self.cfg['max_concurrent_positions']}) reached")

        # --- how much of the book may point the same way ---------------------
        # Alts move together, so N simultaneous shorts are closer to one big short than
        # to N independent bets. The concurrency cap alone doesn't see that: it happily
        # allows 8 positions that are all the same side, and on 2026-08-11 all 8 live
        # trades were shorts and all 8 stopped.
        #
        # This is a DRAWDOWN control, not an edge filter — measured on 1,791 gated
        # signals over 373 days, per-trade expectancy barely moves (+0.660R uncapped
        # vs +0.629R at a cap of 5) but the tail shrinks a lot:
        #     cap  worst day   max DD   total R   totalR/DD
        #     none    -7.4R    -13.9R     632       45.5
        #        5    -5.3R    -10.0R     509       50.8   <- chosen
        #        4    -4.2R     -9.0R     451       50.3
        #        3    -3.6R     -7.9R     399       50.5
        # The uncapped worst day (-7.4R ~ -7.4% at 1% risk) blows straight through the
        # 6% daily kill-switch; a cap of 5 keeps it inside. The improvement is largest
        # in the weaker half of the period (totalR/DD 20.1 -> 28.8), which is what a
        # risk rule should do. The honest cost: ~20% less total R, because ~31% of
        # entries fire into a book that already leans this way.
        side_cap = self.cfg.get("max_same_side_positions")
        if side_cap:
            same = sum(1 for p in open_positions if p.get("side") == side)
            if same >= side_cap:
                direction = "long" if side == 1 else "short"
                return SizingDecision(False, 0, 0, 0,
                                      f"already {same} {direction} positions open "
                                      f"(same-side cap {side_cap}) - not adding correlated risk")

        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0:
            return SizingDecision(False, 0, 0, 0, "invalid stop (zero risk distance)")

        # --- how many dollars to risk on this trade ---
        # Default: % of equity, hard-capped at max_loss_usd (keeps the loss limited).
        # If the conviction engine passed a `risk_usd`, use THAT — the bot is sizing by
        # how good it thinks this specific setup is — but still clamp to a hard 10%-of-
        # equity ceiling so even a max-conviction trade can't blow up a small account.
        risk_amount = self.equity * self.cfg["max_risk_per_trade_pct"] / 100.0
        max_loss = self.cfg.get("max_loss_usd")
        if max_loss:
            risk_amount = min(risk_amount, max_loss)
        if risk_usd is not None:
            risk_amount = min(risk_usd, self.equity * 0.10)
        qty = risk_amount / risk_per_unit
        notional = qty * entry

        # exposure cap: total notional must stay within leverage * equity
        cur_notional = sum(abs(p.get("qty", 0) * p.get("entry", 0)) for p in open_positions)
        max_notional = lev * self.equity
        if cur_notional + notional > max_notional:
            room = max(0.0, max_notional - cur_notional)
            if room <= 0:
                return SizingDecision(False, 0, 0, 0,
                                      f"exposure cap: {lev}x equity already deployed")
            qty = room / entry
            notional = room
            risk_amount = qty * risk_per_unit

        return SizingDecision(True, round(qty, 8), round(notional, 2), round(risk_amount, 2),
                              f"OK: risk ${risk_amount:.2f} ({self.cfg['max_risk_per_trade_pct']}% eq), "
                              f"notional ${notional:.2f} ({notional/self.equity:.1f}x)")


if __name__ == "__main__":
    rm = RiskManager(equity=1000)
    d = rm.evaluate(side=-1, entry=100.0, stop=102.0, open_positions=[])
    print(d)
    rm.update_equity(940)  # -6% day
    print("halted after -6%:", rm.halted, rm.daily_drawdown_pct())
