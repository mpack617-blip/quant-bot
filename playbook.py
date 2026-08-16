"""The discipline layer: WHICH setups get taken, and WHEN the day is over.

WHY THIS MODULE EXISTS. The bot's entry model was already tuned (research_edge.py
picks the EV gate) and its risk manager already caps size. What nobody had measured
was the part between them: the quality filters a discretionary trader applies by
eye, and the daily rules that stop a bad session from becoming a bad week. Measured
over 197 days and 7,257 signals with a walk-forward gate (research_manage.py), that
gap was the difference between losing money and making it:

    live behaviour before     29% win   -0.018R/trade   24% of days ended <=1 loss
    with this module          80% win   +0.321R/trade   98% of days end <=1 loss

TWO INDEPENDENT IDEAS, and it matters that they are separate:

1. QUALITY FILTERS (`entry_veto`) — four conditions, each measured on its own and
   on both halves of the period, that mark setups whose losses are structural
   rather than unlucky. They cut trade count by roughly half.

2. THE LOSS BUDGET (`DayGuard`) — after the day's second losing trade, no new
   entries until tomorrow; after any loss, wait 4 hours. This is the single
   biggest effect in the whole study, and the reason is not psychology (a bot has
   none) — it is that losses CLUSTER. When a regime turns against the strategy it
   stays turned for hours, so the trade after a loss is measurably worse than the
   trade after a win. Capping the day turned -0.018R a trade into +0.217R on the
   same entries.

The numbers above assume the trade management in runner._manage (a third of the
position banked at +0.5R, stop to breakeven) — the three pieces were measured
together and are quoted together.

WHAT THIS MODULE DOES NOT CLAIM. It does not promise a loss-free day. It caps how
much a bad day can cost and it raises the share of days that end green (37% -> 56%
in the study); the rest is variance, and any tool that claims otherwise about
markets is selling something.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

import journal

# --- the day's loss budget -------------------------------------------------
# 2, not 1: at a cap of 1 the study earned the same R per day from fewer trades
# (+0.886 R/day either way), so the extra strictness bought nothing but idleness.
# With the cap at 2, 97.7% of days still ended with at most ONE loss on them, and
# no day in 197 had more than two.
MAX_DAILY_LOSSES = int(os.environ.get("QUANT_MAX_DAILY_LOSSES", "2"))
# Hours to stand down after a loss. Losses cluster; this refuses to re-enter into
# the regime that just took one. Measured worth ~+0.05R/trade on its own.
LOSS_COOLDOWN_H = float(os.environ.get("QUANT_LOSS_COOLDOWN_H", "4"))
# Optional: stop for the day once this many R are banked. Measured as a NET LOSS
# of expectancy (it truncates the days that pay for the rest), so it is OFF by
# default and only exposed because operators ask for it.
LOCK_DAY_AT_R = float(os.environ.get("QUANT_LOCK_DAY_R", "0") or 0)

# --- the quality filters ---------------------------------------------------
# Each figure is the measured R/trade of the setups the filter REMOVES, over 197
# days of walk-forward-gated signals. All four hold in both halves of the period.
#
# MIN_STOP_PCT: a stop closer than 1.25% of price is inside the bar's own noise and
# pays a crippling share of its R in fees — R is the stop distance, so a 0.75% stop
# pays three times the R-cost of a 2.25% one.
#     stop < 1.25%:  32% win, -0.22R    stop > 2%:  42% win, +0.16R
MIN_STOP_PCT = 0.0125
# HTF_ALIGN: never trade against the higher-timeframe trend. Counter-trend entries
# were the worst-behaved group in the study.
#     against HTF:  26% win, -0.325R    with HTF:  38% win, +0.015R
REQUIRE_HTF_ALIGN = True
# MIN_ATR_PCTILE: dead tape does not travel far enough to pay a 2R target. The two
# quietest volatility buckets both lose; the two liveliest both win.
#     ATR pctile < 0.6:  35% win, -0.09R    > 0.6:  40% win, +0.07R
MIN_ATR_PCTILE = 0.5
# SESSION: entries after 17:00 UTC were by far the worst hours (28% win, -0.335R
# over 249 trades) — the window after the US morning, when crypto liquidity thins
# and moves drift rather than trend. This is the weakest-evidence rule here (hour
# filters are the classic way to overfit a backtest), so it is a switch: it held in
# both halves (+0.379 / +0.257) but delete it first if the edge ever fades.
SKIP_HOURS_UTC = (17, 18, 19, 20, 21, 22, 23)


@dataclass
class DayState:
    losses: int
    wins: int
    net_pnl: float
    cooling_until: datetime | None
    blocked: bool
    reason: str


class DayGuard:
    """Owns the answer to 'may the bot open anything right now?'.

    Reads the journal rather than keeping its own counters, so a restart cannot
    hand the bot a fresh loss budget halfway through a bad day — the journal is
    rebuilt from the exchange on boot, and the budget is rebuilt from the journal.
    """

    def __init__(self, max_losses: int = MAX_DAILY_LOSSES,
                 cooldown_h: float = LOSS_COOLDOWN_H,
                 lock_at_r: float = LOCK_DAY_AT_R):
        self.max_losses = max_losses
        self.cooldown_h = cooldown_h
        self.lock_at_r = lock_at_r

    def state(self, now: datetime | None = None) -> DayState:
        now = now or datetime.now(timezone.utc)
        day = journal.day_summary(now.date().isoformat())
        cooling_until = None
        if day["last_loss_utc"] and self.cooldown_h:
            try:
                last = datetime.fromisoformat(day["last_loss_utc"])
                cooling_until = last.timestamp() + self.cooldown_h * 3600
                cooling_until = datetime.fromtimestamp(cooling_until, timezone.utc)
            except ValueError:
                cooling_until = None

        blocked, reason = False, ""
        if day["losses"] >= self.max_losses:
            blocked = True
            reason = (f"day's loss budget spent ({day['losses']}/{self.max_losses} losses) "
                      f"- no new entries until 00:00 UTC")
        elif cooling_until and now < cooling_until:
            mins = int((cooling_until - now).total_seconds() // 60)
            blocked = True
            reason = (f"cooling off {mins} min after a loss - losses cluster, "
                      f"the next setup in this regime is a worse bet")
        elif self.lock_at_r and day["r"] >= self.lock_at_r:
            blocked = True
            reason = f"day locked in at +{day['r']}R - protecting a green day"
        return DayState(losses=day["losses"], wins=day["wins"], net_pnl=day["net_pnl"],
                        cooling_until=cooling_until, blocked=blocked, reason=reason)

    def blocks(self, now: datetime | None = None) -> tuple[bool, str]:
        s = self.state(now)
        return s.blocked, s.reason


def entry_veto(*, entry: float, stop: float, side: int, ctx: dict,
               now: datetime | None = None) -> str | None:
    """Why this setup should NOT be taken — or None if it passes.

    `ctx` is the feature row the model scored (runner._last_row), so the filters see
    exactly the numbers the decision was made on. A missing feature never vetoes: a
    context read that failed must not silently stop the bot from trading.
    """
    now = now or datetime.now(timezone.utc)

    stop_pct = abs(entry - stop) / entry if entry else 0.0
    if stop_pct < MIN_STOP_PCT:
        return (f"stop only {stop_pct*100:.2f}% away (min {MIN_STOP_PCT*100:.2f}%) - "
                f"inside the noise, and fees eat {2*0.00075/max(stop_pct, 1e-9):.0%} of the R")

    if REQUIRE_HTF_ALIGN:
        htf = ctx.get("htf_trend")
        if htf is not None and htf == htf and htf * side < 0:   # NaN-safe
            return (f"against the higher-timeframe trend "
                    f"({'up' if htf > 0 else 'down'} vs a {'long' if side == 1 else 'short'})")

    pct = ctx.get("atr_pctile")
    if pct is not None and pct == pct and pct < MIN_ATR_PCTILE:
        return (f"volatility in the bottom {pct*100:.0f}% of its own range - "
                f"dead tape rarely reaches a 2R target")

    if now.hour in SKIP_HOURS_UTC:
        return (f"{now.hour:02d}:00 UTC is outside the traded session "
                f"(thin post-US hours: 28% win, -0.33R over 249 trades)")
    return None
