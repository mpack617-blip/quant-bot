"""Conviction engine — the bot's own read on 'how good is THIS trade?'.

Instead of one fixed size/stop/target for every signal, the bot now SCORES each
setup and sizes it by its own conviction — exactly how a discretionary trader
leans harder into a high-potential trade and lightly into a marginal one.

Conviction (0..1) blends three honest edges already computed elsewhere:
  - ML win-probability  (`ml.meta.predict_proba`)  — the learned edge   [0.45]
  - trend strength      (ADX)                       — momentum behind it [0.30]
  - the setup's natural reward:risk                 — structural payoff  [0.25]

It then derives the trade plan:
  - risk_usd : dollars to risk  -> drives QUANTITY (more on high-edge trades)
  - stop     : pulled a touch TIGHTER on high conviction (strong trend retraces
               shallowly) -> with the same $ risk this means a BIGGER quantity
  - target   : take-profit set FURTHER out on high conviction (let it pay more)

Loss stays LIMITED (risk_usd is bounded RISK_MIN..RISK_MAX) and a planned winner
is never below MIN_WIN. Trailing (in runner._manage) then protects/rides it.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- tunables ---
# Risk per trade is a PERCENTAGE of equity, not a fixed number of dollars. The old
# fixed $0.55-$0.80 was tuned for a $12 account; on a $50,000 account that is 0.0016%
# per trade — the bot would be "running" while risking nothing and compounding nothing.
# A percentage scales with the account in both directions: it grows the size as the
# account grows, and shrinks it after a drawdown, which is the behaviour that keeps a
# losing streak survivable.
#
# 0.5% -> 1.0% of equity, marginal -> high conviction. At 0.5% risk, twenty consecutive
# losses cost about 10% of the account; that is a bad month, not a dead account.
RISK_PCT_MIN, RISK_PCT_MAX = 0.005, 0.010
# Floor in dollars, for accounts too small for the percentage to clear the exchange's
# minimum order value. Ceiling is a hard stop against a fat-fingered equity read.
RISK_FLOOR_USD, RISK_CEIL_USD = 0.55, 500.0
RISK_MIN, RISK_MAX = 0.55, 0.80    # legacy fixed band, used only if equity is unknown
RR_MIN, RR_MAX = 1.85, 3.0         # take-profit reward:risk: marginal -> high conviction
STOP_TIGHTEN = 0.20                # up to 20% tighter stop at max conviction (more qty, same $ risk)
MIN_WIN = 1.0                      # never PLAN a winner below $1 (user's floor)

# blend weights for the conviction score
W_ML, W_ADX, W_RR = 0.45, 0.30, 0.25

# How much the FORWARD-LOOKING evidence (next-move forecast + live microstructure)
# is allowed to move conviction, as a signed nudge in [-W_FWD, +W_FWD].
# Held to 0.15 on purpose: the forecaster's measured out-of-sample edge is real but
# small (AUC 0.529), so it deserves a vote, not a veto over the tuned three above.
W_FWD = 0.15


@dataclass
class Plan:
    conviction: float   # 0..1
    risk_usd: float     # dollars to risk (drives qty)
    stop: float         # conviction-adjusted stop price
    target: float       # conviction-adjusted take-profit price
    rr: float           # planned reward:risk


def _clip01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def score(prob: float, adx: float, natural_rr: float) -> float:
    """The bot's 0..1 read on this setup's potential."""
    ml_term = _clip01((prob - 0.30) / 0.40)       # 0.30 -> 0, 0.70 -> 1
    adx_term = _clip01((adx - 15.0) / 25.0)        # 15 -> 0, 40 -> 1
    rr_term = _clip01((natural_rr - 1.5) / 1.5)    # 1.5 -> 0, 3.0 -> 1
    return W_ML * ml_term + W_ADX * adx_term + W_RR * rr_term


def assess(*, prob: float, adx: float, natural_rr: float,
           side: int, entry: float, stop: float, fwd_agree: float = 0.0,
           equity: float | None = None) -> Plan:
    """Turn a raw signal into a conviction-sized plan (qty $, stop, target).

    `fwd_agree` is the forward-looking evidence already SIGNED for this trade's
    direction: +1 = the next-move forecast and live order flow both back this side,
    -1 = they both lean against it, 0 = no read. It nudges conviction (and therefore
    size and target) but cannot by itself create or kill a trade.
    """
    c = _clip01(score(prob, adx, natural_rr) + W_FWD * max(-1.0, min(1.0, fwd_agree)))

    if equity and equity > 0:
        pct = RISK_PCT_MIN + c * (RISK_PCT_MAX - RISK_PCT_MIN)
        risk_usd = min(max(equity * pct, RISK_FLOOR_USD), RISK_CEIL_USD)
    else:
        risk_usd = RISK_MIN + c * (RISK_MAX - RISK_MIN)
    rr = RR_MIN + c * (RR_MAX - RR_MIN)
    # The $1 minimum-win floor only makes sense on a tiny account. On a real balance
    # the percentage sizing already dwarfs it, and forcing it would distort the target.
    if risk_usd < 2.0:
        rr = max(rr, MIN_WIN / risk_usd)           # guarantee the PLANNED win >= $1

    stop_dist = abs(entry - stop)
    stop_dist_adj = stop_dist * (1.0 - STOP_TIGHTEN * c)   # tighter when confident
    if stop_dist_adj <= 0:
        stop_dist_adj = stop_dist
    new_stop = entry - side * stop_dist_adj
    target = entry + side * rr * stop_dist_adj

    return Plan(round(c, 3), round(risk_usd, 4), new_stop, target, round(rr, 2))


if __name__ == "__main__":
    for prob, adx, nrr, lbl in [(0.35, 18, 1.6, "marginal"),
                                (0.50, 28, 2.2, "decent"),
                                (0.65, 38, 2.8, "strong")]:
        p = assess(prob=prob, adx=adx, natural_rr=nrr, side=1, entry=100.0, stop=98.0)
        win = p.risk_usd * p.rr
        print(f"{lbl:9s} conv {p.conviction:.2f} | risk ${p.risk_usd:.2f} "
              f"| rr {p.rr} | PLAN win ${win:.2f} / loss ${p.risk_usd:.2f} "
              f"| stop {p.stop:.3f} tgt {p.target:.3f}")
