"""Does the expected-value gate actually hold up? Robustness before deployment.

research_edge.py found that gating on EV = p*RR - (1-p) is the only configuration
with positive expectancy after costs on an honest time split. Before changing the
live bot, three questions have to be answered:

  1. Does it hold in BOTH halves of the year, or is it one lucky regime?
  2. Are the extra context features doing the work, or is the plain feature set
     (which the live bot already computes) just as good? Fewer moving parts wins
     ties — every feature added to the model is code that must run identically in
     the live path or the model is being fed something different from what it
     learned on.
  3. Which EV threshold survives both halves?

Run: python research_gate.py [bars]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config
import runner
from research_edge import BASE_COLS, EXTRA_COLS, wf_probs, net_R


def curve(d: pd.DataFrame, probs: np.ndarray, days: float, gate: str,
          grid) -> dict:
    ok = ~np.isnan(probs)
    p, rr = probs[ok], d["rr"].to_numpy()[ok]
    R = net_R(d)[ok]
    score = p if gate == "prob" else p * rr - (1 - p)
    out = {}
    for g in grid:
        m = score >= g
        if m.sum() < 25:
            continue
        r = R[m]
        w, l = r[r > 0], r[r < 0]
        out[g] = dict(n=int(m.sum()), per_day=m.sum() / days,
                      win=len(w) / len(r) * 100,
                      pf=(w.sum() / -l.sum()) if len(l) else np.inf,
                      exp=r.mean(), rpd=r.mean() * m.sum() / days)
    return out


def show(title: str, c: dict) -> None:
    print(f"\n{title}")
    print(f"{'gate':>6} {'n':>6} {'/day':>6} {'win%':>7} {'PF':>6} {'expR':>8} {'R/day':>8}")
    for g, v in c.items():
        print(f"{g:>6.2f} {v['n']:>6} {v['per_day']:>6.2f} {v['win']:>6.1f}% "
              f"{v['pf']:>6.2f} {v['exp']:>+8.3f} {v['rpd']:>+8.3f}")


def main() -> None:
    bars = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    interval = runner.DECISION_INTERVAL
    cache = config.ROOT / "cache" / f"research_{interval}_{bars}.pkl"
    if not cache.exists():
        print(f"no cached dataset — run: python research_edge.py {bars}")
        return
    data = pd.read_pickle(cache).sort_values("ts").reset_index(drop=True)
    days = (data["ts"].max() - data["ts"].min()).total_seconds() / 86400
    print(f"{len(data):,} signals over {days:.0f} days")

    ev_grid = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)
    pr_grid = (0.45, 0.5, 0.55, 0.6, 0.65, 0.7)

    sets = {"base (live features)": BASE_COLS, "base+context": BASE_COLS + EXTRA_COLS}
    probs = {}
    for name, cols in sets.items():
        p, d = wf_probs(data, cols, by_time=True)
        probs[name] = (p, d)
        show(f"--- {name}: gate on EV ---", curve(d, p, days, "ev", ev_grid))
        show(f"--- {name}: gate on P(win) ---", curve(d, p, days, "prob", pr_grid))

    # --- robustness: the same gate, measured separately in each half of the year ---
    print("\n" + "=" * 70)
    print("SAME GATE, EACH HALF OF THE PERIOD SEPARATELY")
    print("A setting that only works in one half is a regime, not an edge.")
    for name, (p, d) in probs.items():
        ok = ~np.isnan(p)
        mid = d["ts"].iloc[len(d) // 2]
        for label, mask in (("first half", d["ts"] < mid), ("second half", d["ts"] >= mid)):
            sel = ok & mask.to_numpy()
            sub_days = (d["ts"][sel].max() - d["ts"][sel].min()).total_seconds() / 86400
            c = curve(d[sel].reset_index(drop=True), p[sel], sub_days, "ev", ev_grid)
            show(f"--- {name} | EV gate | {label} ---", c)


if __name__ == "__main__":
    main()
