"""One scan, then exit — the entry point for GitHub Actions (or any cron).

The normal `runner.py` is a process that never stops: it loops, sleeps, loops.
A CI runner is the opposite — it boots, gets a few minutes, and dies. So this does
exactly one `tick()` and returns, leaving every piece of state on disk
(`journal.db`, `paper_state.json`) for the next run to pick up.

WHAT THIS TRADES OFF, honestly:
  - GitHub's cron is best-effort. A schedule of "every 10 minutes" can arrive 20+
    minutes late when their runners are busy. Entries will sometimes be missed.
  - That is survivable ONLY because the stop-loss and take-profit live on Bybit's
    servers (see brokers/bybit.py). The exchange closes a losing trade on time even
    if this job never wakes up. Without exchange-side stops, running a bot on a
    delayed cron would be reckless.
  - So: entries are best-effort, risk management is not.
"""
from __future__ import annotations

import os
import sys
import time

from runner import QuantRunner, _log


def main() -> int:
    mode = os.environ.get("QUANT_MODE", "bybit")
    t0 = time.time()

    r = QuantRunner(mode=mode)
    if mode == "bybit" and not r.exchange_info.get("authenticated"):
        # Fail loudly. A green tick on a job that silently traded nothing is worse
        # than a red one — you would believe the bot was working for weeks.
        print(f"::error::Bybit not authenticated: {r.exchange_info.get('error')}")
        return 1

    print(f"mode={mode} book=${r.equity} real=${r.real_equity} "
          f"open={len(__import__('journal').open_positions())}")

    try:
        r.tick()
    except Exception as e:  # noqa: BLE001
        print(f"::error::tick failed: {e}")
        _log(f"[tick_once] FAILED: {e}")
        raise

    print(f"tick done in {time.time() - t0:.0f}s | {r.last_note}")
    for line in r.activity_log[-3:]:
        print("  ", line)
    if r.forecasts and r.forecasts[0].get("model_ready", True):
        f = r.forecasts[0]
        print(f"  next move: {f['symbol']} {f['direction']} {f['prob']}% in {f['horizon']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
