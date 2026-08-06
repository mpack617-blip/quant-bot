"""Reset the bot to a fresh account — use after you reset the TradingView paper
account back to $12.

What it does (safe — it ARCHIVES, never silently destroys):
  1. Copies journal.db -> archive/journal_<timestamp>.db  (your trade history kept)
  2. Clears open positions + closed trades from journal.db (fresh slate)
  3. Writes a fresh paper_state.json with equity = start_equity = the amount you pass
     (default 12.0), so PnL is measured honestly from $12, not the old bogus $1000.

Usage:
    python reset_account.py            # reset to $12.00
    python reset_account.py 12         # same
    python reset_account.py 25         # reset to a different starting balance

After running, (re)start the cockpit/runner — it will pick up start_equity = 12
and begin from a clean journal. The ML model + lessons files are left intact so
the bot keeps what it learned; pass --wipe-model to also forget those.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone

import config

STATE_PATH = config.ROOT / "paper_state.json"
DB_PATH = config.ROOT / "journal.db"
ARCHIVE_DIR = config.ROOT / "archive"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def reset(start_equity: float = 12.0, wipe_model: bool = False) -> None:
    ARCHIVE_DIR.mkdir(exist_ok=True)
    stamp = _ts()

    # 1) archive the journal so history is never lost
    if DB_PATH.exists():
        dest = ARCHIVE_DIR / f"journal_{stamp}.db"
        shutil.copy2(DB_PATH, dest)
        print(f"archived trade history -> {dest.name}")

    # 2) clear positions + trades (keep the schema)
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as c:
            for tbl in ("trades", "positions"):
                try:
                    c.execute(f"DELETE FROM {tbl}")
                except sqlite3.OperationalError:
                    pass
            try:
                c.execute("DELETE FROM sqlite_sequence")  # reset autoincrement ids
            except sqlite3.OperationalError:
                pass
        print("cleared open positions + closed trades")

    # 3) fresh state file
    if STATE_PATH.exists():
        shutil.copy2(STATE_PATH, ARCHIVE_DIR / f"paper_state_{stamp}.json")
    fresh = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "running": False,
        "equity": round(start_equity, 2),
        "start_equity": round(start_equity, 2),
        "pnl": 0.0,
        "daily_dd_pct": 0.0,
        "halted": False,
        "open_positions": [],
        "stats": {"trades": 0, "win_rate_pct": 0.0, "net_pnl": 0.0, "profit_factor": 0.0},
        "market_sentiment": {"score": 0.0, "risk_off": False, "n": 0},
        "headlines": [], "watching": [], "activity": [], "scans": 0,
        "learning": {"live_trades_logged": 0, "model_active": True, "lessons_count": 0},
        "lessons": [], "last_note": f"reset to ${start_equity:.2f}", "tv_view": {},
    }
    STATE_PATH.write_text(json.dumps(fresh, indent=2))
    print(f"wrote fresh paper_state.json — equity = start_equity = ${start_equity:.2f}")

    if wipe_model:
        from ml.meta import MODEL_PATH
        for p in (MODEL_PATH,):
            if Path(p).exists():
                Path(p).unlink()
                print(f"wiped ML model {Path(p).name}")

    print("\nDONE. Reset your TradingView paper account to "
          f"${start_equity:.2f}, then start the bot — it begins clean from ${start_equity:.2f}.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    wipe = "--wipe-model" in sys.argv
    eq = float(args[0]) if args else 12.0
    from pathlib import Path  # noqa: E402  (only needed if --wipe-model)
    reset(eq, wipe)
