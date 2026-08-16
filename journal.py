"""Self-learning trade journal (SQLite, stdlib only).

Every closed trade is recorded with the market context (features) at entry and
its outcome. After a LOSS, an automatic post-mortem records *why* it likely
failed (structured heuristics over the entry context). A rolling "lessons"
summary is exposed so both the ML retrainer and the human (via the cockpit
chat) can see what the bot has learned. This is the automated version of the
discretionary bot's hand-written T3/T4 lessons.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import config

DB_PATH = config.ROOT / "journal.db"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT, side INTEGER, entry REAL, exit REAL, qty REAL,
                pnl REAL, r_multiple REAL, reason TEXT, exit_reason TEXT,
                opened_utc TEXT, closed_utc TEXT,
                context TEXT,           -- json of entry features
                postmortem TEXT,        -- why it lost (null if win)
                paper INTEGER DEFAULT 1
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE, side INTEGER, entry REAL, qty REAL,
                stop REAL, target REAL, opened_utc TEXT, reason TEXT, context TEXT
            )""")
        # WHERE a position actually lives: 'bybit' | 'tradingview' | 'paper' | 'legacy'.
        # Without this, running the bot in Bybit mode would look at a TradingView
        # position, fail to find it on Bybit, and "reconcile" it closed — silently
        # destroying the record of a live trade on another venue. Added 2026-08-06.
        cols = {r[1] for r in c.execute("PRAGMA table_info(positions)")}
        if "venue" not in cols:
            c.execute("ALTER TABLE positions ADD COLUMN venue TEXT DEFAULT 'legacy'")
        # Has the +0.5R partial already been banked on this position? Without a flag
        # the runner would re-sell a third of the trade on every tick that price sat
        # above the level, walking a winner out of the market a slice at a time.
        if "partial" not in cols:
            c.execute("ALTER TABLE positions ADD COLUMN partial INTEGER DEFAULT 0")
        # The exchange's own id for a closed trade ("SYMBOL:updatedTime"). It is what
        # makes re-importing history IDEMPOTENT: the bot re-reads its whole trade
        # history from Bybit on every boot (the host's disk is ephemeral and loses
        # this file), and without a stable id every restart would duplicate the lot.
        tcols = {r[1] for r in c.execute("PRAGMA table_info(trades)")}
        if "ext_id" not in tcols:
            c.execute("ALTER TABLE trades ADD COLUMN ext_id TEXT")
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_ext ON trades(ext_id) "
                  "WHERE ext_id IS NOT NULL")


def _postmortem(side: int, ctx: dict, exit_reason: str) -> str:
    """Heuristic 'why did this lose' — the seed of the self-learning lessons."""
    notes = []
    rsi = ctx.get("rsi14")
    adx = ctx.get("adx14")
    atrp = ctx.get("atr_pct")
    if exit_reason == "stop":
        notes.append("hit stop")
    if adx is not None and adx < 25:
        notes.append(f"weak trend (ADX {adx:.0f}) — chop risk, prefer ADX>25")
    if atrp is not None and atrp > 4:
        notes.append(f"high volatility (ATR {atrp:.1f}%) — stop too tight for the noise")
    if side == -1 and rsi is not None and rsi < 35:
        notes.append(f"shorted into oversold (RSI {rsi:.0f}) — the T3/T4 mistake, wait for rollover")
    if side == 1 and rsi is not None and rsi > 65:
        notes.append(f"bought overbought (RSI {rsi:.0f}) — chased strength")
    return "; ".join(notes) or "stopped within normal variance (no rule broken)"


def record_open(symbol, side, entry, qty, stop, target, reason, context,
                venue: str = "paper") -> None:
    init_db()
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO positions"
                  "(symbol,side,entry,qty,stop,target,opened_utc,reason,context,venue) "
                  "VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (symbol, side, entry, qty, stop, target,
                   datetime.now(timezone.utc).isoformat(), reason, json.dumps(context), venue))


def record_close(symbol, exit_price, pnl, r_multiple, exit_reason,
                 ext_id: str | None = None) -> dict:
    init_db()
    with _conn() as c:
        pos = c.execute("SELECT * FROM positions WHERE symbol=?", (symbol,)).fetchone()
        if pos is None:
            return {}
        ctx = json.loads(pos["context"] or "{}")
        pm = _postmortem(pos["side"], ctx, exit_reason) if pnl < 0 else None
        c.execute("""INSERT OR IGNORE INTO trades
            (symbol,side,entry,exit,qty,pnl,r_multiple,reason,exit_reason,
             opened_utc,closed_utc,context,postmortem,paper,ext_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1,?)""",
            (symbol, pos["side"], pos["entry"], exit_price, pos["qty"], pnl, r_multiple,
             pos["reason"], exit_reason, pos["opened_utc"],
             datetime.now(timezone.utc).isoformat(), pos["context"], pm, ext_id))
        c.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
        return {"symbol": symbol, "pnl": pnl, "postmortem": pm}


def record_partial(symbol, exit_price, qty_closed, pnl, r_multiple, note: str,
                   ext_id: str | None = None) -> None:
    """Book HALF a trade (a partial take-profit) as its own closed row and shrink the
    still-open position by the quantity that was sold.

    A partial has to be journalled, not just netted at the end: it is realised money,
    the equity curve moves on it, and the day's win/loss count — which is what the
    loss budget spends — is counted from these rows.
    """
    init_db()
    with _conn() as c:
        pos = c.execute("SELECT * FROM positions WHERE symbol=?", (symbol,)).fetchone()
        if pos is None:
            return
        c.execute("""INSERT OR IGNORE INTO trades
            (symbol,side,entry,exit,qty,pnl,r_multiple,reason,exit_reason,
             opened_utc,closed_utc,context,postmortem,paper,ext_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL,1,?)""",
            (symbol, pos["side"], pos["entry"], exit_price, qty_closed, pnl, r_multiple,
             pos["reason"], note, pos["opened_utc"],
             datetime.now(timezone.utc).isoformat(), pos["context"], ext_id))
        left = max(0.0, (pos["qty"] or 0) - qty_closed)
        c.execute("UPDATE positions SET qty=?, partial=1 WHERE symbol=?", (left, symbol))


def import_exchange_trades(rows: list[dict], venue: str = "bybit",
                           skip_symbols: set[str] | None = None) -> int:
    """Re-import closed trades straight from the exchange. Returns how many were new.

    THE PROBLEM IT SOLVES. On a free cloud host the disk is ephemeral: every restart
    or redeploy restores journal.db to the copy in git, so the dashboard showed an
    account that had never traded. The exchange remembers, so on boot the bot reads
    its own history back out of Bybit and refills the journal.

    Two dedupe layers, because both kinds of duplicate are possible:
      1. `ext_id` — same exchange record seen twice (the normal case),
      2. a symbol already booked within 3 minutes of that timestamp — the trade the
         bot closed itself and journalled before this import ever ran, and rows
         written before ext_id existed.
    """
    init_db()
    added = 0
    with _conn() as c:
        have = {r["ext_id"] for r in c.execute(
            "SELECT ext_id FROM trades WHERE ext_id IS NOT NULL")}
        skip = skip_symbols or set()
        for r in rows:
            ext = f"{r['symbol']}:{r['closed_ms']}"
            if ext in have:
                continue
            # A symbol the journal still holds OPEN is mid-reconciliation: the runner
            # is about to book it with its entry reason and feature context. Importing
            # a bare "recovered" row here would win the race and throw that away.
            if r["symbol"] in skip:
                continue
            closed = datetime.fromtimestamp(r["closed_ms"] / 1000, timezone.utc)
            near = c.execute(
                "SELECT id FROM trades WHERE symbol=? AND closed_utc IS NOT NULL "
                "AND ABS(strftime('%s',closed_utc) - ?) < 180",
                (r["symbol"], r["closed_ms"] / 1000)).fetchone()
            if near:
                # Same trade, already ours — stamp it so the next import skips it fast.
                c.execute("UPDATE trades SET ext_id=? WHERE id=? AND ext_id IS NULL",
                          (ext, near["id"]))
                have.add(ext)
                continue
            risk = abs(r["entry"] - r["exit"]) * r["qty"]
            c.execute("""INSERT OR IGNORE INTO trades
                (symbol,side,entry,exit,qty,pnl,r_multiple,reason,exit_reason,
                 opened_utc,closed_utc,context,postmortem,paper,ext_id)
                VALUES (?,?,?,?,?,?,NULL,?,?,?,?,NULL,?,1,?)""",
                (r["symbol"], r["side"], r["entry"], r["exit"], r["qty"], r["pnl"],
                 f"recovered from {venue} history (this bot's own trade; the local "
                 f"journal had been wiped by a restart)",
                 "closed on exchange", closed.isoformat(), closed.isoformat(),
                 # No post-mortem: these rows carry no entry context, so any "lesson"
                 # written from them would be invented. Leaving it NULL keeps the
                 # lessons panel to losses the bot can actually explain.
                 None, ext))
            have.add(ext)
            added += 1
    return added


def open_positions() -> list[dict]:
    init_db()
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM positions").fetchall()]


def recent_trades(limit: int = 50) -> list[dict]:
    init_db()
    with _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]


def stats() -> dict:
    init_db()
    with _conn() as c:
        rows = c.execute("SELECT pnl FROM trades").fetchall()
    pnls = [r["pnl"] for r in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gl = -sum(losses)
    return {
        "trades": len(pnls),
        "win_rate_pct": round(len(wins) / len(pnls) * 100, 1) if pnls else 0.0,
        "net_pnl": round(sum(pnls), 2),
        "profit_factor": round(sum(wins) / gl, 2) if gl > 0 else (float("inf") if wins else 0.0),
    }


# A scratch is not a loss. Once the stop has been pulled to breakeven a trade can
# still close a hair under the entry (fees, a tick of slippage) — booking that as one
# of the day's losses would spend the loss budget on a trade that cost nothing. Only a
# real hit, a quarter of the planned risk or worse, counts against the budget.
LOSS_R = -0.25


def day_summary(day: str | None = None) -> dict:
    """The day's scoreboard, read from the journal — the input to the loss budget.

    `day` is a UTC date string (YYYY-MM-DD); defaults to today. Counts partial
    take-profits as their own (winning) rows, which is exactly how they should read:
    banking half a trade IS a realised win.
    """
    init_db()
    day = day or datetime.now(timezone.utc).date().isoformat()
    with _conn() as c:
        rows = [dict(r) for r in c.execute(
            "SELECT pnl, r_multiple, closed_utc, symbol FROM trades "
            "WHERE substr(closed_utc,1,10)=? ORDER BY closed_utc", (day,))]
    wins = losses = scratches = 0
    last_loss = None
    for r in rows:
        rm, pnl = r.get("r_multiple"), r.get("pnl") or 0.0
        is_loss = (rm is not None and rm <= LOSS_R) or (rm is None and pnl < 0)
        if is_loss:
            losses += 1
            last_loss = r["closed_utc"]
        elif pnl > 0:
            wins += 1
        else:
            scratches += 1
    net = round(sum(r.get("pnl") or 0.0 for r in rows), 4)
    r_sum = round(sum(r["r_multiple"] for r in rows if r.get("r_multiple") is not None), 2)
    return {"date": day, "trades": len(rows), "wins": wins, "losses": losses,
            "scratches": scratches, "net_pnl": net, "r": r_sum, "last_loss_utc": last_loss}


def realised_since(start_ms: int) -> float:
    """Realised PnL booked since an epoch-ms anchor — the book's equity movement.

    Taken from the journal rather than straight from the exchange's closed-PnL sum
    so the equity and the trade history on the dashboard can never disagree. The
    journal is refilled from the exchange on every tick, so it holds the same trades
    plus any the exchange has since aged out of its own window.
    """
    init_db()
    start = datetime.fromtimestamp(start_ms / 1000, timezone.utc).isoformat()
    with _conn() as c:
        v = c.execute("SELECT SUM(pnl) FROM trades WHERE closed_utc >= ?", (start,)).fetchone()[0]
    return round(v or 0.0, 6)


def daily_history(limit: int = 30) -> list[dict]:
    """Per-day W/L/net, newest first — so the dashboard can show whether the loss
    budget is actually holding, instead of one all-time average that hides it."""
    init_db()
    with _conn() as c:
        days = [r[0] for r in c.execute(
            "SELECT DISTINCT substr(closed_utc,1,10) d FROM trades "
            "WHERE closed_utc IS NOT NULL ORDER BY d DESC LIMIT ?", (limit,))]
    return [day_summary(d) for d in days]


def lessons(limit: int = 10) -> list[str]:
    """Most recent loss post-mortems — what the bot has learned not to repeat."""
    init_db()
    with _conn() as c:
        rows = c.execute(
            "SELECT symbol,postmortem,closed_utc FROM trades "
            "WHERE postmortem IS NOT NULL ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [f"{r['closed_utc'][:16]} {r['symbol']}: {r['postmortem']}" for r in rows]


if __name__ == "__main__":
    init_db()
    print("journal.db ready at", DB_PATH)
    print("stats:", stats())
