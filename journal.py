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


def record_close(symbol, exit_price, pnl, r_multiple, exit_reason) -> dict:
    init_db()
    with _conn() as c:
        pos = c.execute("SELECT * FROM positions WHERE symbol=?", (symbol,)).fetchone()
        if pos is None:
            return {}
        ctx = json.loads(pos["context"] or "{}")
        pm = _postmortem(pos["side"], ctx, exit_reason) if pnl < 0 else None
        c.execute("""INSERT INTO trades
            (symbol,side,entry,exit,qty,pnl,r_multiple,reason,exit_reason,
             opened_utc,closed_utc,context,postmortem,paper)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
            (symbol, pos["side"], pos["entry"], exit_price, pos["qty"], pnl, r_multiple,
             pos["reason"], exit_reason, pos["opened_utc"],
             datetime.now(timezone.utc).isoformat(), pos["context"], pm))
        c.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
        return {"symbol": symbol, "pnl": pnl, "postmortem": pm}


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
