"""TradingView paper-account broker — drives the live TradingView Desktop chart
through the tradingview-mcp Node scripts (CDP port 9222).

This is what "attach the bot to TradingView" means: the quant-bot's brain stays
in Python (Bybit data + strategy + ML + news + risk), but every ORDER is placed
on the real TradingView paper account, and account state is read back from it.

Cardinal rule (learned the hard way on trade T4): after every order, VERIFY the
position via the account read — never assume a click worked.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

TV_ROOT = Path(r"D:\Trader Bot\tradingview-mcp")
SCRIPTS = TV_ROOT / "scripts"
CLI = TV_ROOT / "src" / "cli" / "index.js"
SHOTS = TV_ROOT / "screenshots"

# colors for the bot's drawn levels (TradingView style)
COL_ENTRY = "#58a6ff"   # blue
COL_STOP = "#f85149"    # red
COL_TARGET = "#3fb950"  # green

# On Windows every `node` subprocess flashes a console window; this hides it so the
# bot doesn't pop a window on screen on every tick (set_chart/draw/screenshot/read).
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def to_tv_symbol(sym: str) -> str:
    """'BTCUSDT' -> 'BINANCE:BTCUSDT' (already-prefixed symbols pass through)."""
    return sym if ":" in sym else f"BINANCE:{sym}"


def _round_qty(qty: float) -> str:
    if qty >= 1000:
        return str(int(round(qty)))
    if qty >= 1:
        return f"{qty:.1f}".rstrip("0").rstrip(".")
    return f"{qty:.4f}".rstrip("0").rstrip(".")


class TradingViewBroker:
    def __init__(self, timeout: int = 45):
        self.timeout = timeout

    # ---- low-level runners ----
    def _cli(self, *args) -> str:
        r = subprocess.run(["node", str(CLI), *args], cwd=str(TV_ROOT),
                           capture_output=True, text=True, timeout=self.timeout,
                           creationflags=_NO_WINDOW)
        return r.stdout.strip()

    def _script(self, name: str, *args) -> str:
        r = subprocess.run(["node", str(SCRIPTS / name), *args], cwd=str(SCRIPTS),
                           capture_output=True, text=True, timeout=self.timeout,
                           creationflags=_NO_WINDOW)
        return (r.stdout + r.stderr).strip()

    # ---- account ----
    def account(self) -> dict:
        out = self._script("account_json.mjs")
        line = out.splitlines()[-1] if out else "{}"
        try:
            return json.loads(line)
        except Exception:  # noqa: BLE001
            return {"error": "parse", "raw": out[-300:]}

    def equity(self) -> float | None:
        return self.account().get("equity")

    def positions(self) -> list[dict]:
        """Open positions, or [] if the account could not be read. AMBIGUOUS on
        purpose-of-convenience: use `fetch_positions()` anywhere emptiness would
        trigger a destructive action."""
        try:
            return self.fetch_positions()
        except Exception:  # noqa: BLE001
            return []

    def fetch_positions(self) -> list[dict]:
        """Open positions, RAISING if the account read failed.

        `account()` returns {"error": "parse"} when TradingView isn't running or CDP
        is down — and `.get("positions", [])` then hands back an empty list that is
        indistinguishable from a genuinely flat account. That is how `close()` came to
        report success for six positions it never touched, booking them at $0.00.
        """
        acc = self.account()
        if "error" in acc:
            raise RuntimeError(f"TradingView account unreadable: {acc.get('error')} "
                               f"(is TradingView Desktop running with CDP 9222?)")
        return acc.get("positions", [])

    # ---- orders ----
    def set_symbol(self, sym: str) -> bool:
        out = self._cli("symbol", to_tv_symbol(sym))
        return '"success": true' in out or '"success":true' in out

    def market(self, sym: str, side: int, qty: float) -> dict:
        """Place a market order on TradingView paper, then VERIFY the fill.
        side: +1 buy/long, -1 sell/short. Returns {ok, fill, qty, raw}."""
        tv_side = "buy" if side == 1 else "sell"
        self.set_symbol(sym)
        qstr = _round_qty(qty)
        self._script("set_qty.mjs", qstr)
        self._script("place_order.mjs", tv_side)
        # verify
        bare = sym.split(":")[-1]
        for p in self.positions():
            if bare in (p.get("symbol") or "") and p.get("side") == ("long" if side == 1 else "short"):
                return {"ok": True, "fill": p.get("avgFill"), "qty": p.get("qty"), "raw": p}
        return {"ok": False, "fill": None, "qty": None, "raw": "position not found after order"}

    def close(self, sym: str) -> dict:
        """Close a position by symbol and verify flat for that symbol.

        Verification uses `fetch_positions`, so an unreadable account is reported as a
        FAILURE rather than mistaken for "the position is gone". The caller then leaves
        the trade open instead of booking a fictional close.
        """
        bare = sym.split(":")[-1]
        out = self._script("close_position.mjs", bare)
        try:
            still = any(bare in (p.get("symbol") or "") for p in self.fetch_positions())
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "raw": f"could not verify close: {e}"}
        return {"ok": not still, "raw": out[-200:]}

    # ---- chart control + visualisation (so the bot trades VISIBLY, like a human) ----
    @staticmethod
    def _ok(out: str) -> bool:
        return '"success": true' in out or '"success":true' in out

    @staticmethod
    def _entity_id(out: str) -> str | None:
        m = re.search(r'"entity_id"\s*:\s*"([^"]+)"', out)
        return m.group(1) if m else None

    @staticmethod
    def _fmt(price: float) -> str:
        """Human price label — no scientific notation across crypto magnitudes."""
        if price >= 1000:
            return f"{price:,.0f}"
        if price >= 1:
            return f"{price:,.2f}"
        return f"{price:.6f}".rstrip("0").rstrip(".")

    def set_chart(self, sym: str, tf: str = "15") -> bool:
        """Point the TradingView chart at what the bot is analysing right now.
        The symbol switch renders asynchronously, so settle before any draw/shot
        (else the screenshot can catch the previous symbol)."""
        a = self._cli("symbol", to_tv_symbol(sym))
        b = self._cli("timeframe", tf)
        time.sleep(3)
        return self._ok(a) and self._ok(b)

    def clear_drawings(self) -> None:
        try:
            self._cli("draw", "clear")
        except Exception:  # noqa: BLE001
            pass

    def _hline(self, price: float, color: str, text: str = "", width: int = 2) -> str | None:
        ovr = {"linecolor": color, "linewidth": width, "linestyle": 0,
               "showLabel": True, "textcolor": color, "text": text}
        out = self._cli("draw", "shape", "-t", "horizontal_line", "-p", f"{price}",
                        "--time", str(int(time.time())), "--overrides", json.dumps(ovr))
        return self._entity_id(out)

    def draw_levels(self, sym: str, entry: float, stop: float, target: float,
                    side: int, label: str) -> None:
        """Draw the bot's trade plan on the chart: entry / stop / target lines + a
        text note — exactly the levels a discretionary trader would mark."""
        try:
            self.clear_drawings()
            arrow = "LONG ▲" if side == 1 else "SHORT ▼"
            self._hline(entry, COL_ENTRY, f"{arrow}  ENTRY {self._fmt(entry)} — {label}")
            self._hline(stop, COL_STOP, f"STOP {self._fmt(stop)}")
            self._hline(target, COL_TARGET, f"TARGET {self._fmt(target)}")
        except Exception:  # noqa: BLE001
            pass

    def screenshot(self, name: str = "bot_view") -> str | None:
        """Capture the current chart; returns the saved PNG path (or None)."""
        try:
            out = self._cli("screenshot", "-r", "chart", "-o", name)
            if self._ok(out):
                p = SHOTS / f"{name}.png"
                return str(p) if p.exists() else None
        except Exception:  # noqa: BLE001
            pass
        return None


if __name__ == "__main__":
    b = TradingViewBroker()
    print("account:", b.account())
