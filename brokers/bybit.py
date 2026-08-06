"""Bybit v5 execution broker — the bot's real exchange account.

Until now the brain read Bybit DATA but placed ORDERS on the TradingView paper
account (`brokers/tradingview.py`). This module is the real thing: signed v5 REST
calls that place, verify, protect and close positions on the user's own Bybit
account — demo, testnet or mainnet.

THREE ENVIRONMENTS (config in `bybit_keys.json`, key `env`):
  "demo"    -> https://api-demo.bybit.com     Bybit Demo Trading (fake money, REAL
                                              mainnet prices, same UI as live). DEFAULT.
  "testnet" -> https://api-testnet.bybit.com  separate testnet site + separate keys.
  "mainnet" -> https://api.bybit.com          REAL MONEY. Must be set deliberately.

Cardinal rules kept from the TradingView broker:
  - after EVERY order, VERIFY by re-reading the position (never assume a fill),
  - round qty to the instrument's real qtyStep/minOrderQty (Bybit rejects otherwise),
  - put the stop-loss ON THE EXCHANGE, not only in Python. The runner ticks every
    few minutes; a crypto wick doesn't wait. `market()` attaches SL/TP server-side.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
KEYS_PATH = ROOT / "bybit_keys.json"

HOSTS = {
    "demo": "https://api-demo.bybit.com",
    "testnet": "https://api-testnet.bybit.com",
    "mainnet": "https://api.bybit.com",
}

RECV_WINDOW = "10000"


class BybitError(RuntimeError):
    pass


@dataclass
class Instrument:
    symbol: str
    qty_step: float
    min_qty: float
    max_qty: float
    tick_size: float
    max_leverage: float
    # Bybit enforces a minimum order VALUE (5 USDT on linear perps) on top of the
    # minimum QUANTITY — they are separate checks and passing one says nothing about
    # the other. DOGE's min qty is 1 coin, worth about 7 cents; that order is rejected
    # with 110094 even though the quantity is legal.
    min_notional: float = 5.0

    def round_qty(self, qty: float) -> float:
        """Snap to the exchange's lot size (round DOWN — never size up past risk)."""
        if self.qty_step <= 0:
            return round(qty, 3)
        steps = int(qty / self.qty_step)
        q = steps * self.qty_step
        # float noise: 0.001*3 = 0.003000000000000001 -> clean it to the step's decimals
        dec = max(0, len(f"{self.qty_step:.10f}".rstrip("0").split(".")[-1]))
        return round(q, dec)

    def round_price(self, price: float) -> float:
        if self.tick_size <= 0:
            return price
        dec = max(0, len(f"{self.tick_size:.10f}".rstrip("0").split(".")[-1]))
        return round(round(price / self.tick_size) * self.tick_size, dec)


def load_keys() -> dict:
    """API credentials, in priority order:
       1. env  BYBIT_API_KEY / BYBIT_API_SECRET / BYBIT_ENV
       2. file bybit_keys.json  {"api_key":..,"api_secret":..,"env":"demo"}
    Never hardcode keys in source."""
    key = os.environ.get("BYBIT_API_KEY")
    sec = os.environ.get("BYBIT_API_SECRET")
    env = os.environ.get("BYBIT_ENV")
    if key and sec:
        return {"api_key": key, "api_secret": sec, "env": (env or "demo").lower()}
    if KEYS_PATH.exists():
        try:
            cfg = json.loads(KEYS_PATH.read_text(encoding="utf-8"))
            return {"api_key": cfg.get("api_key", ""), "api_secret": cfg.get("api_secret", ""),
                    "env": (env or cfg.get("env") or "demo").lower()}
        except Exception as e:  # noqa: BLE001
            raise BybitError(f"could not read {KEYS_PATH.name}: {e}")
    return {"api_key": "", "api_secret": "", "env": (env or "demo").lower()}


class BybitBroker:
    """Signed v5 REST client, shaped like TradingViewBroker so the runner can swap
    one for the other: equity() / positions() / market() / close()."""

    def __init__(self, api_key: str | None = None, api_secret: str | None = None,
                 env: str | None = None, category: str = "linear",
                 timeout: int = 15, leverage: int = 5):
        cfg = load_keys()
        self.api_key = api_key or cfg["api_key"]
        self.api_secret = api_secret or cfg["api_secret"]
        self.env = (env or cfg["env"] or "demo").lower()
        if self.env not in HOSTS:
            raise BybitError(f"unknown env '{self.env}' (use demo/testnet/mainnet)")
        self.base = HOSTS[self.env]
        self.category = category
        self.timeout = timeout
        self.leverage = leverage
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "quant-bot/1.0", "Content-Type": "application/json"})
        self._instruments: dict[str, Instrument] = {}
        self._lev_set: set[str] = set()
        self._time_offset_ms = 0     # our clock vs Bybit's (signature dies if we drift)

    # ------------------------------------------------------------------ signing
    @property
    def live_money(self) -> bool:
        return self.env == "mainnet"

    def _ts(self) -> str:
        return str(int(time.time() * 1000) + self._time_offset_ms)

    def _sign(self, ts: str, payload: str) -> str:
        raw = f"{ts}{self.api_key}{RECV_WINDOW}{payload}"
        return hmac.new(self.api_secret.encode(), raw.encode(), hashlib.sha256).hexdigest()

    def _request(self, method: str, path: str, params: dict | None = None,
                 signed: bool = True, _retry: bool = True) -> dict:
        params = params or {}
        url = self.base + path
        if method == "GET":
            qs = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            payload, body, full = qs, None, (url + ("?" + qs if qs else ""))
        else:
            body = json.dumps(params, separators=(",", ":"))
            payload, full = body, url

        headers = {}
        if signed:
            if not self.api_key or not self.api_secret:
                raise BybitError("no API keys — fill bybit_keys.json (see setup_bybit.py)")
            ts = self._ts()
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": ts,
                "X-BAPI-RECV-WINDOW": RECV_WINDOW,
                "X-BAPI-SIGN": self._sign(ts, payload),
            }
        r = self.s.request(method, full, data=body, headers=headers, timeout=self.timeout)
        try:
            out = r.json()
        except Exception:  # noqa: BLE001
            raise BybitError(f"non-JSON reply {r.status_code}: {r.text[:200]}")
        code = out.get("retCode")
        if code != 0:
            # 10002 = timestamp outside recv_window -> resync our clock to the server once
            if code == 10002 and _retry:
                self._sync_time()
                return self._request(method, path, params, signed, _retry=False)
            raise BybitError(f"{code} {out.get('retMsg')} ({path})")
        return out.get("result") or {}

    def _sync_time(self) -> None:
        try:
            r = self.s.get(self.base + "/v5/market/time", timeout=self.timeout).json()
            server_ms = int(r["result"]["timeNano"]) // 1_000_000
            self._time_offset_ms = server_ms - int(time.time() * 1000)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------- instruments
    def instrument(self, symbol: str) -> Instrument:
        """Lot size / tick size / leverage cap — cached. Bybit REJECTS an order whose
        qty isn't a multiple of qtyStep or is below minOrderQty, so every order goes
        through this first."""
        if symbol in self._instruments:
            return self._instruments[symbol]
        res = self._request("GET", "/v5/market/instruments-info",
                            {"category": self.category, "symbol": symbol}, signed=False)
        lst = res.get("list") or []
        if not lst:
            raise BybitError(f"instrument not found: {symbol}")
        it = lst[0]
        lot, pf = it.get("lotSizeFilter", {}), it.get("priceFilter", {})
        inst = Instrument(
            symbol=symbol,
            qty_step=float(lot.get("qtyStep", 0.001)),
            min_qty=float(lot.get("minOrderQty", 0.001)),
            max_qty=float(lot.get("maxOrderQty", 1e12)),
            tick_size=float(pf.get("tickSize", 0.01)),
            max_leverage=float((it.get("leverageFilter") or {}).get("maxLeverage", 25)),
            min_notional=float(lot.get("minNotionalValue") or 5.0),
        )
        self._instruments[symbol] = inst
        return inst

    def last_price(self, symbol: str) -> float | None:
        """Last traded price (public). Used to check an order's value against the
        exchange's minimum before sending it."""
        try:
            res = self._request("GET", "/v5/market/tickers",
                                {"category": self.category, "symbol": symbol}, signed=False)
            return float((res.get("list") or [{}])[0].get("lastPrice") or 0) or None
        except Exception:  # noqa: BLE001
            return None

    # ----------------------------------------------------------------- account
    def account(self) -> dict:
        """{equity, available, positions[]} — the same shape the runner already
        consumes from the TradingView broker."""
        try:
            eq, avail = self._wallet()
            return {"equity": eq, "available": avail, "positions": self.positions(),
                    "env": self.env}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e), "equity": None, "positions": [], "env": self.env}

    SETTLE_COIN = "USDT"

    def _wallet(self) -> tuple[float | None, float | None]:
        """Equity in the SETTLEMENT coin — deliberately NOT the account's total equity.

        A Unified account values everything it holds into `totalEquity`. This demo
        account happens to hold 1 BTC and 1 ETH, so `totalEquity` swings by hundreds of
        dollars whenever BTC moves — noise that has nothing to do with the bot. Sizing
        off that number made a $200 book jump to $37 on a single BTC dip, and would
        have tripped the daily kill-switch on market movement rather than on losses.

        USDT-settled perpetuals credit and debit USDT, so the USDT balance moves if and
        only if the bot's own trades move it. That is the number to trade against.
        """
        res = self._request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"})
        lst = res.get("list") or []
        if not lst:
            return None, None
        acc = lst[0]
        for c in acc.get("coin") or []:
            if c.get("coin") == self.SETTLE_COIN:
                eq = c.get("equity")
                avail = c.get("availableToWithdraw") or c.get("walletBalance")
                if eq not in (None, ""):
                    return (float(eq),
                            float(avail) if avail not in (None, "") else None)
        # No USDT line (fresh account) — fall back to the account total.
        eq = acc.get("totalEquity") or acc.get("totalWalletBalance")
        avail = acc.get("totalAvailableBalance")
        return (float(eq) if eq not in (None, "") else None,
                float(avail) if avail not in (None, "") else None)

    def equity(self) -> float | None:
        try:
            return self._wallet()[0]
        except Exception:  # noqa: BLE001
            return None

    def positions(self) -> list[dict]:
        """Open positions, or [] if the read failed. Convenient, but AMBIGUOUS: an
        empty list can mean 'flat' or 'the call blew up'. Anything that would act
        destructively on emptiness must use `fetch_positions()` instead."""
        try:
            return self.fetch_positions()
        except Exception:  # noqa: BLE001
            return []

    def fetch_positions(self) -> list[dict]:
        """Open positions, normalised to the runner's vocabulary
        (side 'long'/'short', avgFill, qty, unrealised). RAISES on failure, so the
        caller can tell 'no positions' apart from 'I could not check'."""
        res = self._request("GET", "/v5/position/list",
                            {"category": self.category, "settleCoin": "USDT", "limit": 200})
        out = []
        for p in res.get("list") or []:
            size = float(p.get("size") or 0)
            if size <= 0:
                continue
            out.append({
                "symbol": p.get("symbol"),
                "side": "long" if p.get("side") == "Buy" else "short",
                "qty": size,
                "avgFill": float(p.get("avgPrice") or 0),
                "mark": float(p.get("markPrice") or 0),
                "unrealised": float(p.get("unrealisedPnl") or 0),
                "stop": float(p.get("stopLoss") or 0) or None,
                "target": float(p.get("takeProfit") or 0) or None,
                "leverage": p.get("leverage"),
                "positionIdx": p.get("positionIdx", 0),
            })
        return out

    def set_leverage(self, symbol: str, lev: int | None = None) -> bool:
        """Set leverage once per symbol per process. Bybit returns 110043
        ('leverage not modified') when it's already right — that's success."""
        lev = int(lev or self.leverage)
        if symbol in self._lev_set:
            return True
        inst = self.instrument(symbol)
        lev = max(1, min(lev, int(inst.max_leverage)))
        try:
            self._request("POST", "/v5/position/set-leverage",
                          {"category": self.category, "symbol": symbol,
                           "buyLeverage": str(lev), "sellLeverage": str(lev)})
        except BybitError as e:
            if "110043" not in str(e) and "not modified" not in str(e).lower():
                return False
        self._lev_set.add(symbol)
        return True

    # ------------------------------------------------------------------ orders
    def market(self, sym: str, side: int, qty: float,
               stop: float | None = None, target: float | None = None) -> dict:
        """Market order + EXCHANGE-SIDE stop-loss/take-profit, then VERIFY the fill.

        side: +1 long / -1 short. Returns {ok, fill, qty, order_id, protected, raw}.

        The SL/TP go to Bybit itself so the position is protected even if this bot,
        the PC, or the internet dies between ticks — the single biggest difference
        between paper trading and real money.
        """
        symbol = sym.split(":")[-1].upper()
        try:
            inst = self.instrument(symbol)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "fill": None, "qty": None, "raw": f"instrument: {e}"}

        self.set_leverage(symbol)
        q = inst.round_qty(qty)
        if q < inst.min_qty:
            return {"ok": False, "fill": None, "qty": None,
                    "raw": (f"qty {qty:.8f} -> {q} below exchange minimum {inst.min_qty} "
                            f"for {symbol}. Raise risk-per-trade or pick a cheaper coin.")}
        # Second, INDEPENDENT exchange floor: order VALUE. Checked here rather than
        # letting Bybit reject it (110094), so the reason reaches the activity feed in
        # plain language. We refuse instead of bumping the quantity up to reach $5 —
        # a bigger quantity means a bigger loss at the stop, and silently trading more
        # size than the risk plan called for is exactly the bug you never notice.
        px = self.last_price(symbol)
        if px and q * px < inst.min_notional:
            need = inst.min_notional / px
            return {"ok": False, "fill": None, "qty": None,
                    "raw": (f"order value ${q * px:.2f} is below {symbol}'s ${inst.min_notional:.0f} "
                            f"minimum (needs qty >= {need:.4g}). Widening the stop or raising "
                            f"risk-per-trade would fix it, but both mean risking more.")}

        order = {
            "category": self.category,
            "symbol": symbol,
            "side": "Buy" if side == 1 else "Sell",
            "orderType": "Market",
            "qty": str(q),
            "timeInForce": "IOC",
            "positionIdx": 0,          # one-way mode
            "reduceOnly": False,
        }
        # Attach protection with the entry when we have it (atomic — no naked window).
        if stop:
            order["stopLoss"] = str(inst.round_price(stop))
            order["slTriggerBy"] = "LastPrice"
        if target:
            order["takeProfit"] = str(inst.round_price(target))
            order["tpTriggerBy"] = "LastPrice"

        try:
            res = self._request("POST", "/v5/order/create", order)
        except BybitError as e:
            return {"ok": False, "fill": None, "qty": None, "raw": str(e)}

        order_id = res.get("orderId")
        # ---- VERIFY: a returned orderId is not a filled position ----
        fill = qty_filled = None
        pos = None
        for _ in range(6):
            time.sleep(0.5)
            pos = next((p for p in self.positions()
                        if p["symbol"] == symbol
                        and p["side"] == ("long" if side == 1 else "short")), None)
            if pos:
                fill, qty_filled = pos["avgFill"], pos["qty"]
                break
        if not pos:
            return {"ok": False, "fill": None, "qty": None, "order_id": order_id,
                    "raw": "order accepted but no position appeared — check Bybit manually"}

        # If SL/TP didn't ride along with the entry, set them now on the position.
        protected = bool(pos.get("stop"))
        if (stop or target) and not protected:
            protected = self.set_stops(symbol, stop, target)

        return {"ok": True, "fill": fill, "qty": qty_filled, "order_id": order_id,
                "protected": protected, "raw": pos}

    def set_stops(self, sym: str, stop: float | None = None,
                  target: float | None = None) -> bool:
        """Move the exchange-side SL/TP on an open position (used by the trailing stop)."""
        symbol = sym.split(":")[-1].upper()
        try:
            inst = self.instrument(symbol)
            body = {"category": self.category, "symbol": symbol,
                    "positionIdx": 0, "tpslMode": "Full"}
            if stop:
                body["stopLoss"] = str(inst.round_price(stop))
                body["slTriggerBy"] = "LastPrice"
            if target:
                body["takeProfit"] = str(inst.round_price(target))
                body["tpTriggerBy"] = "LastPrice"
            self._request("POST", "/v5/position/trading-stop", body)
            return True
        except BybitError as e:
            # 34040 / "not modified" = the level is already exactly there.
            return "not modified" in str(e).lower() or "34040" in str(e)

    def close(self, sym: str) -> dict:
        """Flatten a symbol with a reduce-only market order, then verify flat."""
        symbol = sym.split(":")[-1].upper()
        pos = next((p for p in self.positions() if p["symbol"] == symbol), None)
        if not pos:
            return {"ok": True, "raw": "already flat"}
        try:
            self._request("POST", "/v5/order/create", {
                "category": self.category,
                "symbol": symbol,
                "side": "Sell" if pos["side"] == "long" else "Buy",
                "orderType": "Market",
                "qty": str(pos["qty"]),
                "timeInForce": "IOC",
                "positionIdx": 0,
                "reduceOnly": True,
            })
        except BybitError as e:
            return {"ok": False, "raw": str(e)}
        for _ in range(6):
            time.sleep(0.5)
            if not any(p["symbol"] == symbol for p in self.positions()):
                return {"ok": True, "raw": "closed"}
        return {"ok": False, "raw": "close order sent but position still open"}

    def closed_pnl(self, symbol: str | None = None, limit: int = 20) -> list[dict]:
        """Realised PnL of recent closed trades — the honest record of what the
        exchange actually paid us (fees included), not our own estimate."""
        try:
            res = self._request("GET", "/v5/position/closed-pnl",
                                {"category": self.category, "symbol": symbol, "limit": limit})
        except Exception:  # noqa: BLE001
            return []
        return [{"symbol": r.get("symbol"), "side": r.get("side"),
                 "qty": float(r.get("qty") or 0),
                 "entry": float(r.get("avgEntryPrice") or 0),
                 "exit": float(r.get("avgExitPrice") or 0),
                 "pnl": float(r.get("closedPnl") or 0),
                 "closed_ms": int(r.get("updatedTime") or 0)}
                for r in (res.get("list") or [])]

    # ------------------------------------------------------------------ health
    def ping(self) -> dict:
        """One-call health check used by setup_bybit.py and the cockpit."""
        out = {"env": self.env, "host": self.base, "keys": bool(self.api_key and self.api_secret)}
        try:
            self._request("GET", "/v5/market/time", signed=False)
            out["public"] = True
        except Exception as e:  # noqa: BLE001
            out["public"] = False
            out["error"] = str(e)
            return out
        if not out["keys"]:
            out["authenticated"] = False
            out["error"] = "no API keys configured"
            return out
        try:
            eq, avail = self._wallet()
            out.update({"authenticated": True, "equity": eq, "available": avail,
                        "open_positions": len(self.positions())})
        except Exception as e:  # noqa: BLE001
            out["authenticated"] = False
            out["error"] = str(e)
        return out


if __name__ == "__main__":
    b = BybitBroker()
    print(json.dumps(b.ping(), indent=2))
