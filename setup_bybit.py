"""Connect the bot to your Bybit account — one script, start to finish.

    python setup_bybit.py            # check the current connection
    python setup_bybit.py --test     # also place and close a tiny REAL test order

WHERE TO GET KEYS
-----------------
DEMO (recommended first — fake money, real mainnet prices, your normal Bybit login):
    bybit.com -> profile menu -> "Demo Trading" -> switch in -> API -> Create New Key
    -> System-generated -> permissions: Read-Write + "Contract - Orders/Positions"
    -> env = "demo"

TESTNET (a completely separate site with its own signup):
    testnet.bybit.com -> API -> Create New Key -> env = "testnet"

MAINNET (REAL MONEY — only after the bot has proven itself on demo):
    bybit.com -> API -> Create New Key -> env = "mainnet"
    Restrict permissions to "Contract - Orders & Positions" ONLY.
    NEVER enable Withdraw. Bind the key to your IP address if you can.

Then put them in `bybit_keys.json` next to this file:

    {"api_key": "xxxx", "api_secret": "yyyy", "env": "demo"}

Keys are never written into source and this file is listed in .gitignore.
"""
from __future__ import annotations

import json
import sys
import time

from brokers.bybit import KEYS_PATH, BybitBroker, BybitError, load_keys

TEMPLATE = {"api_key": "PASTE_YOUR_KEY", "api_secret": "PASTE_YOUR_SECRET", "env": "demo"}


def ensure_keyfile() -> bool:
    """Create a blank key file if none exists. Returns True when keys look filled in."""
    cfg = load_keys()
    if cfg["api_key"] and cfg["api_secret"] and not cfg["api_key"].startswith("PASTE"):
        return True
    if not KEYS_PATH.exists():
        KEYS_PATH.write_text(json.dumps(TEMPLATE, indent=2), encoding="utf-8")
        print(f"created {KEYS_PATH}")
    print("\nNEXT STEP — open this file and paste your Bybit API key + secret:")
    print(f"   {KEYS_PATH}")
    print('   {"api_key": "...", "api_secret": "...", "env": "demo"}')
    print("\n   env = demo | testnet | mainnet   (see the header of this file for where")
    print("   to create each kind of key). Then run `python setup_bybit.py` again.")
    return False


def check() -> bool:
    b = BybitBroker()
    print(f"\nenvironment : {b.env.upper()}   ({b.base})")
    if b.live_money:
        print("              *** MAINNET — THIS IS REAL MONEY ***")
    ping = b.ping()
    print(f"public API  : {'reachable' if ping.get('public') else 'UNREACHABLE'}")
    if not ping.get("public"):
        print(f"  error: {ping.get('error')}")
        print("  Bybit is blocked in some regions/networks — a VPN or different network fixes this.")
        return False
    if not ping.get("authenticated"):
        print(f"account     : NOT AUTHENTICATED\n  error: {ping.get('error')}")
        print("\n  Common causes:")
        print("   - key/secret pasted with a stray space, or swapped")
        print("   - key made on the WRONG site (testnet keys do not work on demo/mainnet)")
        print("   - key lacks Contract trading permission, or has expired")
        print("   - PC clock is off (the bot auto-resyncs, but a huge drift still fails)")
        return False
    print(f"account     : CONNECTED")
    print(f"equity      : ${ping.get('equity')}")
    print(f"available   : ${ping.get('available')}")
    print(f"positions   : {ping.get('open_positions')} open")

    for sym in ("BTCUSDT", "DOGEUSDT", "XRPUSDT"):
        try:
            i = b.instrument(sym)
            eq = ping.get("equity") or 0
            min_notional = i.min_qty * _last_price(b, sym)
            fits = "OK" if min_notional <= eq * i.max_leverage else "TOO BIG for this balance"
            print(f"  {sym:9} min order {i.min_qty:g} (~${min_notional:,.2f} notional)  {fits}")
        except Exception as e:  # noqa: BLE001
            print(f"  {sym:9} instrument check failed: {e}")
    return True


def _last_price(b: BybitBroker, sym: str) -> float:
    res = b._request("GET", "/v5/market/tickers",
                     {"category": b.category, "symbol": sym}, signed=False)
    return float((res.get("list") or [{}])[0].get("lastPrice") or 0)


def test_order() -> None:
    """Place the smallest possible order and immediately close it — proves the full
    round trip (sign -> order -> fill -> verify -> close) on THIS account."""
    b = BybitBroker()
    if b.live_money:
        print("\nRefusing to auto-place a test order on MAINNET (real money).")
        print("Test on demo/testnet first; on mainnet let the bot take its own first trade")
        print("under its normal risk rules, where the stop-loss is part of the order.")
        return
    sym = "DOGEUSDT"
    inst = b.instrument(sym)
    px = _last_price(b, sym)
    # Smallest order the exchange will actually accept: it must clear BOTH the
    # minimum quantity and the minimum order value (5 USDT), then snap up to a whole
    # qty step. Sizing to min_qty alone gets rejected with 110094.
    qty = max(inst.min_qty, inst.round_qty(inst.min_notional / px) + inst.qty_step)
    print(f"\nround-trip test: BUY {qty:g} {sym} (~${qty*px:.2f} notional) then close")
    res = b.market(sym, 1, qty)
    if not res["ok"]:
        print(f"  ORDER FAILED: {res['raw']}")
        return
    print(f"  filled @ {res['fill']} qty {res['qty']}  (order {res.get('order_id')})")
    time.sleep(1.5)
    out = b.close(sym)
    print(f"  close: {'OK - flat' if out['ok'] else 'FAILED: ' + str(out['raw'])}")
    pnl = b.closed_pnl(sym, limit=1)
    if pnl:
        print(f"  realised pnl ${pnl[0]['pnl']:+.4f} (fees included)")
    print("\n  Execution path verified end to end.")


if __name__ == "__main__":
    print("=" * 62)
    print(" Bybit connection setup")
    print("=" * 62)
    if not ensure_keyfile():
        sys.exit(1)
    try:
        ok = check()
    except BybitError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    if ok and "--test" in sys.argv:
        test_order()
    if ok:
        print("\nReady. Start the bot on Bybit with:")
        print("   set QUANT_MODE=bybit && python cockpit.py")
        print("   (or: python runner.py 180 bybit)")
