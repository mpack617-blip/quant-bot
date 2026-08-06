"""Market microstructure — the forward-looking half of the bot's eyes.

Candles tell you what ALREADY happened. These four Bybit feeds tell you what
POSITIONING looks like right now, which is the closest thing to a read on the
next move that exists without a crystal ball:

  1. OPEN INTEREST vs price  — the classic four-quadrant read of who is driving a
     move. Price up on RISING OI = new money going long (trend has fuel). Price up
     on FALLING OI = shorts covering (a rally with no buyer behind it → fades).
     Price down on FALLING OI = longs being liquidated out (capitulation → bounce).
  2. FUNDING RATE — what leveraged traders are PAYING to hold their side. Extreme
     positive = the long side is crowded and paying up → squeeze risk DOWN.
     Contrarian by nature; it says where the crowd is trapped.
  3. ORDER-BOOK IMBALANCE — resting bid vs ask liquidity near the mid. A thick bid
     stack absorbs sells; a thin one lets price fall through.
  4. TAKER FLOW (CVD proxy) — market-order aggression. Who is *hitting* the book
     right now, buyers or sellers. This is the fastest of the four.

Every read is a PROBABILITY TILT, not a prediction. Each returns a score in
[-1, +1] (negative = bearish) plus a plain-English reason, and `snapshot()` folds
them into one directional bias the strategy layer can use as a gate or a feature.

All endpoints are Bybit v5 PUBLIC (no API key). Responses are TTL-cached because
the runner scans 24 symbols a tick and none of this changes second-to-second.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "quant-bot/1.0"})

# endpoint -> (payload, fetched_at). Short TTLs: flow moves fast, funding does not.
_CACHE: dict[tuple, tuple[object, float]] = {}
TTL = {"oi": 120, "funding": 900, "book": 20, "trades": 30, "ticker": 30}


def _get(path: str, params: dict, ttl: int, key: tuple) -> dict:
    hit = _CACHE.get(key)
    if hit and time.time() - hit[1] < ttl:
        return hit[0]  # type: ignore[return-value]
    url = f"{config.DATA_BASE_URL}{path}"
    r = _SESSION.get(url, params=params, timeout=12)
    r.raise_for_status()
    body = r.json()
    if body.get("retCode") != 0:
        raise RuntimeError(f"Bybit {body.get('retCode')}: {body.get('retMsg')}")
    res = body.get("result") or {}
    _CACHE[key] = (res, time.time())
    return res


def _clip(x: float) -> float:
    return float(max(-1.0, min(1.0, x)))


# --------------------------------------------------------------------- 1. OI
@dataclass
class Read:
    score: float                 # -1 bearish .. +1 bullish
    reason: str
    detail: dict = field(default_factory=dict)


def open_interest(symbol: str, interval: str = "15min", lookback: int = 8) -> Read:
    """Positioning flow: is new money entering, or are positions being unwound?

    Interpretation table (price move x OI move):
        +price +OI  -> new longs opening ....... trend has real fuel      (+)
        +price -OI  -> shorts covering ......... rally without buyers     (-)
        -price +OI  -> new shorts opening ...... downtrend has fuel       (-)
        -price -OI  -> longs liquidating out ... capitulation, bounce due (+)
    """
    try:
        res = _get("/v5/market/open-interest",
                   {"category": config.DEFAULT_CATEGORY, "symbol": symbol,
                    "intervalTime": interval, "limit": max(lookback + 1, 12)},
                   TTL["oi"], ("oi", symbol, interval))
        rows = res.get("list") or []
        if len(rows) < 3:
            return Read(0.0, "OI: not enough history")
        # Bybit returns newest-first
        oi = np.array([float(r["openInterest"]) for r in rows][::-1], dtype=float)
        n = min(lookback, len(oi) - 1)
        oi_chg = (oi[-1] / oi[-1 - n] - 1) * 100 if oi[-1 - n] else 0.0

        px = _price_change(symbol, interval, n)
        # Strength = how decisive BOTH moves are; a flat tape means nothing.
        mag = min(1.0, (abs(oi_chg) / 1.5) ** 0.5) * min(1.0, (abs(px) / 1.0) ** 0.5)
        if px >= 0 and oi_chg >= 0:
            s, why = +mag, f"new longs entering (price +{px:.2f}%, OI +{oi_chg:.2f}%) - trend has fuel"
        elif px >= 0 and oi_chg < 0:
            s, why = -mag * 0.8, f"short-covering rally (price +{px:.2f}%, OI {oi_chg:.2f}%) - no real buyers"
        elif px < 0 and oi_chg >= 0:
            s, why = -mag, f"new shorts entering (price {px:.2f}%, OI +{oi_chg:.2f}%) - downtrend has fuel"
        else:
            s, why = +mag * 0.8, f"longs liquidated out (price {px:.2f}%, OI {oi_chg:.2f}%) - capitulation, bounce risk"
        return Read(_clip(s), f"OI: {why}", {"oi_chg_pct": round(oi_chg, 2), "px_chg_pct": round(px, 2)})
    except Exception as e:  # noqa: BLE001
        return Read(0.0, f"OI: unavailable ({e})")


def _price_change(symbol: str, interval: str, bars: int) -> float:
    """% price change over the same window the OI was measured on."""
    iv = {"5min": "5", "15min": "15", "30min": "30", "1h": "60", "4h": "240", "1d": "D"}.get(interval, "15")
    res = _get("/v5/market/kline",
               {"category": config.DEFAULT_CATEGORY, "symbol": symbol,
                "interval": iv, "limit": bars + 2},
               TTL["oi"], ("kl", symbol, iv, bars))
    rows = res.get("list") or []
    if len(rows) < 2:
        return 0.0
    closes = [float(r[4]) for r in rows][::-1]
    return (closes[-1] / closes[0] - 1) * 100 if closes[0] else 0.0


# ----------------------------------------------------------------- 2. FUNDING
def funding(symbol: str) -> Read:
    """Crowd positioning cost. Funding is paid long->short when positive.

    Bybit's baseline is 0.01% per 8h (~11%/yr). Well above that = longs crowded and
    bleeding → the fuel for a long squeeze. Well below/negative = shorts crowded.
    This is a CONTRARIAN read: it scores against whichever side is paying.
    """
    try:
        res = _get("/v5/market/funding/history",
                   {"category": config.DEFAULT_CATEGORY, "symbol": symbol, "limit": 8},
                   TTL["funding"], ("fund", symbol))
        rows = res.get("list") or []
        if not rows:
            return Read(0.0, "funding: no data")
        rates = np.array([float(r["fundingRate"]) for r in rows], dtype=float)  # newest first
        cur = rates[0] * 100                      # -> percent per 8h
        avg = float(rates[:6].mean()) * 100
        # 0.01% = neutral baseline. Score ramps to +-1 around +-0.05% (5x baseline).
        excess = (cur - 0.01) / 0.04
        score = _clip(-excess)                    # crowded longs => bearish tilt
        if cur > 0.03:
            why = f"longs crowded, paying {cur:.4f}%/8h - squeeze-down risk"
        elif cur < -0.01:
            why = f"shorts crowded, paying {abs(cur):.4f}%/8h - squeeze-up fuel"
        else:
            why = f"funding normal ({cur:.4f}%/8h) - no crowd extreme"
        return Read(score, f"funding: {why}",
                    {"funding_pct_8h": round(cur, 5), "avg6_pct": round(avg, 5),
                     "annualised_pct": round(cur * 3 * 365, 1)})
    except Exception as e:  # noqa: BLE001
        return Read(0.0, f"funding: unavailable ({e})")


# --------------------------------------------------------------- 3. ORDERBOOK
def book_imbalance(symbol: str, depth: int = 50, band_pct: float = 0.5) -> Read:
    """Resting liquidity within `band_pct` of mid: who is willing to absorb?

    Thick bids under price = a floor; thin bids = air below. Honest caveat: resting
    orders can be pulled, so this is a weak/fast signal — weighted low in the blend.
    """
    try:
        res = _get("/v5/market/orderbook",
                   {"category": config.DEFAULT_CATEGORY, "symbol": symbol, "limit": 200},
                   TTL["book"], ("book", symbol))
        bids = [(float(p), float(q)) for p, q in (res.get("b") or [])]
        asks = [(float(p), float(q)) for p, q in (res.get("a") or [])]
        if not bids or not asks:
            return Read(0.0, "book: empty")
        mid = (bids[0][0] + asks[0][0]) / 2
        lo, hi = mid * (1 - band_pct / 100), mid * (1 + band_pct / 100)
        bid_usd = sum(p * q for p, q in bids[:depth] if p >= lo)
        ask_usd = sum(p * q for p, q in asks[:depth] if p <= hi)
        tot = bid_usd + ask_usd
        if tot <= 0:
            return Read(0.0, "book: no depth in band")
        imb = (bid_usd - ask_usd) / tot            # -1 .. +1
        side = "bids" if imb > 0 else "asks"
        return Read(_clip(imb * 1.5),
                    f"book: {side} thicker by {abs(imb)*100:.0f}% within {band_pct}% of mid",
                    {"bid_usd": round(bid_usd), "ask_usd": round(ask_usd), "imbalance": round(imb, 3)})
    except Exception as e:  # noqa: BLE001
        return Read(0.0, f"book: unavailable ({e})")


# ------------------------------------------------------------- 4. TAKER FLOW
def taker_flow(symbol: str, limit: int = 1000) -> Read:
    """Aggression: of the volume that just CROSSED the spread, how much was buys?

    A cumulative-volume-delta proxy over the last few hundred prints. This is the
    fastest read here — it shows pressure before it shows up on the candle.
    """
    try:
        res = _get("/v5/market/recent-trade",
                   {"category": config.DEFAULT_CATEGORY, "symbol": symbol, "limit": limit},
                   TTL["trades"], ("trades", symbol))
        rows = res.get("list") or []
        if len(rows) < 20:
            return Read(0.0, "flow: too few prints")
        buy = sum(float(t["size"]) * float(t["price"]) for t in rows if t.get("side") == "Buy")
        sell = sum(float(t["size"]) * float(t["price"]) for t in rows if t.get("side") == "Sell")
        tot = buy + sell
        if tot <= 0:
            return Read(0.0, "flow: no volume")
        delta = (buy - sell) / tot
        who = "buyers" if delta > 0 else "sellers"
        return Read(_clip(delta * 2.0),
                    f"flow: {who} aggressive ({abs(delta)*100:.0f}% net of ${tot:,.0f} taker vol)",
                    {"buy_usd": round(buy), "sell_usd": round(sell), "delta": round(delta, 3)})
    except Exception as e:  # noqa: BLE001
        return Read(0.0, f"flow: unavailable ({e})")


# ------------------------------------------------- 5. MULTI-TIMEFRAME ALIGNMENT
def mtf_alignment(symbol: str, tfs: tuple[str, ...] = ("15", "60", "240")) -> Read:
    """Do the fast, medium and slow charts agree? Trades taken WITH higher-timeframe
    structure survive noise; trades against it get run over. Each timeframe votes on
    close-vs-EMA20 and EMA20-vs-EMA50; the slower the chart, the heavier the vote."""
    try:
        votes, weights, parts = [], [], []
        for tf, w in zip(tfs, (1.0, 1.5, 2.0)):
            res = _get("/v5/market/kline",
                       {"category": config.DEFAULT_CATEGORY, "symbol": symbol,
                        "interval": tf, "limit": 120},
                       TTL["oi"], ("mtf", symbol, tf))
            rows = res.get("list") or []
            if len(rows) < 60:
                continue
            c = np.array([float(r[4]) for r in rows][::-1], dtype=float)
            e20 = _ema(c, 20)[-1]
            e50 = _ema(c, 50)[-1]
            v = (0.5 if c[-1] > e20 else -0.5) + (0.5 if e20 > e50 else -0.5)
            votes.append(v)
            weights.append(w)
            # ASCII only: these strings reach runner.log's print(), and a Windows
            # cp1252 console raises UnicodeEncodeError on arrows — that would kill a tick.
            parts.append(f"{_tf_name(tf)}{'UP' if v > 0 else ('DN' if v < 0 else '--')}")
        if not votes:
            return Read(0.0, "mtf: no data")
        score = float(np.average(votes, weights=weights))
        agree = all(v > 0 for v in votes) or all(v < 0 for v in votes)
        why = " ".join(parts) + (" - all timeframes agree" if agree else " - timeframes split")
        return Read(_clip(score), f"mtf: {why}", {"aligned": agree, "votes": votes})
    except Exception as e:  # noqa: BLE001
        return Read(0.0, f"mtf: unavailable ({e})")


def _ema(a: np.ndarray, span: int) -> np.ndarray:
    alpha = 2 / (span + 1)
    out = np.empty_like(a)
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i - 1]
    return out


def _tf_name(tf: str) -> str:
    return {"15": "15m", "60": "1h", "240": "4h", "D": "1d"}.get(tf, tf)


# ------------------------------------------------------------------ THE BLEND
# Weights reflect how much each read is actually WORTH, not how exciting it looks.
# MTF structure and OI-flow are the durable ones; the order book is easily spoofed
# and funding only matters at extremes — so they get the small votes.
WEIGHTS = {"mtf": 0.30, "oi": 0.28, "flow": 0.20, "funding": 0.14, "book": 0.08}

FEATURE_COLS = ["ms_oi", "ms_funding", "ms_book", "ms_flow", "ms_mtf", "ms_bias"]


@dataclass
class Snapshot:
    symbol: str
    bias: float                 # -1 .. +1 blended directional tilt
    confidence: float           # 0 .. 1 — how much the reads AGREE
    direction: str              # "UP" / "DOWN" / "NEUTRAL"
    reads: dict                 # name -> Read
    reasons: list

    def as_features(self) -> dict:
        return {
            "ms_oi": self.reads["oi"].score,
            "ms_funding": self.reads["funding"].score,
            "ms_book": self.reads["book"].score,
            "ms_flow": self.reads["flow"].score,
            "ms_mtf": self.reads["mtf"].score,
            "ms_bias": self.bias,
        }

    def agrees_with(self, side: int, min_bias: float = 0.15) -> tuple[bool, str]:
        """Veto gate: block a trade only when microstructure leans CLEARLY the other
        way. A neutral read never blocks — absence of evidence isn't evidence."""
        if self.bias * side < -min_bias:
            worst = min(self.reads.values(), key=lambda r: r.score * side)
            return False, f"microstructure against {'LONG' if side == 1 else 'SHORT'}: {worst.reason}"
        return True, f"microstructure ok (bias {self.bias:+.2f})"


def snapshot(symbol: str) -> Snapshot:
    """All five reads for one symbol, blended into one directional bias.

    `confidence` is deliberately agreement-based: five weak reads pointing the same
    way is a far better signal than one strong read fighting the other four.
    """
    reads = {
        "oi": open_interest(symbol),
        "funding": funding(symbol),
        "book": book_imbalance(symbol),
        "flow": taker_flow(symbol),
        "mtf": mtf_alignment(symbol),
    }
    bias = sum(reads[k].score * w for k, w in WEIGHTS.items())
    live = [r.score for r in reads.values() if abs(r.score) > 0.02]
    if live:
        same = sum(1 for s in live if s * bias > 0) / len(live)
        confidence = float(min(1.0, abs(bias) * 1.6) * same)
    else:
        confidence = 0.0
    direction = "UP" if bias > 0.15 else ("DOWN" if bias < -0.15 else "NEUTRAL")
    reasons = [r.reason for r in reads.values() if abs(r.score) > 0.05]
    return Snapshot(symbol.replace("USDT", ""), round(_clip(bias), 3), round(confidence, 2),
                    direction, reads, reasons)


if __name__ == "__main__":
    import json
    for sym in (sys.argv[1:] or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]):
        s = snapshot(sym)
        print(f"\n=== {sym}  {s.direction}  bias {s.bias:+.2f}  confidence {s.confidence:.2f}")
        for line in s.reasons:
            print("   •", line)
        print("   features:", json.dumps(s.as_features()))
