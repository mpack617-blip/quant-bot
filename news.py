"""News + sentiment layer (free RSS, no paid API).

Pulls headlines from public crypto RSS feeds, maps each to the symbols it
mentions, and scores a lightweight lexicon sentiment. The runner uses this two
ways: (1) a per-symbol bias the strategy can require to agree with the trade
direction, and (2) a market-wide risk-off flag (e.g. a flood of 'hack',
'lawsuit', 'ban' headlines) that can tighten or pause trading.

Deliberately simple + dependency-light (feedparser only). Swappable for a paid
sentiment API later without touching callers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import feedparser

FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://news.bitcoin.com/feed/",
    "https://cryptoslate.com/feed/",
]

# symbol -> keywords that map a headline to it
SYMBOL_KEYWORDS = {
    "BTCUSDT": ["bitcoin", "btc"],
    "ETHUSDT": ["ethereum", "ether ", "eth ", "vitalik"],
    "SOLUSDT": ["solana", "sol "],
    "BNBUSDT": ["binance", "bnb"],
    "XRPUSDT": ["xrp", "ripple"],
    "DOGEUSDT": ["dogecoin", "doge"],
    "ADAUSDT": ["cardano", "ada "],
    "AVAXUSDT": ["avalanche", "avax"],
    "LINKUSDT": ["chainlink", "link "],
    "DOTUSDT": ["polkadot", "dot "],
    "SUIUSDT": ["sui "],
    "INJUSDT": ["injective", "inj "],
    "NEARUSDT": ["near protocol", "near "],
    "AAVEUSDT": ["aave"],
}

POSITIVE = ["surge", "soar", "rally", "bull", "gain", "jump", "adopt", "approval",
            "approve", "etf", "partnership", "upgrade", "record high", "breakout",
            "institutional", "buy", "accumulate", "support", "launch"]
NEGATIVE = ["hack", "exploit", "lawsuit", "sue", "ban", "crash", "plunge", "dump",
            "bear", "sell-off", "selloff", "liquidation", "fraud", "scam", "fud",
            "outflow", "decline", "warning", "delay", "reject", "fine", "sec "]

_CACHE: dict = {"ts": 0.0, "items": []}
_TTL = 600  # 10 min


@dataclass
class Sentiment:
    score: float          # -1 .. +1
    n_headlines: int
    risk_off: bool
    top: list = field(default_factory=list)


def _fetch(max_age: float = _TTL) -> list[dict]:
    if time.time() - _CACHE["ts"] < max_age and _CACHE["items"]:
        return _CACHE["items"]
    items = []
    for url in FEEDS:
        try:
            d = feedparser.parse(url)
            for e in d.entries[:40]:
                items.append({"title": (e.get("title") or "").strip(),
                              "summary": (e.get("summary") or "")[:300]})
        except Exception:  # noqa: BLE001
            continue
    if items:
        _CACHE.update(ts=time.time(), items=items)
    return items


def _score_text(text: str) -> int:
    t = text.lower()
    return sum(w in t for w in POSITIVE) - sum(w in t for w in NEGATIVE)


def market_sentiment() -> Sentiment:
    items = _fetch()
    if not items:
        return Sentiment(0.0, 0, False, [])
    scores = [_score_text(i["title"] + " " + i["summary"]) for i in items]
    neg = sum(1 for s in scores if s < 0)
    raw = sum(scores)
    norm = max(-1.0, min(1.0, raw / (len(scores) or 1)))
    risk_off = neg >= max(5, len(items) * 0.35)
    top = sorted(zip(scores, [i["title"] for i in items]), key=lambda x: x[0])[:3]
    return Sentiment(round(norm, 2), len(items), risk_off,
                     [f"({s:+d}) {t}" for s, t in top])


def recent_headlines(n: int = 8) -> list[dict]:
    """Latest crypto headlines the bot is 'reading', each with a sentiment tag.
    Source = crypto-only RSS feeds (FEEDS) — the bot sees nothing but trading news."""
    items = _fetch()
    out = []
    for it in items[:n]:
        s = _score_text(it["title"] + " " + it["summary"])
        tag = "bull" if s > 0 else ("bear" if s < 0 else "neutral")
        out.append({"title": it["title"][:120], "score": s, "tag": tag})
    return out


def symbol_sentiment(symbol: str) -> Sentiment:
    kws = SYMBOL_KEYWORDS.get(symbol, [])
    if not kws:
        return Sentiment(0.0, 0, False, [])
    items = _fetch()
    hits = [i for i in items if any(k in (i["title"] + " " + i["summary"]).lower() for k in kws)]
    if not hits:
        return Sentiment(0.0, 0, False, [])
    scores = [_score_text(i["title"] + " " + i["summary"]) for i in hits]
    norm = max(-1.0, min(1.0, sum(scores) / len(scores)))
    return Sentiment(round(norm, 2), len(hits), False, [h["title"] for h in hits[:3]])


def agrees_with(symbol: str, side: int, *, min_conf: int = 2,
                strong: float = 0.5) -> tuple[bool, str]:
    """True if news doesn't CLEARLY contradict the trade. Neutral/thin news is
    allowed — only a confident multi-headline bias against us vetoes the trade.

    Softened 2026-06-07 (user wants more activity): a single bearish headline used
    to veto every long (e.g. one 'XRP selloff' story blocked all XRP longs). Now we
    require >=2 headlines AND a clear average bias (|score|>=strong) to block."""
    s = symbol_sentiment(symbol)
    if s.n_headlines < min_conf or s.score == 0:
        return True, "news neutral/thin"
    if side == 1 and s.score <= -strong:
        return False, f"news clearly bearish ({s.score}, {s.n_headlines}h) vs LONG"
    if side == -1 and s.score >= strong:
        return False, f"news clearly bullish ({s.score}, {s.n_headlines}h) vs SHORT"
    return True, f"news ok ({s.score})"


if __name__ == "__main__":
    m = market_sentiment()
    print(f"MARKET sentiment {m.score} over {m.n_headlines} headlines | risk_off={m.risk_off}")
    for t in m.top:
        print("  most-negative:", t)
    for sym in ["BTCUSDT", "ETHUSDT", "XRPUSDT"]:
        s = symbol_sentiment(sym)
        print(f"{sym}: score {s.score} ({s.n_headlines} headlines)")
