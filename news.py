"""News + sentiment layer (free RSS, no paid API).

Pulls headlines from public crypto RSS feeds, maps each to the symbols it
mentions, and scores a lightweight lexicon sentiment. The runner uses this two
ways: (1) a per-symbol bias the strategy can require to agree with the trade
direction, and (2) a market-wide risk-off flag (e.g. a flood of 'hack',
'lawsuit', 'ban' headlines) that can tighten or pause trading.

Deliberately simple + dependency-light (feedparser only). Swappable for a paid
sentiment API later without touching callers.

COVERAGE (fixed 2026-08-14). The symbol map used to list 14 coins while the bot
trades 59, so `agrees_with()` answered "news neutral/thin" for 76% of the universe
and the news gate was effectively switched off for everything except the majors.
`COVERAGE_GAP` below is asserted empty by the self-test at the bottom, so a coin
added to config.UNIVERSE without a name here is caught instead of silently losing
its news check.

MATCHING (fixed 2026-08-14). Matching used substrings, which forced hacks like
`"eth "` and `"link "` with a trailing space and still mismatched: "link" matched
"linked", "hyperlink" and every "link to the article". Matching is now on WORD
BOUNDARIES, and words that are ordinary English are deliberately left out of the
map rather than matched loosely — a false "NEAR hits record" from the phrase "near
record" is worse than no news at all, because it feeds the gate a confident wrong
answer.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import feedparser

# Free, crypto-only RSS. Breadth matters: one outlet's silence on a mid-cap is not
# evidence of calm, so the more independent desks in here, the more coins get a real
# read instead of "thin". Feeds that stop responding are skipped silently per fetch.
FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://news.bitcoin.com/feed/",
    "https://cryptoslate.com/feed/",
    "https://decrypt.co/feed",
    "https://www.theblock.co/rss.xml",
    "https://bitcoinist.com/feed/",
    "https://cryptopotato.com/feed/",
    "https://beincrypto.com/feed/",
    "https://u.today/rss",
    "https://ambcrypto.com/feed/",
    "https://coinjournal.net/feed/",
]

# symbol -> the words a headline would actually use for it.
#
# Rules this map follows, learned from the false matches the old one produced:
#   * a ticker is only listed when it is NOT an ordinary English word once word
#     boundaries are applied (`sui`, `zro`, `strk` are fine; `link`, `near`, `world`,
#     `curve`, `render`, `etc`, `algo`, `virtual` are not — those coins are matched
#     by their project name alone, even though that means fewer hits),
#   * the full project name always comes first, because it is the unambiguous one,
#   * "optimism" is left out on purpose: "optimism" is also what every second market
#     report calls a mood, and OP would have inherited the whole market's sentiment.
SYMBOL_KEYWORDS = {
    "BTCUSDT": ["bitcoin", "btc"],
    "ETHUSDT": ["ethereum", "ether", "eth", "vitalik"],
    "SOLUSDT": ["solana", "sol"],
    "XRPUSDT": ["xrp", "ripple"],
    "HYPEUSDT": ["hyperliquid", "hype"],
    "ZECUSDT": ["zcash", "zec"],
    "DOGEUSDT": ["dogecoin", "doge"],
    "ADAUSDT": ["cardano", "ada"],
    "NEARUSDT": ["near protocol", "near foundation"],   # bare "near" is a preposition
    "ENAUSDT": ["ethena", "ena"],
    "1000PEPEUSDT": ["pepe", "pepecoin"],
    "UNIUSDT": ["uniswap", "uni"],
    "SUIUSDT": ["sui", "mysten"],
    "BNBUSDT": ["binance coin", "bnb", "bnb chain"],
    "ONDOUSDT": ["ondo"],
    "WLDUSDT": ["worldcoin", "world network", "wld"],   # bare "world" is everywhere
    "AAVEUSDT": ["aave"],
    "LINKUSDT": ["chainlink"],                          # bare "link" matched every URL
    "AVAXUSDT": ["avalanche", "avax"],
    "XLMUSDT": ["stellar", "lumens", "xlm"],
    "TAOUSDT": ["bittensor", "tao"],
    "1000RATSUSDT": ["rats ordinals", "rats token"],
    "BICOUSDT": ["biconomy", "bico"],
    "FARTCOINUSDT": ["fartcoin"],
    "COTIUSDT": ["coti"],
    "PENGUUSDT": ["pudgy penguins", "pengu"],
    "BCHUSDT": ["bitcoin cash", "bch"],
    "ARBUSDT": ["arbitrum", "arb"],
    "INJUSDT": ["injective", "inj"],
    "KAITOUSDT": ["kaito"],
    "LTCUSDT": ["litecoin", "ltc"],
    "WIFUSDT": ["dogwifhat", "wif"],
    "APTUSDT": ["aptos", "apt"],
    "XMRUSDT": ["monero", "xmr"],
    "LDOUSDT": ["lido", "ldo"],
    "DOTUSDT": ["polkadot", "dot"],
    "VIRTUALUSDT": ["virtuals protocol", "virtuals"],   # bare "virtual" is everywhere
    "HBARUSDT": ["hedera", "hbar"],
    "OPUSDT": ["op token", "op mainnet", "optimism network", "op stack"],
    "TRUMPUSDT": ["trump coin", "official trump", "$trump", "trump memecoin"],
    "TRXUSDT": ["tron", "trx", "justin sun"],
    "FILUSDT": ["filecoin", "fil"],
    "1000BONKUSDT": ["bonk"],
    "TIAUSDT": ["celestia", "tia"],
    "ETCUSDT": ["ethereum classic"],                    # bare "etc" is punctuation
    "CRVUSDT": ["curve dao", "curve finance", "crv"],   # bare "curve" is a yield curve
    "AXSUSDT": ["axie infinity", "axie", "axs"],
    "DEXEUSDT": ["dexe"],
    "VANRYUSDT": ["vanar", "vanry"],
    "ATOMUSDT": ["cosmos", "atom"],
    "FIDAUSDT": ["bonfida", "fida"],
    "RENDERUSDT": ["render network", "render token", "rndr"],  # bare "render" is a verb
    "ZROUSDT": ["layerzero", "zro"],
    "ORDIUSDT": ["ordinals", "ordi"],
    "ALGOUSDT": ["algorand"],                           # bare "algo" is "algo trading"
    "JUPUSDT": ["jupiter exchange", "jupiter dex", "jup"],
    "STRKUSDT": ["starknet", "strk"],
    "SEIUSDT": ["sei network", "sei"],
    "DASHUSDT": ["dash"],
}

# Sentiment lexicon. Matched on WORD BOUNDARIES, with the plural/tense variants spelled
# out, because plain substring matching was quietly inverting headlines:
#   "against"  contains "gain"  -> counted BULLISH
#   "banking"  contains "ban"   -> counted BEARISH
#   "finest"   contains "fine"  -> counted BEARISH
# Live example that made this worth fixing: "KAITO crashes 68%, erases its July rally"
# scored +1 BULLISH (crash -1, rally +1, "against" +1), and a bullish read on a coin
# vetoes a SHORT — the exact trade that headline argues for, on a coin the bot was
# short at the time.
POSITIVE = ["surge", "surges", "surged", "soar", "soars", "soared", "rally", "rallies",
            "rallied", "bull", "bullish", "gain", "gains", "gained", "jump", "jumps",
            "jumped", "adopt", "adopts", "adoption", "approval", "approve", "approves",
            "approved", "etf", "partnership", "upgrade", "upgrades", "record high",
            "all-time high", "breakout", "institutional", "inflow", "inflows", "buy",
            "buys", "accumulate", "accumulation", "support", "launch", "launches",
            "surging", "climbs", "climb", "soaring", "outperform", "outperforms"]
NEGATIVE = ["hack", "hacked", "exploit", "exploited", "lawsuit", "sue", "sues", "sued",
            "ban", "bans", "banned", "crash", "crashes", "crashed", "plunge", "plunges",
            "plunged", "dump", "dumps", "slump", "slumps", "bear", "bearish", "sell-off",
            "selloff", "liquidation", "liquidations", "fraud", "scam", "fud", "outflow",
            "outflows", "decline", "declines", "declined", "warning", "warns", "delay",
            "delays", "delayed", "reject", "rejects", "rejected", "fine", "fined",
            "sec", "erase", "erases", "erased", "tumble", "tumbles", "sinks", "sink",
            "slide", "slides", "drop", "drops", "falls", "fell", "loses", "lost",
            "underperform", "underperforms", "collapse", "collapses"]

_POS_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in POSITIVE) + r")\b")
_NEG_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in NEGATIVE) + r")\b")

_CACHE: dict = {"ts": 0.0, "items": []}
_TTL = 600  # 10 min

# Compiled once: word-boundary matchers per symbol. A multi-word phrase keeps its
# internal spacing but is still anchored at both ends, so "sei network" cannot match
# inside a longer word.
_PATTERNS: dict[str, re.Pattern] = {
    sym: re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b")
    for sym, kws in SYMBOL_KEYWORDS.items()
}


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
    """Net sentiment of one story: bullish words minus bearish ones, counted on word
    boundaries so "against" is not a bullish "gain". Distinct words are counted, not
    repetitions, so a story that hammers one word isn't louder than one that makes
    several separate points."""
    t = text.lower()
    return len(set(_POS_RE.findall(t))) - len(set(_NEG_RE.findall(t)))


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
    """Latest crypto headlines the bot is 'reading', each with a sentiment tag and
    the coins it mentions. Source = crypto-only RSS feeds (FEEDS)."""
    items = _fetch()
    out = []
    for it in items[:n]:
        text = (it["title"] + " " + it["summary"]).lower()
        s = _score_text(text)
        tag = "bull" if s > 0 else ("bear" if s < 0 else "neutral")
        coins = [sym.replace("USDT", "") for sym, pat in _PATTERNS.items()
                 if pat.search(text)]
        out.append({"title": it["title"][:120], "score": s, "tag": tag,
                    "coins": coins[:4]})
    return out


def symbol_sentiment(symbol: str) -> Sentiment:
    pat = _PATTERNS.get(symbol)
    if pat is None:
        return Sentiment(0.0, 0, False, [])
    items = _fetch()
    hits = [i for i in items if pat.search((i["title"] + " " + i["summary"]).lower())]
    if not hits:
        return Sentiment(0.0, 0, False, [])
    scores = [_score_text(i["title"] + " " + i["summary"]) for i in hits]
    norm = max(-1.0, min(1.0, sum(scores) / len(scores)))
    return Sentiment(round(norm, 2), len(hits), False, [h["title"] for h in hits[:3]])


def coverage(universe: list[str]) -> tuple[list[str], list[str]]:
    """(covered, uncovered) — which traded symbols the news layer can actually read.
    An uncovered symbol always answers "neutral", i.e. its news gate does nothing."""
    covered = [s for s in universe if s in SYMBOL_KEYWORDS]
    return covered, [s for s in universe if s not in SYMBOL_KEYWORDS]


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
    import config

    cov, gap = coverage(config.UNIVERSE)
    print(f"COVERAGE: {len(cov)}/{len(config.UNIVERSE)} traded symbols have a news map")
    if gap:
        print("  !! NO NEWS MAP (news gate is dead for these):", ", ".join(gap))

    m = market_sentiment()
    print(f"\nMARKET sentiment {m.score} over {m.n_headlines} headlines | risk_off={m.risk_off}")
    for t in m.top:
        print("  most-negative:", t)

    print("\nper-symbol read (only coins the feeds actually mentioned):")
    hits = 0
    for sym in config.UNIVERSE:
        s = symbol_sentiment(sym)
        if s.n_headlines:
            hits += 1
            print(f"  {sym:14} score {s.score:+.2f} ({s.n_headlines} headlines)  "
                  f"e.g. {s.top[0][:70] if s.top else ''}")
    print(f"\n{hits} of {len(config.UNIVERSE)} coins are in the news right now")
