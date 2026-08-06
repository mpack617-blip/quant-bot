"""Central configuration for quant-bot."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# --- Bybit endpoints (v5) ---
BYBIT_MAINNET = "https://api.bybit.com"
BYBIT_TESTNET = "https://api-testnet.bybit.com"

# Public market data is always pulled from mainnet (real prices).
# Order execution will use TESTNET first (set in execution layer later).
DATA_BASE_URL = BYBIT_MAINNET

# category: "linear" = USDT perpetuals (what we trade), "spot" also available
DEFAULT_CATEGORY = "linear"

# Bybit kline interval codes: 1,3,5,15,30,60,120,240,360,720 (minutes), D, W, M
# Map friendly -> bybit code
INTERVALS = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
    "1d": "D", "1w": "W",
}

# Asset class is locked to CRYPTO for now (user request 2026-06-06).
# The runner refuses any non-crypto symbol; only *USDT crypto pairs are traded.
ASSET_CLASS = "crypto"

# Bybit now lists TOKENISED STOCKS AND COMMODITIES as USDT perps — SNDKUSDT is
# SanDisk, SOXLUSDT an ETF, XAUUSDT gold, TSLAUSDT/NVDAUSDT/QQQUSDT equities. Several
# sit near the TOP of the 24h turnover table, so any "most liquid USDT perps" scan
# picks them up. They end in USDT, so the old suffix check happily called them crypto.
#
# They must not be traded here: the user locked this bot to crypto, the strategies and
# ML models were fitted on crypto behaviour, and news.py reads crypto-only RSS — a
# stock would be traded on sentiment that has nothing to do with it. Equities also
# have market hours and gaps that the 24/7 assumptions break on.
#
# Whitelisting is the safe direction: UNIVERSE is hand-curated, and this blocklist is
# the second line of defence for anything added later.
NON_CRYPTO = {
    # US equities / semis
    "SNDK", "MU", "NVDA", "INTC", "MRVL", "TSLA", "GOOGL", "MSTR", "AAOI", "NBIS",
    "SKHYNIX", "SKHY", "SAMSUNG", "CRCL", "RDW", "HEI", "SPCX", "DRAM", "MUU",
    # ETFs / indices
    "SOXL", "SOXS", "QQQ", "EWY", "SNXX", "KORU", "CYS", "HFT",
    # commodities (incl. tokenised metal + oil)
    "XAU", "XAG", "XAUT", "PAXG", "CL", "BZ",
}


def is_crypto_symbol(sym: str) -> bool:
    s = sym.upper().replace("BINANCE:", "")
    if not s.endswith(("USDT", "USD")):
        return False
    base = s.removesuffix("USDT").removesuffix("USD")
    return base not in NON_CRYPTO

# Universe of liquid USDT crypto perps to scan (extend freely — crypto only)
UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "HYPEUSDT", "ZECUSDT",
    "DOGEUSDT", "ADAUSDT", "NEARUSDT", "ENAUSDT", "1000PEPEUSDT", "UNIUSDT",
    "SUIUSDT", "BNBUSDT", "ONDOUSDT", "WLDUSDT", "AAVEUSDT", "LINKUSDT",
    "AVAXUSDT", "XLMUSDT", "TAOUSDT", "1000RATSUSDT", "BICOUSDT", "FARTCOINUSDT",
    "COTIUSDT", "PENGUUSDT", "BCHUSDT", "ARBUSDT", "INJUSDT", "KAITOUSDT",
    "LTCUSDT", "WIFUSDT", "APTUSDT", "XMRUSDT", "LDOUSDT", "DOTUSDT",
    "VIRTUALUSDT", "HBARUSDT", "OPUSDT", "TRUMPUSDT", "TRXUSDT", "FILUSDT",
    "1000BONKUSDT", "TIAUSDT", "ETCUSDT", "CRVUSDT", "AXSUSDT", "DEXEUSDT",
    "VANRYUSDT", "ATOMUSDT", "FIDAUSDT", "RENDERUSDT", "ZROUSDT", "ORDIUSDT",
    "ALGOUSDT", "JUPUSDT", "STRKUSDT", "SEIUSDT", "DASHUSDT",
]

# --- Risk defaults (used by risk manager later) ---
RISK = {
    "max_risk_per_trade_pct": 1.0,   # % of equity risked per trade
    "max_concurrent_positions": 3,
    "max_daily_drawdown_pct": 5.0,   # kill-switch
    "default_leverage": 3,
}

# --- Backtest defaults ---
BACKTEST = {
    "taker_fee": 0.00055,   # Bybit taker ~0.055%
    "maker_fee": 0.0002,    # Bybit maker ~0.02%
    "slippage_bps": 2,      # assumed slippage in basis points
}
