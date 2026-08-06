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

def is_crypto_symbol(sym: str) -> bool:
    return sym.upper().replace("BINANCE:", "").endswith(("USDT", "USD"))

# Universe of liquid USDT crypto perps to scan (extend freely — crypto only)
UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT",
    "ATOMUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "INJUSDT",
    "NEARUSDT", "TIAUSDT", "SEIUSDT", "FILUSDT", "UNIUSDT", "AAVEUSDT",
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
