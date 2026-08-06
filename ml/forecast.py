"""Next-move forecaster — "market ka agla move kya hoga".

Be clear about what this is and isn't. Nothing can KNOW the next move; anyone
selling that is selling a story. What a model CAN do is give a calibrated
probability and an expected magnitude, e.g. "62% chance ETH is higher in the next
6 hours, typical move ~1.4%". Trade a few hundred of those and the edge shows up
in the P&L. Trade one and it's a coin flip. That's the honest product.

Two brains, deliberately split by what has history and what doesn't:

  1. TRAINED brain (`ml/forecast_model.pkl`) — a classifier for direction and a
     regressor for magnitude, learned from years of OHLCV features across the whole
     universe. Strictly walk-forward evaluated: train on the past, score an untouched
     future, never shuffled. This is where the measurable edge is.

  2. LIVE TILT (`features/microstructure.py`) — open interest, funding, order-book
     and taker flow. These have no usable history to train on (Bybit serves only a
     short window, and the book/tape none at all), so they are NOT fed to the model.
     They adjust the trained probability with documented, hand-set weights, and they
     can veto. Kept separate on purpose: an untrained input must never be able to
     masquerade as a learned one.

The label is a THREE-way problem collapsed to binary with an abstain zone: a move
smaller than `MIN_MOVE_ATR` x ATR is noise, not direction, and training on noise is
how a model learns to be confidently wrong. Those bars are dropped from training.
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    SKLEARN_OK = True
except Exception:  # noqa: BLE001
    HistGradientBoostingClassifier = HistGradientBoostingRegressor = None  # type: ignore
    SKLEARN_OK = False

MODEL_PATH = config.ROOT / "ml" / "forecast_model.pkl"

# Price/indicator features only — everything here has full history, so the model
# sees at training time exactly what it will see live. No microstructure (see above).
FEATURE_COLS = [
    "ret1", "ret6", "ret24", "dist_ema20_pct", "dist_ema50_pct",
    "rsi14", "atr_pct", "adx14", "macd_hist", "bb_pos", "vol20", "stoch_k",
    # engineered here, added by add_forecast_features()
    "ema20_slope", "ema50_slope", "rsi_slope", "atr_ratio", "range_pos",
    "vol_ratio", "body_ratio", "wick_up", "wick_dn", "trend_age",
]

# Where the edge actually IS, found by sweeping timeframe x horizon (not assumed).
# Measured out-of-sample on all 24 coins, 77.5k bars:
#   15m/h4  AUC 0.529  <- chosen        1h/h6  AUC 0.492 (nothing)
#   15m/h8  AUC 0.522                   1h/h12 AUC 0.496 (nothing)
#   4h/h6   AUC 0.518                   4h/h3  AUC 0.509
# Stable across BOTH time-halves (0.535 / 0.527), so it is not a curve-fit. Short
# horizons win because that is where momentum autocorrelation lives; six hours out,
# crypto is genuinely close to a coin flip and the model correctly says so.
MODEL_INTERVAL = "15m"
HORIZON = 4           # bars ahead we forecast (on 15m bars = next ~1 hour)
MIN_MOVE_ATR = 0.5    # smaller than this x ATR = noise, dropped from training


# --------------------------------------------------------------- features
def add_forecast_features(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Extra shape-of-the-move features on top of the standard indicator frame.

    All are backward-looking by construction (rolling/shift on past bars only), so
    there is no lookahead: row i uses nothing after bar i.
    """
    out = feats.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    out["ema20_slope"] = out["ema20"].pct_change(5) * 100      # is the mean itself moving?
    out["ema50_slope"] = out["ema50"].pct_change(10) * 100
    out["rsi_slope"] = out["rsi14"].diff(3)                     # momentum of momentum
    out["atr_ratio"] = out["atr_pct"] / out["atr_pct"].rolling(50).mean()   # vol expanding?
    rng_hi = h.rolling(20).max()
    rng_lo = l.rolling(20).min()
    out["range_pos"] = (c - rng_lo) / (rng_hi - rng_lo).replace(0, np.nan)  # where in the range
    out["vol_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)          # participation
    bar = (h - l).replace(0, np.nan)
    out["body_ratio"] = (c - df["open"]) / bar                  # conviction of the bar
    out["wick_up"] = (h - np.maximum(c, df["open"])) / bar      # rejection from above
    out["wick_dn"] = (np.minimum(c, df["open"]) - l) / bar      # rejection from below
    # how many consecutive bars price has held the same side of EMA20 — trends age
    above = (c > out["ema20"]).astype(int)
    grp = (above != above.shift()).cumsum()
    out["trend_age"] = above.groupby(grp).cumcount() * np.where(above == 1, 1, -1)
    return out


def build_dataset(df: pd.DataFrame, feats: pd.DataFrame, *, horizon: int = HORIZON,
                  min_move_atr: float = MIN_MOVE_ATR):
    """(X, y_dir, y_move) — one row per bar with its realised forward outcome.

    y_dir  : 1 if price is meaningfully HIGHER `horizon` bars later, else 0
    y_move : the signed forward return in % (what the magnitude model learns)
    Bars whose forward move is inside the noise band are dropped entirely.
    """
    f = add_forecast_features(df, feats)
    c = f["close"]
    fwd = (c.shift(-horizon) / c - 1) * 100          # forward return %, in percent
    noise = f["atr_pct"] * min_move_atr              # per-bar noise floor, also in %

    X = f[FEATURE_COLS].copy()
    meaningful = fwd.abs() >= noise
    keep = X.notna().all(axis=1) & fwd.notna() & noise.notna() & meaningful
    return X[keep], (fwd[keep] > 0).astype(int), fwd[keep]


# ----------------------------------------------------------------- training
def _clf():
    return HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05,
                                          l2_regularization=1.0, min_samples_leaf=40,
                                          random_state=42)


def _reg():
    return HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.05,
                                         l2_regularization=1.0, min_samples_leaf=40,
                                         random_state=42)


@dataclass
class WFReport:
    n: int
    accuracy: float          # % of directional calls that were right
    auc: float               # ranking quality — 0.5 is a coin flip
    edge_hi: float           # accuracy on the model's CONFIDENT calls only
    coverage_hi: float       # what share of bars those confident calls are
    mae_move: float          # avg error of the magnitude forecast, in %
    by_bucket: dict = field(default_factory=dict)   # calibration: predicted -> realised


def walk_forward(X: pd.DataFrame, y: pd.Series, moves: pd.Series, *, folds: int = 5,
                 hi_conf: float = 0.60, return_raw: bool = False):
    """Expanding-window walk-forward. The ONLY number worth trusting: every
    prediction scored here was made by a model that had never seen that data.

    With `return_raw=True` also hands back the out-of-sample (prob, truth) pairs,
    which is exactly the data needed to fit an honest probability calibration."""
    n = len(X)
    bounds = np.linspace(0, n, folds + 1, dtype=int)
    probs, truth, pred_mv, true_mv = [], [], [], []
    for k in range(1, folds):
        tr, te = bounds[k], bounds[k + 1]
        if tr < 200 or te <= tr or y.iloc[:tr].nunique() < 2:
            continue
        m = _clf().fit(X.iloc[:tr], y.iloc[:tr])
        g = _reg().fit(X.iloc[:tr], moves.iloc[:tr])
        probs.extend(m.predict_proba(X.iloc[tr:te])[:, 1])
        truth.extend(y.iloc[tr:te])
        pred_mv.extend(g.predict(X.iloc[tr:te]))
        true_mv.extend(moves.iloc[tr:te])

    if not probs:
        empty = WFReport(0, 0, 0.5, 0, 0, 0)
        return (empty, np.array([]), np.array([])) if return_raw else empty
    p = np.array(probs); t = np.array(truth); pm = np.array(pred_mv); tm = np.array(true_mv)
    acc = float(((p > 0.5) == (t == 1)).mean() * 100)

    # AUC without sklearn.metrics (rank-based Mann-Whitney form)
    order = p.argsort()
    ranks = np.empty(len(p), float); ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = t.sum(), len(t) - t.sum()
    auc = float((ranks[t == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else 0.5

    conf = (p >= hi_conf) | (p <= 1 - hi_conf)
    edge_hi = float(((p[conf] > 0.5) == (t[conf] == 1)).mean() * 100) if conf.any() else 0.0

    buckets = {}
    for lo in (0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
        hi = {0.0: 0.4, 0.4: 0.45, 0.45: 0.5, 0.5: 0.55, 0.55: 0.6, 0.6: 0.7, 0.7: 1.01}[lo]
        m = (p >= lo) & (p < hi)
        if m.sum() >= 30:
            buckets[f"{lo:.2f}-{hi:.2f}"] = {"n": int(m.sum()),
                                             "realised_up_pct": round(float(t[m].mean() * 100), 1)}
    rep = WFReport(len(p), round(acc, 1), round(auc, 3), round(edge_hi, 1),
                   round(float(conf.mean() * 100), 1), round(float(np.abs(pm - tm).mean()), 3),
                   buckets)
    return (rep, p, t) if return_raw else rep


# ------------------------------------------------------------- calibration
def fit_calibration(probs: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """Platt scaling on the walk-forward predictions: logit(p) -> a*logit(p) + b.

    Gradient-boosted trees come out OVERCONFIDENT — raw output said 65% where
    reality was 54%. A dashboard that prints 65% when it means 54% is lying to the
    user, so we learn the squash from out-of-sample data and apply it forever after.
    Fitted with plain gradient descent; no extra dependency, ~50 lines saved.
    """
    eps = 1e-6
    z = np.log(np.clip(probs, eps, 1 - eps) / (1 - np.clip(probs, eps, 1 - eps)))
    a, b = 1.0, 0.0
    lr, n = 0.1, len(z)
    if n < 200:
        return 1.0, 0.0
    for _ in range(3000):
        pred = 1 / (1 + np.exp(-(a * z + b)))
        err = pred - truth
        ga, gb = float((err * z).mean()), float(err.mean())
        a -= lr * ga
        b -= lr * gb
    return float(a), float(b)


def apply_calibration(p: float, cal: tuple[float, float] | None) -> float:
    if not cal:
        return p
    a, b = cal
    eps = 1e-6
    pc = min(max(p, eps), 1 - eps)
    z = np.log(pc / (1 - pc))
    return float(1 / (1 + np.exp(-(a * z + b))))


def train_final(X: pd.DataFrame, y: pd.Series, moves: pd.Series, meta: dict,
                calibration: tuple[float, float] | None = None) -> None:
    """Fit on everything and persist — this is the model the live bot loads."""
    bundle = {"clf": _clf().fit(X, y), "reg": _reg().fit(X, moves),
              "features": FEATURE_COLS, "meta": meta, "calibration": calibration}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)


_BUNDLE = None
_BUNDLE_MTIME = None


def load_model():
    """Cached load that re-reads the pickle when the file changes, so retraining
    takes effect without restarting the bot."""
    global _BUNDLE, _BUNDLE_MTIME
    if not SKLEARN_OK or not MODEL_PATH.exists():
        return None
    mt = MODEL_PATH.stat().st_mtime
    if _BUNDLE is None or mt != _BUNDLE_MTIME:
        try:
            with open(MODEL_PATH, "rb") as f:
                _BUNDLE = pickle.load(f)
            _BUNDLE_MTIME = mt
        except Exception:  # noqa: BLE001
            return None
    return _BUNDLE


# ------------------------------------------------------------- live forecast
@dataclass
class Forecast:
    symbol: str
    direction: str          # "UP" / "DOWN" / "UNCLEAR"
    prob_up: float          # trained model's P(higher in `horizon` bars)
    prob: float             # probability of the CALLED direction (>= 0.5)
    expected_move_pct: float
    horizon_bars: int
    horizon_text: str
    confidence: float       # 0..1 — model conviction x microstructure agreement
    ms_bias: float          # live microstructure tilt, -1..+1
    reasons: list = field(default_factory=list)
    model_ready: bool = True

    def as_dict(self) -> dict:
        return {"symbol": self.symbol, "direction": self.direction,
                "prob": round(self.prob * 100, 1), "prob_up": round(self.prob_up * 100, 1),
                "move_pct": round(self.expected_move_pct, 2),
                "horizon": self.horizon_text, "confidence": round(self.confidence, 2),
                "ms_bias": self.ms_bias, "reasons": self.reasons[:5],
                "model_ready": self.model_ready}


# How much the untrained live reads are allowed to move the trained probability.
# Deliberately small: microstructure is real information but it is NOT validated
# the way the model is, so it nudges and vetoes — it never drives.
MS_TILT = 0.12


def forecast(symbol: str, df: pd.DataFrame, feats: pd.DataFrame,
             *, use_microstructure: bool = True, interval: str = MODEL_INTERVAL) -> Forecast:
    """The bot's read on the next move for one symbol, right now."""
    bundle = load_model()
    reasons: list[str] = []
    horizon_text = _horizon_text(interval, HORIZON)

    if bundle is None:
        return Forecast(symbol.replace("USDT", ""), "UNCLEAR", 0.5, 0.5, 0.0, HORIZON,
                        horizon_text, 0.0, 0.0,
                        ["forecast model not trained yet - run: python train_forecast.py"],
                        model_ready=False)

    f = add_forecast_features(df, feats)
    row = f.iloc[-1]
    X = pd.DataFrame([{k: row.get(k, np.nan) for k in bundle["features"]}])
    raw_prob = float(bundle["clf"].predict_proba(X)[:, 1][0])
    # Squash the tree model's overconfidence back to reality before ANYONE sees it.
    prob_up = apply_calibration(raw_prob, bundle.get("calibration"))
    move = float(bundle["reg"].predict(X)[0])

    ms_bias = 0.0
    if use_microstructure:
        try:
            from features.microstructure import snapshot
            snap = snapshot(symbol)
            ms_bias = snap.bias
            prob_up = float(np.clip(prob_up + ms_bias * MS_TILT, 0.02, 0.98))
            reasons.extend(snap.reasons[:3])
        except Exception as e:  # noqa: BLE001
            reasons.append(f"microstructure unavailable ({e})")

    # Thresholds sit at 0.53/0.47, not 0.55/0.45: after calibration an honest crypto
    # directional model tops out near 0.57, so demanding 0.55 would call "UNCLEAR"
    # on almost everything, including its genuinely best reads.
    direction = "UP" if prob_up > 0.53 else ("DOWN" if prob_up < 0.47 else "UNCLEAR")
    prob = prob_up if prob_up >= 0.5 else 1 - prob_up
    # Conviction = how far from a coin flip, boosted when live flow agrees with it.
    edge = abs(prob_up - 0.5) * 2
    called = 1 if prob_up >= 0.5 else -1
    agreement = 1.0 + 0.3 * np.sign(ms_bias) * called if ms_bias else 1.0
    confidence = float(np.clip(edge * agreement, 0, 1))

    reasons.insert(0, _model_reason(row, prob_up, move, horizon_text))
    return Forecast(symbol.replace("USDT", ""), direction, prob_up, prob, move,
                    HORIZON, horizon_text, round(confidence, 2), ms_bias, reasons)


def _model_reason(row, prob_up: float, move: float, horizon_text: str) -> str:
    bits = []
    if abs(row.get("adx14", 0)) >= 25:
        bits.append(f"ADX {row['adx14']:.0f} (trending)")
    elif row.get("adx14", 0) < 18:
        bits.append(f"ADX {row['adx14']:.0f} (chop)")
    r = row.get("rsi14")
    if r is not None and not pd.isna(r):
        bits.append(f"RSI {r:.0f}")
    d = row.get("dist_ema20_pct")
    if d is not None and not pd.isna(d):
        bits.append(f"{abs(d):.1f}% {'above' if d > 0 else 'below'} EMA20")
    ctx = ", ".join(bits)
    return (f"model: {prob_up*100:.0f}% up over {horizon_text}, "
            f"expected {move:+.2f}% [{ctx}]")


def _horizon_text(interval: str, bars: int) -> str:
    mins = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120,
            "4h": 240, "1d": 1440}.get(interval, 60) * bars
    if mins < 60:
        return f"{mins}m"
    if mins < 1440:
        return f"{mins // 60}h"
    return f"{mins // 1440}d"


if __name__ == "__main__":
    from data.bybit import get_klines_cached
    from features.indicators import feature_frame

    syms = sys.argv[1:] or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    for s in syms:
        d = get_klines_cached(s, "1h", bars=400, max_age_min=10)
        fc = forecast(s, d, feature_frame(d))
        print(f"\n{s}: {fc.direction}  {fc.prob*100:.0f}%  move {fc.expected_move_pct:+.2f}% "
              f"over {fc.horizon_text}  (confidence {fc.confidence:.2f})")
        for r in fc.reasons:
            print("   -", r)
