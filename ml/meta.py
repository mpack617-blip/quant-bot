"""Meta-labeling ML layer (López de Prado style, simplified).

Idea: the rule strategy decides WHEN there is a setup. A gradient-boosted
classifier then decides whether to TAKE it, by predicting P(this setup wins).
Trades whose predicted win-prob is below a threshold are skipped. The model
learns from the realised outcome of every past signal — i.e. it learns from
the bot's own mistakes, which is exactly the "seekhna" the user asked for.

Leakage control: features at a signal bar use only information up to that bar;
the label is the FORWARD outcome. Evaluation is strictly walk-forward (train on
the past, predict the untouched future), never shuffled.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

# scikit-learn ships unsigned native .pyd files; Windows Smart App Control can
# BLOCK them ("can't confirm who published _ball_tree..."), which would crash the
# whole bot on import. Degrade gracefully: if sklearn won't load, the ML filter
# turns neutral (predict_proba -> 0.5) and the bot still trades on signal+news+risk.
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    SKLEARN_OK = True
except Exception as _e:  # noqa: BLE001
    HistGradientBoostingClassifier = None  # type: ignore
    SKLEARN_OK = False
    _SKLEARN_ERR = str(_e)

import config

MODEL_PATH = config.ROOT / "ml" / "meta_model.pkl"

FEATURE_COLS = [
    "ret1", "ret6", "ret24", "dist_ema20_pct", "dist_ema50_pct",
    "rsi14", "atr_pct", "adx14", "macd_hist", "bb_pos", "vol20", "stoch_k",
]

# What a CURRENT model trains on: the candle in front of it plus the situation it
# fires into. Measured over a year of signals with a chronological split and costs,
# the plain FEATURE_COLS set has no edge left; with context it does. See
# research_edge.py / research_gate.py for the numbers.
from features.context import CONTEXT_COLS  # noqa: E402
MODEL_COLS = FEATURE_COLS + CONTEXT_COLS + ["side"]


def expected_value(prob: float, rr: float) -> float:
    """What this setup is worth, in R, if the model's probability is honest.

    This is the number to gate on, not the probability. A 40% shot paying 3:1 is a
    better trade than a 55% shot paying 1:1, and a flat P(win) threshold cannot tell
    them apart — it throws away exactly the trades that pay for the losers.
    """
    return prob * rr - (1.0 - prob)


def _signal_outcome(h, low, c, entry_bar, side, stop, target, max_hold) -> int | None:
    """Simulate one signal forward: 1 = target hit first, 0 = stop hit first.
    Timeout → label by whether the trade was in profit at the horizon."""
    n = len(c)
    end = min(entry_bar + max_hold, n)
    for j in range(entry_bar, end):
        if side == 1:
            if low[j] <= stop:
                return 0
            if h[j] >= target:
                return 1
        else:
            if h[j] >= stop:
                return 0
            if low[j] <= target:
                return 1
    if end <= entry_bar:
        return None
    final = c[end - 1]
    entry = c[entry_bar]
    return int((final - entry) * side > 0)


def build_dataset(df, feats, signals, *, max_hold: int = 48):
    """Return (X, y, idx_list): one row per fired signal with its realised win/loss."""
    h = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    rows, labels, idxs = [], [], []
    for i, sig in enumerate(signals):
        if sig is None or i + 1 >= len(df):
            continue
        y = _signal_outcome(h, low, c, i + 1, sig.side, sig.stop, sig.target, max_hold)
        if y is None:
            continue
        feat = feats.iloc[i]
        if feat[FEATURE_COLS].isna().any():
            continue
        row = feat[FEATURE_COLS].to_dict()
        row["side"] = sig.side
        rows.append(row)
        labels.append(y)
        idxs.append(i)
    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="win")
    return X, y, idxs


def _new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=200, learning_rate=0.05,
        l2_regularization=1.0, min_samples_leaf=20, random_state=42,
    )


@dataclass
class WFResult:
    n: int
    base_winrate: float
    model_winrate: float
    base_pf: float
    model_pf: float
    coverage: float
    threshold: float


def walk_forward_eval(X, y, pnls, *, folds: int = 4, threshold: float = 0.5) -> WFResult:
    """Expanding-window walk-forward: train on past folds, predict the next.
    Compare PF/win-rate of taking ALL signals vs only model-approved ones."""
    n = len(X)
    if n < folds * 6:
        folds = max(2, n // 6)
    bounds = np.linspace(0, n, folds + 1, dtype=int)
    taken_pnl, taken_win, base_pnl, base_win = [], [], [], []
    for k in range(1, folds):
        tr_end = bounds[k]
        te_end = bounds[k + 1]
        if tr_end < 8 or te_end <= tr_end:
            continue
        Xtr, ytr = X.iloc[:tr_end], y.iloc[:tr_end]
        Xte = X.iloc[tr_end:te_end]
        if ytr.nunique() < 2:
            continue
        m = _new_model().fit(Xtr, ytr)
        proba = m.predict_proba(Xte)[:, 1]
        for local, p in enumerate(proba):
            gi = tr_end + local
            base_pnl.append(pnls[gi]); base_win.append(y.iloc[gi])
            if p >= threshold:
                taken_pnl.append(pnls[gi]); taken_win.append(y.iloc[gi])

    def pf(arr):
        a = np.array(arr, float)
        gw = a[a > 0].sum(); gl = -a[a < 0].sum()
        return gw / gl if gl > 0 else (np.inf if gw > 0 else 0.0)

    return WFResult(
        n=len(base_pnl),
        base_winrate=round(np.mean(base_win) * 100, 1) if base_win else 0.0,
        model_winrate=round(np.mean(taken_win) * 100, 1) if taken_win else 0.0,
        base_pf=round(pf(base_pnl), 2),
        model_pf=round(pf(taken_pnl), 2),
        coverage=round(len(taken_pnl) / len(base_pnl) * 100, 1) if base_pnl else 0.0,
        threshold=threshold,
    )


def train_final(X, y) -> HistGradientBoostingClassifier:
    """Train on ALL data and persist — this is the live model.

    The column list is taken from X itself, not from a constant: a model saved with
    a feature list it wasn't trained on scores garbage at runtime, silently.
    """
    m = _new_model().fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": m, "features": list(X.columns)}, f)
    return m


def load_model():
    # unpickling restores a sklearn estimator, so it needs sklearn importable too
    if not SKLEARN_OK or not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:  # noqa: BLE001
        return None


def predict_proba(feat_row: dict, side: int) -> float:
    """Live win-probability for a single setup. 0.5 (neutral) if the model is
    unavailable — e.g. sklearn blocked by Smart App Control — so the bot keeps
    trading, just without the ML filter."""
    bundle = load_model()
    if bundle is None:
        return 0.5
    cols = bundle["features"]
    row = {k: feat_row.get(k, np.nan) for k in cols}
    row["side"] = side
    X = pd.DataFrame([row])
    return float(bundle["model"].predict_proba(X)[:, 1][0])
