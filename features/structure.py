"""Market structure — the things a human draws on a chart, computed instead of eyeballed.

A trader draws four things and each answers one question:

  SWING POINTS   where did price actually turn? Everything else is built on these.
  TREND LINE     a sloping line along successive swing highs (resistance) or swing
                 lows (support). It says "the sellers/buyers keep showing up EARLIER
                 each time" — so a break of it is a change in behaviour, not just a
                 new price. A line needs at least 3 touches to mean anything; two
                 points make a line through any two points, which is not evidence.
  RECTANGLE      a range: a flat ceiling and floor price has bounced between. Inside
                 it, edges are where you fade; a close outside it is a breakout.
  S/R LEVEL      a horizontal price that several swings share. The more turns it has
                 caused, the more orders sit there.

This module finds them from the candles. Nothing here predicts anything on its own —
it describes where price is relative to the structure, which is exactly the context
a setup needs: "long at the bottom of a range" and "long into a falling trend line"
are very different trades even when the indicators look identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Line:
    """A sloping trend line: price = slope * bar_index + intercept."""
    kind: str                  # "resistance" | "support"
    slope: float
    intercept: float
    i0: int                    # first and last bar it is drawn between
    i1: int
    touches: int
    def at(self, i: float) -> float:
        return self.slope * i + self.intercept


@dataclass
class Box:
    """A rectangle: a range price has been contained in."""
    top: float
    bottom: float
    i0: int
    i1: int
    touches_top: int
    touches_bottom: int
    @property
    def height_pct(self) -> float:
        return (self.top / self.bottom - 1) * 100 if self.bottom else 0.0


@dataclass
class Structure:
    swing_highs: list[int] = field(default_factory=list)
    swing_lows: list[int] = field(default_factory=list)
    resistance: Line | None = None
    support: Line | None = None
    box: Box | None = None
    levels: list[dict] = field(default_factory=list)   # [{price, touches, kind}]
    notes: list[str] = field(default_factory=list)     # plain English, for the UI


def swing_points(high: np.ndarray, low: np.ndarray, k: int = 3) -> tuple[list[int], list[int]]:
    """Fractal swings: bar i is a swing high if its high is the highest of the k bars
    each side. k is the definition of "how big a turn counts" — small k finds noise,
    large k finds only major pivots."""
    n = len(high)
    highs, lows = [], []
    for i in range(k, n - k):
        w_hi = high[i - k:i + k + 1]
        w_lo = low[i - k:i + k + 1]
        if high[i] == w_hi.max() and (high[i] > high[i - 1] or high[i] > high[i + 1]):
            highs.append(i)
        if low[i] == w_lo.min() and (low[i] < low[i - 1] or low[i] < low[i + 1]):
            lows.append(i)
    return highs, lows


def _fit_line(idx: list[int], px: np.ndarray, kind: str, series: np.ndarray,
              tol_pct: float = 0.35) -> Line | None:
    """Draw a line through the two most recent swings of one kind, then COUNT how
    many other swings sit on it. The count is the whole point: a line with 2 touches
    is a coincidence, 3+ is a level people are trading."""
    if len(idx) < 2:
        return None
    i0, i1 = idx[-2], idx[-1]
    if i1 == i0:
        return None
    slope = (px[i1] - px[i0]) / (i1 - i0)
    intercept = px[i0] - slope * i0
    touches = 0
    for j in idx:
        line_px = slope * j + intercept
        if line_px and abs(px[j] - line_px) / line_px * 100 <= tol_pct:
            touches += 1
    # A line the market has already blown through is not a line any more: if price
    # closed decisively on the wrong side after the last touch, drop it.
    for j in range(i1 + 1, len(series)):
        line_px = slope * j + intercept
        if not line_px:
            continue
        broke = (series[j] > line_px * 1.004) if kind == "resistance" else (series[j] < line_px * 0.996)
        if broke:
            return None
    return Line(kind, slope, intercept, i0, i1, touches)


def _find_box(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              lookback: int = 60, tol_pct: float = 0.6) -> Box | None:
    """A rectangle exists when the recent highs cluster near one price and the lows
    near another, and price has visited both edges more than once."""
    n = len(close)
    if n < lookback + 5:
        return None
    s = n - lookback
    top, bottom = float(high[s:].max()), float(low[s:].min())
    if bottom <= 0:
        return None
    if (top / bottom - 1) * 100 > 25:      # too wide to be a range
        return None
    t_hits = int(np.sum(high[s:] >= top * (1 - tol_pct / 100)))
    b_hits = int(np.sum(low[s:] <= bottom * (1 + tol_pct / 100)))
    if t_hits < 2 or b_hits < 2:
        return None
    return Box(top, bottom, s, n - 1, t_hits, b_hits)


def _levels(idx_h: list[int], idx_l: list[int], high: np.ndarray, low: np.ndarray,
            price: float, tol_pct: float = 0.5, keep: int = 3) -> list[dict]:
    """Horizontal S/R: cluster swing prices that sit within tol of each other. The
    cluster size IS the strength — a price three swings have reversed at is where
    resting orders are.

    Two deliberate details: a level is called support or resistance by where it sits
    relative to price NOW (yesterday's ceiling is today's floor once price is above
    it), and levels within tol of each other are merged, because two lines a quarter
    of a percent apart are one zone drawn twice.
    """
    pts = sorted([float(high[i]) for i in idx_h] + [float(low[i]) for i in idx_l])
    clusters: list[list[float]] = []
    for p in pts:
        if clusters and abs(p - clusters[-1][0]) / clusters[-1][0] * 100 <= tol_pct:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    out = [{"price": float(np.mean(c)), "touches": len(c),
            "kind": "resistance" if np.mean(c) >= price else "support"}
           for c in clusters if len(c) >= 2]
    out.sort(key=lambda d: -d["touches"])
    kept: list[dict] = []
    for lv in out:
        if any(abs(lv["price"] - k["price"]) / k["price"] * 100 <= tol_pct for k in kept):
            continue
        kept.append(lv)
        if len(kept) >= keep:
            break
    return kept


def analyse(df: pd.DataFrame, k: int = 3, lookback: int = 120) -> Structure:
    """Read the structure of the last `lookback` bars and describe it in words."""
    d = df.iloc[-lookback:] if len(df) > lookback else df
    high = d["high"].to_numpy(float)
    low = d["low"].to_numpy(float)
    close = d["close"].to_numpy(float)
    sh, sl = swing_points(high, low, k)
    st = Structure(swing_highs=sh, swing_lows=sl)
    st.resistance = _fit_line(sh, high, "resistance", close)
    st.support = _fit_line(sl, low, "support", close)
    st.box = _find_box(high, low, close, lookback=min(60, len(d) - 5))
    price = float(close[-1])
    st.levels = _levels(sh, sl, high, low, price)
    i = len(d) - 1
    if st.resistance and st.resistance.touches >= 3:
        lp = st.resistance.at(i)
        st.notes.append(f"Falling resistance line ({st.resistance.touches} touches), "
                        f"price {(price/lp-1)*100:+.1f}% from it"
                        if st.resistance.slope < 0 else
                        f"Rising resistance line ({st.resistance.touches} touches), "
                        f"price {(price/lp-1)*100:+.1f}% from it")
    if st.support and st.support.touches >= 3:
        lp = st.support.at(i)
        st.notes.append(f"{'Rising' if st.support.slope > 0 else 'Falling'} support line "
                        f"({st.support.touches} touches), price {(price/lp-1)*100:+.1f}% from it")
    if st.box:
        pos = (price - st.box.bottom) / (st.box.top - st.box.bottom) * 100 \
            if st.box.top > st.box.bottom else 50
        where = "at the ceiling" if pos > 80 else ("at the floor" if pos < 20 else "mid-range")
        st.notes.append(f"Range {st.box.bottom:.6g}-{st.box.top:.6g} "
                        f"({st.box.height_pct:.1f}% tall) — price {where} ({pos:.0f}%)")
    for lv in st.levels[:2]:
        st.notes.append(f"{lv['kind'].title()} at {lv['price']:.6g} "
                        f"({lv['touches']} touches, {(price/lv['price']-1)*100:+.1f}% away)")
    if not st.notes:
        st.notes.append("No clean structure right now — no line with 3+ touches and no range.")
    return st


def to_drawings(st: Structure, d: pd.DataFrame) -> list[dict]:
    """Serialise the structure as drawing instructions the dashboard can render.
    Bar indices are relative to the frame handed in."""
    n = len(d)
    out = []
    for line in (st.resistance, st.support):
        if line is None or line.touches < 3:
            continue
        out.append({
            "type": "trendline", "kind": line.kind, "touches": line.touches,
            "x0": line.i0, "y0": line.at(line.i0),
            "x1": n - 1, "y1": line.at(n - 1),
            "label": f"{line.kind} line · {line.touches} touches",
        })
    if st.box:
        out.append({"type": "rect", "x0": st.box.i0, "x1": n - 1,
                    "y0": st.box.bottom, "y1": st.box.top,
                    "label": f"range {st.box.height_pct:.1f}%"})
    for lv in st.levels:
        out.append({"type": "level", "kind": lv["kind"], "y": lv["price"],
                    "label": f"{lv['kind'][:3]} {lv['touches']}x"})
    return out
