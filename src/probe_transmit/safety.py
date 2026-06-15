"""Loss/safety functions and shared simulator utilities."""
from __future__ import annotations

import numpy as np

# Default environmental temperature safety band used by the experiments.
SAFE_MIN = 18.0
SAFE_MAX = 32.0
RANGE = SAFE_MAX - SAFE_MIN


def loss_terms(xt: np.ndarray, xh: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    err = (xt - xh) / RANGE
    track = err ** 2
    missed = ((xt < SAFE_MIN) | (xt > SAFE_MAX)) & ((xh >= SAFE_MIN) & (xh <= SAFE_MAX))
    safety = 6.0 * missed.astype(float)
    return track + safety, missed.astype(float), track


def jain(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    s = counts.sum()
    if s <= 0:
        return 1.0
    return float((s ** 2) / (len(counts) * float(np.sum(counts ** 2)) + 1e-9))


def normalize_unit(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def margin_to_safety(values: np.ndarray) -> np.ndarray:
    lower_margin = np.maximum(0.0, values - SAFE_MIN)
    upper_margin = np.maximum(0.0, SAFE_MAX - values)
    closest = np.minimum(lower_margin, upper_margin)
    return 1.0 - np.clip(closest / (RANGE / 2), 0.0, 1.0)
