"""Gilbert-Elliott burst channel models used in the ProbeTransmit experiments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChannelParams:
    name: str
    p_gb: float  # good -> bad transition probability
    p_bg: float  # bad -> good transition probability
    p_ok_good: float
    p_ok_bad: float


CHANNELS: dict[str, ChannelParams] = {
    "burst": ChannelParams(name="burst", p_gb=0.08, p_bg=0.30, p_ok_good=0.985, p_ok_bad=0.55),
    "severe_burst": ChannelParams(name="severe_burst", p_gb=0.12, p_bg=0.20, p_ok_good=0.97, p_ok_bad=0.30),
}


def step_state(params: ChannelParams, rng: np.random.Generator, bad: bool) -> tuple[bool, bool]:
    if bad:
        if rng.random() < params.p_bg:
            bad = False
    else:
        if rng.random() < params.p_gb:
            bad = True
    p_ok = params.p_ok_bad if bad else params.p_ok_good
    return bool(rng.random() < p_ok), bad


def stationary_bad_belief(params: ChannelParams) -> float:
    return params.p_gb / (params.p_gb + params.p_bg)


def update_bad_belief(prior_bad: float, ok: bool, params: ChannelParams) -> float:
    """Bayesian update of P(bad) after observing one delivery outcome."""
    p_ok_given_bad = params.p_ok_bad
    p_ok_given_good = params.p_ok_good
    if ok:
        like_bad = p_ok_given_bad
        like_good = p_ok_given_good
    else:
        like_bad = 1.0 - p_ok_given_bad
        like_good = 1.0 - p_ok_given_good
    posterior = like_bad * prior_bad / (like_bad * prior_bad + like_good * (1.0 - prior_bad) + 1e-12)
    return float(np.clip(posterior, 1e-3, 1.0 - 1e-3))


def predict_success(prior_bad: float, params: ChannelParams) -> float:
    return float(prior_bad * params.p_ok_bad + (1.0 - prior_bad) * params.p_ok_good)


# ---------------- per-arm (vectorized) Gilbert-Elliott channel ----------------
# Each sensor i owns an INDEPENDENT Gilbert-Elliott link: its own good/bad mode
# bad[i] and its own belief pi_bad[i]. This makes p_succ_i sensor-specific, so
# the scheduler can do genuine opportunistic per-link scheduling (defer a sensor
# whose link is deep-fading, favour one whose link is good) -- the substance
# behind the "Channel-Aware" name. It also restores per-arm independence, the
# structural premise of the RMAB asymptotic-optimality theory.


def step_state_vec(
    params: ChannelParams, rng: np.random.Generator, bad: np.ndarray, served: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Advance N independent G-E links one slot and return (ok, bad') vectors.

    - bad: (N,) bool current mode per sensor (the latent good/bad state).
    - served: (N,) bool, True for sensors that actually transmit this slot.

    EVERY link's latent mode evolves each slot (restless: modes drift whether or
    not we transmit). ``ok`` is only meaningful for served sensors; unserved
    entries are returned False and ignored by the caller.
    """
    bad = np.asarray(bad, dtype=bool).copy()
    n = bad.shape[0]
    # Mode transitions for all links (restless).
    flip_bg = (rng.random(n) < params.p_bg) & bad        # bad -> good
    flip_gb = (rng.random(n) < params.p_gb) & (~bad)      # good -> bad
    bad = bad & ~flip_bg
    bad = bad | flip_gb
    # Delivery outcome per served link given its (new) mode.
    p_ok = np.where(bad, params.p_ok_bad, params.p_ok_good)
    ok = (rng.random(n) < p_ok) & np.asarray(served, dtype=bool)
    return ok, bad


def stationary_bad_belief_vec(params: ChannelParams, n: int) -> np.ndarray:
    """(N,) vector initialised to the stationary P(bad) for every sensor."""
    return np.full(n, stationary_bad_belief(params), dtype=float)


def update_bad_belief_vec(
    prior_bad: np.ndarray, ok: np.ndarray, served: np.ndarray, params: ChannelParams
) -> np.ndarray:
    """Per-arm belief filter over the latent good/bad mode.

    Two-step POMDP filter, vectorized over sensors:
      1. PREDICT (all sensors, every slot): push belief through the G-E mode
         transition kernel, since the latent mode is restless.
      2. UPDATE (served sensors only): Bayesian correction from the observed
         delivery outcome ok[i]. Unserved sensors keep the predicted belief.
    """
    prior_bad = np.asarray(prior_bad, dtype=float)
    served = np.asarray(served, dtype=bool)
    # 1. Predict: P(bad') = P(bad)*(1-p_bg) + P(good)*p_gb
    pred = prior_bad * (1.0 - params.p_bg) + (1.0 - prior_bad) * params.p_gb
    # 2. Bayesian update from delivery outcome, served sensors only.
    like_bad = np.where(ok, params.p_ok_bad, 1.0 - params.p_ok_bad)
    like_good = np.where(ok, params.p_ok_good, 1.0 - params.p_ok_good)
    post = (like_bad * pred) / (like_bad * pred + like_good * (1.0 - pred) + 1e-12)
    out = np.where(served, post, pred)
    return np.clip(out, 1e-3, 1.0 - 1e-3)


def predict_success_vec(prior_bad: np.ndarray, params: ChannelParams) -> np.ndarray:
    """(N,) per-arm expected delivery success probability."""
    prior_bad = np.asarray(prior_bad, dtype=float)
    return prior_bad * params.p_ok_bad + (1.0 - prior_bad) * params.p_ok_good
