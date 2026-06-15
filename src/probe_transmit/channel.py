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
