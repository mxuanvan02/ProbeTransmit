"""Recent (2023-2026) SoTA index-rule baselines for the SOTA comparison table.

These are OUR faithful implementations of the published *index rules* (not
re-runs of the authors' code), evaluated apples-to-apples with CAW-VoU: same
per-arm Gilbert-Elliott channel, same budgets, same windows, same metrics. We do
NOT claim to reproduce the authors' exact numbers or experimental setup.

Implemented (all pure index policies, no training, runnable in-harness):

1. OnlineWhittle  -- Online Whittle Index Policy for fair+efficient sensor
   scheduling (Fair and Efficient Scheduling for Sensor Networks via Online
   Whittle Index Policy, arXiv:2605.08674, 2026). Per-arm index increases with
   estimation error (proxy: forecast variance / track gap) and with a fairness
   term driven by the long-run activation deficit; channel-weighted by p_succ.

2. MultiChannelWhittle -- Whittle index for remote estimation of Gauss-Markov
   processes over channels (Remote Estimation of Gauss-Markov Processes over
   Multiple Channels: A Whittle Index Policy, arXiv:2305.04809, 2023). Index =
   expected one-step error-variance reduction if served, scaled by per-arm
   channel success probability -- the strongest "channel-aware" competitor since
   it directly exploits the new per-arm channel belief.

3. RiskAwareAoII -- Risk-Aware AoII-based scheduling (Risk-Aware AoII-Based
   Scheduling with Hybrid Transmission for a Semi-Markov Source, arXiv:2606.11905,
   2026). Index = AoII-style staleness weighted by a risk term = probability the
   belief is in a danger band (threshold-crossing risk), the closest published
   semantic competitor to VoU. Uses belief only (no genie), so it is a fair,
   information-matched competitor to CAW-VoU.

Each builds a two-stage (probe + payload) policy mirroring its own score so the
comparison is self-consistent (as AoII/MaxWeight/VoI baselines already are).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from probe_transmit import safety  # noqa: E402
from probe_transmit.channel import predict_success_vec  # noqa: E402
from probe_transmit.policies import (  # noqa: E402
    Policy,
    SchedulerState,
    TwoStagePolicy,
    service_debt,
    topk,
)

_H = 4


def _p_succ(state: SchedulerState) -> np.ndarray:
    return np.clip(predict_success_vec(state.pi_bad, state.channel), 0.25, 1.0)


def _danger(state: SchedulerState, mu: np.ndarray) -> np.ndarray:
    """Near-threshold relevance in [0,1] (1 = at a safety bound)."""
    return safety.margin_to_safety(mu)


def _apply_safety(idx: np.ndarray, danger: np.ndarray, gamma: float) -> np.ndarray:
    """Fair safety-aware adaptation: reweight a base index by threshold relevance.

    gamma = 0  -> as-published (pure estimation index, no safety awareness)
    gamma > 0  -> multiply by danger**gamma so budget concentrates on
                  safety-relevant sensors, mirroring CAW-VoU's threshold focus.
    A small floor keeps every sensor reachable (avoids hard starvation).
    """
    if gamma <= 0:
        return idx
    return idx * (0.05 + 0.95 * np.clip(danger, 0.0, 1.0)) ** gamma


def _probe_debt(state: SchedulerState) -> np.ndarray:
    return service_debt(state.probe_counts, state.total_probe_choices,
                        state.probe_target_share())


def _payload_debt(state: SchedulerState) -> np.ndarray:
    return service_debt(state.payload_counts, state.total_payload_choices,
                        state.payload_target_share())


# ---------------------------------------------------------------------------
# 1. Online Whittle Index (arXiv:2605.08674, 2026)
# ---------------------------------------------------------------------------
class OnlineWhittleProbe(Policy):
    """Index ~ (forecast uncertainty + fairness deficit), channel-weighted."""

    name = "online_whittle_probe"

    def __init__(self, w_fair: float = 0.5, h: int = _H, gamma: float = 0.0):
        self.w_fair = float(w_fair)
        self.h = int(h)
        self.gamma = float(gamma)

    def _index(self, state: SchedulerState, debt: np.ndarray) -> np.ndarray:
        mu, var = state.ar.forecast_stats(state.xh, state.age, h=self.h)
        # estimation-error proxy: predictive std (Gauss-Markov error growth)
        err = np.sqrt(np.maximum(var, 1e-12))
        # online Whittle index: marginal value of activation = error reduction
        # gained by serving, plus a fairness term from the activation deficit.
        idx = err + self.w_fair * debt
        idx = _apply_safety(idx, _danger(state, mu), self.gamma)
        return _p_succ(state) * idx

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        b = min(state.b_probe, state.n)
        if b <= 0:
            return np.empty(0, dtype=int)
        return topk(self._index(state, _probe_debt(state)), b)


class OnlineWhittlePayload:
    name = "online_whittle_payload"

    def __init__(self, w_fair: float = 0.5, h: int = _H, gamma: float = 0.0):
        self.w_fair = float(w_fair)
        self.h = int(h)
        self.gamma = float(gamma)

    def select(self, state, probe_set, metadata, mask) -> np.ndarray:
        b = min(state.b_payload, state.n)
        if b <= 0:
            return np.empty(0, dtype=int)
        mu, var = state.ar.forecast_stats(state.xh, state.age, h=self.h)
        err = np.sqrt(np.maximum(var, 1e-12))
        idx = err + self.w_fair * _payload_debt(state)
        # use probed info: if a probe revealed danger, bump it.
        danger = safety.margin_to_safety(np.where(mask, metadata, state.xh))
        idx = _apply_safety(idx, danger, self.gamma)
        idx = _p_succ(state) * idx + mask.astype(float) * danger * 0.5
        return topk(idx, b)


# ---------------------------------------------------------------------------
# 2. Multi-Channel Whittle for Gauss-Markov remote estimation (arXiv:2305.04809)
# ---------------------------------------------------------------------------
class MultiChannelWhittleProbe(Policy):
    """Index = expected error-variance reduction * channel success prob."""

    name = "mc_whittle_probe"

    def __init__(self, h: int = _H, gamma: float = 0.0):
        self.h = int(h)
        self.gamma = float(gamma)

    def _index(self, state: SchedulerState) -> np.ndarray:
        mu, var = state.ar.forecast_stats(state.xh, state.age, h=self.h)
        # serving resets error to ~1-step noise; reduction ~ (var - sigma^2)_+
        sig2 = np.asarray(state.ar.sigma, dtype=float) ** 2
        reduction = np.maximum(var - sig2, 0.0)
        reduction = _apply_safety(reduction, _danger(state, mu), self.gamma)
        return _p_succ(state) * reduction

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        b = min(state.b_probe, state.n)
        if b <= 0:
            return np.empty(0, dtype=int)
        return topk(self._index(state), b)


class MultiChannelWhittlePayload:
    name = "mc_whittle_payload"

    def __init__(self, h: int = _H, gamma: float = 0.0):
        self.h = int(h)
        self.gamma = float(gamma)

    def select(self, state, probe_set, metadata, mask) -> np.ndarray:
        b = min(state.b_payload, state.n)
        if b <= 0:
            return np.empty(0, dtype=int)
        mu, var = state.ar.forecast_stats(state.xh, state.age, h=self.h)
        sig2 = np.asarray(state.ar.sigma, dtype=float) ** 2
        reduction = np.maximum(var - sig2, 0.0)
        danger = safety.margin_to_safety(np.where(mask, metadata, state.xh))
        reduction = _apply_safety(reduction, danger, self.gamma)
        idx = _p_succ(state) * reduction + mask.astype(float) * danger * 0.5
        return topk(idx, b)


# ---------------------------------------------------------------------------
# 3. Risk-Aware AoII (arXiv:2606.11905, 2026) -- belief-only, no genie
# ---------------------------------------------------------------------------
_BELIEF_AGE = "risk_aoii_age"


def _belief_age(state: SchedulerState) -> np.ndarray:
    age = state.extras.get(_BELIEF_AGE)
    if age is None or len(age) != state.n:
        age = np.asarray(state.age, dtype=float).copy()
        state.extras[_BELIEF_AGE] = age
    return age


class RiskAwareAoIIProbe(Policy):
    """Index = staleness (age) * risk(belief in danger band)."""

    name = "risk_aoii_probe"

    def __init__(self, w_debt: float = 0.05, h: int = _H):
        self.w_debt = float(w_debt)
        self.h = int(h)

    def _index(self, state: SchedulerState, debt: np.ndarray) -> np.ndarray:
        mu, var = state.ar.forecast_stats(state.xh, state.age, h=self.h)
        risk = safety.margin_to_safety(mu)  # in [0,1], higher = closer to bound
        stale = np.asarray(state.age, dtype=float)
        idx = stale * risk + self.w_debt * debt
        return _p_succ(state) * idx

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        b = min(state.b_probe, state.n)
        if b <= 0:
            return np.empty(0, dtype=int)
        return topk(self._index(state, _probe_debt(state)), b)


class RiskAwareAoIIPayload:
    name = "risk_aoii_payload"

    def __init__(self, w_debt: float = 0.05, h: int = _H):
        self.w_debt = float(w_debt)
        self.h = int(h)

    def select(self, state, probe_set, metadata, mask) -> np.ndarray:
        b = min(state.b_payload, state.n)
        if b <= 0:
            return np.empty(0, dtype=int)
        mu, var = state.ar.forecast_stats(state.xh, state.age, h=self.h)
        risk = safety.margin_to_safety(np.where(mask, metadata, mu))
        stale = np.asarray(state.age, dtype=float)
        idx = stale * risk + self.w_debt * _payload_debt(state)
        return topk(_p_succ(state) * idx, b)


# ---------------------------------------------------------------------------
# 4. QAoI-Whittle: Age of Information AT QUERY (arXiv:2411.02108, 2024)
# ---------------------------------------------------------------------------
# Pull-based model: freshness matters only WHEN a query arrives. In a safety
# monitor the implicit "query" is the need to know a sensor's state precisely
# when it is near a safety boundary, so we model the per-slot query probability
# as the threshold relevance q_i = danger(mu_i) in [0,1]. The QAoI Whittle index
# prioritises the expected age penalty paid at the next query:
#   index_i = age_i * q_i  (channel-weighted), i.e. stale AND likely-queried.
class QAoIWhittleProbe(Policy):
    name = "qaoi_whittle_probe"

    def __init__(self, h: int = _H):
        self.h = int(h)

    def _index(self, state: SchedulerState) -> np.ndarray:
        mu, _ = state.ar.forecast_stats(state.xh, state.age, h=self.h)
        q = _danger(state, mu)  # query probability proxy in [0,1]
        age = np.asarray(state.age, dtype=float)
        return _p_succ(state) * age * q

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        b = min(state.b_probe, state.n)
        if b <= 0:
            return np.empty(0, dtype=int)
        return topk(self._index(state), b)


class QAoIWhittlePayload:
    name = "qaoi_whittle_payload"

    def __init__(self, h: int = _H):
        self.h = int(h)

    def select(self, state, probe_set, metadata, mask) -> np.ndarray:
        b = min(state.b_payload, state.n)
        if b <= 0:
            return np.empty(0, dtype=int)
        mu, _ = state.ar.forecast_stats(state.xh, state.age, h=self.h)
        q = safety.margin_to_safety(np.where(mask, metadata, mu))
        age = np.asarray(state.age, dtype=float)
        return topk(_p_succ(state) * age * q, b)


# ---------------------------------------------------------------------------
# 5. QVAoI: Query Version Age of Information (arXiv:2407.08587, 2024)
# ---------------------------------------------------------------------------
# Version AoI counts how many source "versions" (meaningful changes) the
# receiver is behind, evaluated at query. We proxy the version count by the
# number of forecast standard deviations the belief has drifted since the last
# update (how many "just-noticeable" changes were missed), weighted by the
# query relevance q_i = danger(mu_i):
#   index_i = version_lag_i * q_i,  version_lag_i = |mu - xh| / sigma_i.
class QVAoIProbe(Policy):
    name = "qvaoi_probe"

    def __init__(self, h: int = _H):
        self.h = int(h)

    def _index(self, state: SchedulerState) -> np.ndarray:
        mu, _ = state.ar.forecast_stats(state.xh, state.age, h=self.h)
        sig = np.asarray(state.ar.sigma, dtype=float) + 1e-9
        version_lag = np.abs(mu - state.xh) / sig
        q = _danger(state, mu)
        return _p_succ(state) * version_lag * q

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        b = min(state.b_probe, state.n)
        if b <= 0:
            return np.empty(0, dtype=int)
        return topk(self._index(state), b)


class QVAoIPayload:
    name = "qvaoi_payload"

    def __init__(self, h: int = _H):
        self.h = int(h)

    def select(self, state, probe_set, metadata, mask) -> np.ndarray:
        b = min(state.b_payload, state.n)
        if b <= 0:
            return np.empty(0, dtype=int)
        mu, _ = state.ar.forecast_stats(state.xh, state.age, h=self.h)
        sig = np.asarray(state.ar.sigma, dtype=float) + 1e-9
        ref = np.where(mask, metadata, mu)
        version_lag = np.abs(ref - state.xh) / sig
        q = safety.margin_to_safety(ref)
        return topk(_p_succ(state) * version_lag * q, b)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def make_sota_policy(which: str, gamma: float = 0.0) -> TwoStagePolicy:
    """Build a recent-SoTA two-stage baseline.

    gamma is the safety-aware adaptation strength (0 = as-published estimation
    index; >0 concentrates budget on near-threshold sensors, the fair safety
    adaptation reported alongside the as-published variant).
    """
    if which == "online_whittle":
        return TwoStagePolicy("OnlineWhittle-2026",
                              OnlineWhittleProbe(gamma=gamma),
                              OnlineWhittlePayload(gamma=gamma))
    if which == "mc_whittle":
        return TwoStagePolicy("MultiChannelWhittle-2023",
                              MultiChannelWhittleProbe(gamma=gamma),
                              MultiChannelWhittlePayload(gamma=gamma))
    if which == "risk_aoii":
        return TwoStagePolicy("RiskAwareAoII-2026",
                              RiskAwareAoIIProbe(), RiskAwareAoIIPayload())
    if which == "qaoi_whittle":
        return TwoStagePolicy("QAoI-Whittle-2024",
                              QAoIWhittleProbe(), QAoIWhittlePayload())
    if which == "qvaoi":
        return TwoStagePolicy("QVAoI-2024",
                              QVAoIProbe(), QVAoIPayload())
    raise ValueError(f"unknown SoTA baseline: {which}")
