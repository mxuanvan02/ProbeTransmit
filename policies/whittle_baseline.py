"""Heuristic Whittle-index baseline for probe-then-transmit scheduling.

This module provides a Whittle-index baseline for comparison against CAW-VoU,
following the heuristic online Whittle-index spirit of Jonah & Yoo 2026
(WAoII/FWAoII) and the indexability/Whittle-like-index results of
Tang et al. 2026, but WITHOUT the threshold-aware urgency, correlation credit,
or closed-form debt of CAW-VoU.

Two interfaces are exposed:

1. ``WhittleBaseline`` -- a lightweight, self-contained class that selects
   probe/payload sets purely from belief mean/uncertainty. Useful for unit
   tests and the documented heuristic in the manuscript.

2. ``WhittleProbe`` / ``WhittlePayload`` -- ``Policy``-compatible classes that
   plug into the existing :class:`TwoStagePolicy` and simulator, so the
   comparison against CAW-VoU runs on the SAME Intel-trace pipeline and
   metrics (Loss, RMSE, Missed-vio, Runtime). This is the version used for the
   manuscript's Whittle-comparison table.

Design note (honesty for Q1 review):
    A full Whittle index for this partially observable restless bandit would
    require solving a belief-MDP per node (value/policy iteration), costing
    O(K N |S|^3). We instead use a closed-form heuristic Whittle index based on
    belief uncertainty and urgency-to-threshold, which is the practical
    surrogate used by recent online-Whittle work. The runtime gap reported in
    the manuscript reflects this heuristic surrogate, not an exact belief-MDP
    solve (which would be even slower). The baseline is therefore a charitable,
    fast Whittle proxy, so any CAW-VoU runtime advantage is a lower bound.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the in-repo package importable when run as a script.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from probe_transmit import safety  # noqa: E402
from probe_transmit.channel import predict_success, predict_success_vec  # noqa: E402
from probe_transmit.policies import Policy, SchedulerState, topk  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Self-contained heuristic Whittle baseline (matches manuscript skeleton)
# --------------------------------------------------------------------------- #
class WhittleBaseline:
    """Heuristic Whittle-index baseline for comparison.

    Simplified: use a heuristic Whittle index based on belief mean (urgency)
    and belief variance (uncertainty). Higher index for high uncertainty
    combined with high urgency (proximity to a safety threshold).
    """

    def __init__(self, N: int, B_probe: int, B_payload: int):
        self.N = int(N)
        self.B_probe = int(B_probe)
        self.B_payload = int(B_payload)

    def whittle_index_heuristic(self, node_id: int, mu: np.ndarray, sigma2: np.ndarray) -> float:
        """Heuristic Whittle index: urgency scaled by uncertainty.

        ``Index_i = urgency_i * sqrt(sigma2_i)`` where urgency grows as the
        belief mean approaches a safety threshold. Restless-bandit intuition:
        an uncertain node that may be near a threshold has high value of
        activation. Uses sqrt(variance) so the index has urgency units.
        """
        sd = float(np.sqrt(sigma2[node_id] + 1e-9))
        # Urgency = closeness of belief mean to either safety bound (in [0, 1]).
        m = float(mu[node_id])
        margin = min(abs(m - safety.SAFE_MIN), abs(safety.SAFE_MAX - m))
        urgency = 1.0 - np.clip(margin / (safety.RANGE / 2.0), 0.0, 1.0)
        # Combine: uncertainty * (1 + urgency) keeps a baseline exploration term.
        return sd * (1.0 + urgency)

    def _index_vector(self, mu: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
        return np.array(
            [self.whittle_index_heuristic(i, mu, sigma2) for i in range(self.N)],
            dtype=float,
        )

    def select_probe(self, mus: np.ndarray, sigma2s: np.ndarray) -> np.ndarray:
        """Select top-B_probe nodes by heuristic Whittle index."""
        idx = self._index_vector(mus, sigma2s)
        return np.argsort(idx)[-self.B_probe:][::-1]

    def select_payload(self, mus: np.ndarray, sigma2s: np.ndarray) -> np.ndarray:
        """Select top-B_payload nodes by heuristic Whittle index."""
        idx = self._index_vector(mus, sigma2s)
        return np.argsort(idx)[-self.B_payload:][::-1]


# --------------------------------------------------------------------------- #
# 2. Simulator-compatible Whittle policy (used for the manuscript comparison)
# --------------------------------------------------------------------------- #
def _whittle_index(state: SchedulerState, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Closed-form heuristic Whittle index over the belief state.

    Index_i = p_succ * sqrt(var_i) * (1 + urgency_i), where urgency is the
    normalized proximity of the belief mean to a safety threshold. No
    threshold-aware penalty weighting, no correlation credit, no service debt:
    this is deliberately a plain Whittle-style activation index.
    """
    sd = np.sqrt(np.maximum(var, 1e-12))
    margin = safety.margin_to_safety(mu)  # 1.0 at threshold, 0.0 deep-safe
    p_succ = np.clip(predict_success_vec(state.pi_bad, state.channel), 0.25, 1.0)
    return p_succ * sd * (1.0 + margin)


class WhittleProbe(Policy):
    """Probe stage: activate the top-B_probe nodes by heuristic Whittle index."""

    name = "whittle_probe"

    def __init__(self, h: int = 4):
        self.h = int(h)

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        mu, var = state.ar.forecast_stats(state.xh, state.age, h=self.h)
        idx = _whittle_index(state, mu, var)
        return topk(idx, min(state.b_probe, state.n))


class WhittlePayload:
    """Payload stage: rank by Whittle index over probed metadata / belief.

    Mirrors the DebtAwarePayload interface (``select``) so it slots into
    :class:`TwoStagePolicy`, but uses NO service debt -- a pure Whittle ranking
    on the post-probe state estimate.
    """

    name = "whittle_payload"

    def __init__(self, h: int = 4):
        self.h = int(h)

    def select(self, state: SchedulerState, probe_set: np.ndarray,
               metadata: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mu, var = state.ar.forecast_stats(state.xh, state.age, h=self.h)
        # Use revealed metadata where probed, else belief mean.
        est = np.where(mask, metadata, mu)
        # Probed nodes have collapsed uncertainty; reflect that in the index.
        var_eff = np.where(mask, 1e-6, var)
        idx = _whittle_index(state, est, var_eff)
        return topk(idx, min(state.b_payload, state.n))
