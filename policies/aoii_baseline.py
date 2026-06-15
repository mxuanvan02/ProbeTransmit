"""Age of Incorrect Information (AoII) baselines for probe-then-transmit.

AoII (Maatouk et al. 2020; Kam et al.) measures *semantic* freshness: the age
accumulated only while the gateway's held estimate is INCORRECT. Formally, for
node i,

    AoII_i(t) = t - max{ tau <= t : |xh_i(tau) - x_i(tau)| < delta_i },

i.e. the number of steps since the held estimate xh_i was last within an
accuracy band delta_i of the true value x_i. If the estimate is currently
correct, AoII_i = 0; otherwise it increments every step.

Realizability note (honesty for Q1 review)
------------------------------------------
The pull-based gateway does NOT observe x_i before probing, so the exact AoII
indicator |xh_i - x_i| < delta is not computable at decision time. We therefore
implement a **genie-aided AoII oracle**: the AoII counter is updated from the
ground-truth window data inside :class:`AoIITwoStagePolicy.step` (which already
receives ``x_true``). This gives the AoII baselines a *charitable* information
advantage over CAW-VoU (which uses only belief). If CAW-VoU still matches or
beats genie-AoII, the comparison is a strong lower bound on CAW-VoU's merit; if
genie-AoII wins, that is reported honestly as a real threat. A separate
belief-only AoII proxy already exists as ``policies.AoIIProbe`` in the core
package; this module is the oracle counterpart requested for the SOTA table.

Three probe variants (selected via ``variant``):

1. ``greedy``           : score_i = AoII_i.                  (top-B by AoII)
2. ``debt``             : score_i = AoII_i + w_debt * debt_i (same w_debt=0.05
                          fairness floor as CAW-VoU; debt is the probe-stage
                          service deficit in [0,1], so it breaks ties / rescues
                          starved nodes).
3. ``debt_threshold``   : score_i = AoII_i * I(near_threshold_i) + w_debt*debt_i
                          where the near-threshold indicator fires when the
                          belief mean sits within the safety-margin band, so the
                          policy spends AoII budget only on safety-relevant loops.

``AoIIPayload`` mirrors the probe scoring for the transmit stage so each AoII
baseline is a self-consistent two-stage policy (apples-to-apples with CAW-VoU,
which pairs CorrVoUProbe with DebtAwarePayload).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from probe_transmit import safety  # noqa: E402
from probe_transmit.policies import (  # noqa: E402
    Policy,
    SchedulerState,
    TwoStagePolicy,
    service_debt,
    topk,
)

_AOII_KEY = "aoii_counter"
_THRESH_KEY = "aoii_threshold"
_MARGIN_CUTOFF = 0.5  # near-threshold when margin_to_safety(mu) >= this


def _near_threshold(state: SchedulerState, h: int = 4) -> np.ndarray:
    """Indicator (float 0/1) that node i's belief is near a safety bound."""
    mu, _ = state.ar.forecast_stats(state.xh, state.age, h=h)
    return (safety.margin_to_safety(mu) >= _MARGIN_CUTOFF).astype(float)


def _aoii_vector(state: SchedulerState) -> np.ndarray:
    n = state.n
    aoii = state.extras.get(_AOII_KEY)
    if aoii is None or len(aoii) != n:
        aoii = np.zeros(n, dtype=float)
        state.extras[_AOII_KEY] = aoii
    return aoii


def _probe_debt(state: SchedulerState) -> np.ndarray:
    return service_debt(
        state.probe_counts, state.total_probe_choices, state.probe_target_share()
    )


def _payload_debt(state: SchedulerState) -> np.ndarray:
    return service_debt(
        state.payload_counts, state.total_payload_choices, state.payload_target_share()
    )


class AoIIProbeOracle(Policy):
    """Genie-aided AoII probe rule (reads AoII counter maintained by the wrapper)."""

    name = "aoii_oracle_probe"

    def __init__(self, variant: str = "greedy", w_debt: float = 0.05, h: int = 4):
        assert variant in {"greedy", "debt", "debt_threshold"}
        self.variant = variant
        self.w_debt = float(w_debt)
        self.h = int(h)

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        b = min(state.b_probe, state.n)
        if b <= 0:
            return np.empty(0, dtype=int)
        aoii = _aoii_vector(state)
        if self.variant == "greedy":
            score = aoii
        elif self.variant == "debt":
            score = aoii + self.w_debt * _probe_debt(state)
        else:  # debt_threshold
            gate = _near_threshold(state, self.h)
            score = aoii * gate + self.w_debt * _probe_debt(state)
        return topk(score, b)


class AoIIPayload:
    """Genie-aided AoII payload rule, mirroring the probe variant."""

    name = "aoii_oracle_payload"

    def __init__(self, variant: str = "greedy", w_debt: float = 0.05, h: int = 4):
        assert variant in {"greedy", "debt", "debt_threshold"}
        self.variant = variant
        self.w_debt = float(w_debt)
        self.h = int(h)

    def select(self, state: SchedulerState, probe_set: np.ndarray,
               metadata: np.ndarray, mask: np.ndarray) -> np.ndarray:
        b = min(state.b_payload, state.n)
        if b <= 0:
            return np.empty(0, dtype=int)
        aoii = _aoii_vector(state)
        if self.variant == "greedy":
            score = aoii.copy()
        elif self.variant == "debt":
            score = aoii + self.w_debt * _payload_debt(state)
        else:  # debt_threshold
            gate = _near_threshold(state, self.h)
            score = aoii * gate + self.w_debt * _payload_debt(state)
        # Probed loops revealing danger get a small confirmation bonus so probe
        # information is actually used (parallels DebtAwarePayload). Scaled small
        # relative to AoII so it only breaks ties among comparable AoII scores.
        danger = safety.margin_to_safety(np.where(mask, metadata, state.xh))
        score = score + mask.astype(float) * danger * 0.5
        return topk(score, b)


class AoIITwoStagePolicy(TwoStagePolicy):
    """TwoStagePolicy that maintains the genie AoII counter from ``x_true``.

    The AoII counter is updated at the START of each step using the current held
    estimate ``state.xh`` (carried from previous successful payloads) against the
    fresh ground truth ``x_true``. A loop whose held estimate is within the
    accuracy band ``delta`` is "correct" (AoII -> 0); otherwise AoII += 1. The
    accuracy band defaults to the per-node AR(1) one-step residual std, i.e. an
    estimate is correct if it is within one forecast-noise standard deviation,
    but can be overridden via ``state.extras['aoii_threshold']``.
    """

    def _threshold(self, state: SchedulerState) -> np.ndarray:
        thr = state.extras.get(_THRESH_KEY)
        if thr is None:
            override = getattr(self, "_threshold_override", None)
            if override is not None:
                thr = np.broadcast_to(np.asarray(override, dtype=float),
                                      (state.n,)).astype(float).copy()
            else:
                thr = np.asarray(state.ar.sigma, dtype=float).copy()
            state.extras[_THRESH_KEY] = thr
        return np.broadcast_to(thr, (state.n,))

    def step(self, state: SchedulerState, x_true: np.ndarray, *,
             metadata_noise: float, metadata_loss: float):
        aoii = _aoii_vector(state)
        thr = self._threshold(state)
        err = np.abs(state.xh - x_true)
        correct = err < thr
        np.copyto(aoii, np.where(correct, 0.0, aoii + 1.0))
        state.extras[_AOII_KEY] = aoii
        return super().step(
            state, x_true,
            metadata_noise=metadata_noise, metadata_loss=metadata_loss,
        )


def make_aoii_policy(variant: str = "greedy", w_debt: float = 0.05,
                     threshold: float | np.ndarray | None = None) -> AoIITwoStagePolicy:
    """Build a full genie-AoII two-stage policy for the requested variant.

    ``threshold`` overrides the default per-node AR-sigma accuracy band. A scalar
    is broadcast to all nodes. It is stashed into the policy and copied into
    ``state.extras`` on the first step.
    """
    label = {
        "greedy": "AoII-greedy",
        "debt": "AoII+debt",
        "debt_threshold": "AoII+debt+threshold",
    }[variant]
    pol = AoIITwoStagePolicy(
        name=label,
        probe_policy=AoIIProbeOracle(variant=variant, w_debt=w_debt),
        payload_policy=AoIIPayload(variant=variant, w_debt=w_debt),
    )
    pol._threshold_override = (  # type: ignore[attr-defined]
        None if threshold is None else np.asarray(threshold, dtype=float)
    )
    return pol
