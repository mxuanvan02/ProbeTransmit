"""Value-of-Information (VoI) baselines for probe-then-transmit scheduling.

Value of Information here is the expected reduction in mean-squared estimation
error obtained by probing a node. Under the AR(1) Gaussian belief, the gateway's
prior variance for node i after ``age_i`` un-refreshed steps is ``sigma2_i``
(``forecast_stats`` returns it as ``var``). A probe reveals the (near-noiseless)
measurement, collapsing the posterior variance to the measurement-noise floor.
The expected MSE reduction is therefore

    VoI_i = sigma2_i - sigma2_noise  ~=  sigma2_i

so VoI-greedy reduces to **probe the most uncertain nodes** (top-B by sigma2).
This is the maximum-uncertainty / maximum-entropy active-sensing rule and is, as
the task notes, closely related to a Whittle/VoI ranking; we frame it explicitly
as posterior-variance reduction.

Two variants
------------
1. ``greedy`` (VoI-greedy): pure sigma2 ranking for probe and payload.
2. ``debt``   (VoI+debt)  : sigma2 + w_debt * debt_i, adding the same fairness
                            floor (w_debt = 0.05) used by CAW-VoU so VoI does
                            not permanently starve quiet-but-uncertain nodes.

Both pair a VoI probe with a VoI payload to form a consistent two-stage policy,
matching the CAW-VoU (CorrVoUProbe + DebtAwarePayload) structure.
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

# Measurement-noise variance floor subtracted from prior variance to get the
# realizable VoI. Small relative to forecast variance; tracked explicitly so the
# "VoI = posterior-variance reduction" framing is exact rather than hand-wavy.
_SIGMA_NOISE2 = 1e-6


class VoIProbe(Policy):
    """Probe stage: top-B by value of information (posterior variance reduction)."""

    name = "voi_probe"

    def __init__(self, variant: str = "greedy", w_debt: float = 0.05, h: int = 4):
        assert variant in {"greedy", "debt"}
        self.variant = variant
        self.w_debt = float(w_debt)
        self.h = int(h)

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        b = min(state.b_probe, state.n)
        if b <= 0:
            return np.empty(0, dtype=int)
        _, var = state.ar.forecast_stats(state.xh, state.age, h=self.h)
        voi = np.maximum(var - _SIGMA_NOISE2, 0.0)
        if self.variant == "debt":
            debt = service_debt(
                state.probe_counts, state.total_probe_choices,
                state.probe_target_share(),
            )
            # Scale debt to the VoI magnitude so w_debt has comparable leverage
            # to its role in CAW-VoU (where VoU is also an O(1)-ish quantity).
            voi = voi + self.w_debt * debt * (float(np.max(voi)) + 1e-9)
        return topk(voi, b)


class VoIPayload:
    """Payload stage: top-B by VoI, using collapsed variance for probed loops."""

    name = "voi_payload"

    def __init__(self, variant: str = "greedy", w_debt: float = 0.05, h: int = 4):
        assert variant in {"greedy", "debt"}
        self.variant = variant
        self.w_debt = float(w_debt)
        self.h = int(h)

    def select(self, state: SchedulerState, probe_set: np.ndarray,
               metadata: np.ndarray, mask: np.ndarray) -> np.ndarray:
        b = min(state.b_payload, state.n)
        if b <= 0:
            return np.empty(0, dtype=int)
        _, var = state.ar.forecast_stats(state.xh, state.age, h=self.h)
        # Probed loops have revealed information: their remaining uncertainty is
        # the measurement floor, but the VALUE of transmitting them is the gap
        # between revealed metadata and held estimate (what payload would fix).
        revealed_gap = np.where(
            mask, (metadata - state.xh) ** 2 / (safety.RANGE ** 2), 0.0
        )
        voi = np.maximum(var - _SIGMA_NOISE2, 0.0) + revealed_gap
        if self.variant == "debt":
            debt = service_debt(
                state.payload_counts, state.total_payload_choices,
                state.payload_target_share(),
            )
            voi = voi + self.w_debt * debt * (float(np.max(voi)) + 1e-9)
        return topk(voi, b)


def make_voi_policy(variant: str = "greedy", w_debt: float = 0.05) -> TwoStagePolicy:
    label = {"greedy": "VoI-greedy", "debt": "VoI+debt"}[variant]
    return TwoStagePolicy(
        name=label,
        probe_policy=VoIProbe(variant=variant, w_debt=w_debt),
        payload_policy=VoIPayload(variant=variant, w_debt=w_debt),
    )
