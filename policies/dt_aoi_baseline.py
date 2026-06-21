"""Digital-Twin + Age-of-Information (DT+AoI) baseline for probe-then-transmit.

This baseline operationalizes the digital-twin synchronization line of recent
work (Guo et al. 2024, "Age-of-Information and Energy Optimization in Digital
Twin Edge Networks"; Cosandal & Ulukus 2026, "Age of Staleness"). The gateway
holds a per-sensor digital twin ``xh_i`` (the AR(1) predictive replica) and the
scheduler synchronizes the twins that are **stale and have drifted**, weighted
by channel quality so a sync only counts if the payload is likely to land.

Twin-sync index (channel-aware AoI of the twin, *no* threshold term):

    DT_i = p_succ_i * age_i * (1 + kappa * drift_i)

where
  - ``age_i``  : Age-of-Information of twin i (steps since last sync) -> the AoI
                 freshness driver, the core of the DT+AoI line.
  - ``drift_i``: |mu_i - xh_i| / RANGE, the expected divergence between the twin
                 and its physical counterpart under the AR(1) forecast -> the
                 "twin fidelity gap" / age-of-staleness signal.
  - ``p_succ_i``: predictive channel-success probability (same per-arm belief
                 filter CAW-VoU uses), so scarce slots are not wasted on twins
                 whose link is deep-fading.
  - ``kappa``  : weight on drift relative to pure age (default 1.0).

Crucially this rule is **threshold-agnostic**: it never consults the safety
boundary. It is the strongest modern freshness-and-fidelity DT baseline we can
build *without* CAW-VoU's threshold-aware VoU term, which is exactly the
comparison of interest -- it isolates the value of the threshold term.

Two variants
------------
1. ``greedy`` (DT+AoI)     : pure channel-aware twin-AoI ranking.
2. ``debt``   (DT+AoI+debt): adds the same w_debt=0.05 fairness floor used by
                             CAW-VoU so quiet twins are not permanently starved.
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
    predict_success_vec,
    service_debt,
    topk,
)

_KAPPA = 1.0


def _twin_score(state: SchedulerState, h: int) -> np.ndarray:
    """Channel-aware twin-AoI index (no threshold term)."""
    mu, _ = state.ar.forecast_stats(state.xh, state.age, h=h)
    drift = np.abs(mu - state.xh) / safety.RANGE
    age = state.age.astype(float)
    p_succ = np.clip(predict_success_vec(state.pi_bad, state.channel), 0.25, 1.0)
    return p_succ * age * (1.0 + _KAPPA * drift)


class DTAoIProbe(Policy):
    """Probe stage: top-B by channel-aware twin Age-of-Information."""

    name = "dt_aoi_probe"

    def __init__(self, variant: str = "greedy", w_debt: float = 0.05, h: int = 4):
        assert variant in {"greedy", "debt"}
        self.variant = variant
        self.w_debt = float(w_debt)
        self.h = int(h)

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        b = min(state.b_probe, state.n)
        if b <= 0:
            return np.empty(0, dtype=int)
        score = _twin_score(state, self.h)
        if self.variant == "debt":
            debt = service_debt(
                state.probe_counts, state.total_probe_choices,
                state.probe_target_share(),
            )
            score = score + self.w_debt * debt * (float(np.max(score)) + 1e-9)
        return topk(score, b)


class DTAoIPayload:
    """Payload stage: sync the twins with the largest revealed divergence + AoI."""

    name = "dt_aoi_payload"

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
        mu, _ = state.ar.forecast_stats(state.xh, state.age, h=self.h)
        age = state.age.astype(float)
        # For probed twins, use the revealed metadata to measure true drift;
        # for unprobed twins fall back to the forecast drift.
        drift = np.where(
            mask,
            np.abs(metadata - state.xh) / safety.RANGE,
            np.abs(mu - state.xh) / safety.RANGE,
        )
        score = age * (1.0 + _KAPPA * drift)
        if self.variant == "debt":
            debt = service_debt(
                state.payload_counts, state.total_payload_choices,
                state.payload_target_share(),
            )
            score = score + self.w_debt * debt * (float(np.max(score)) + 1e-9)
        return topk(score, b)


def make_dt_aoi_policy(variant: str = "greedy", w_debt: float = 0.05) -> TwoStagePolicy:
    label = {"greedy": "DT+AoI", "debt": "DT+AoI+debt"}[variant]
    return TwoStagePolicy(
        name=label,
        probe_policy=DTAoIProbe(variant=variant, w_debt=w_debt),
        payload_policy=DTAoIPayload(variant=variant, w_debt=w_debt),
    )
