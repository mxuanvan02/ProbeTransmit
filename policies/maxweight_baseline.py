"""MaxWeight / Lyapunov-drift scheduling baselines for probe-then-transmit.

MaxWeight scheduling (Tassiulas-Ephremides; for AoI: Kadota et al. 2018,
"Scheduling Policies for Minimizing Age of Information in Broadcast Wireless
Networks") maintains a per-node virtual queue Q_i(t) and serves the nodes with
the largest weighted backlog. The queue dynamics are

    Q_i(t+1) = max(Q_i(t) - service_i(t), 0) + arrival_i(t)

where ``service_i(t) = 1`` iff node i is scheduled this slot and
``arrival_i(t)`` is the per-slot urgency/work that accrues to node i. The
scheduler picks the top-B nodes by ``Q_i(t) * weight_i`` (here weight_i = 1, so
plain MaxWeight). This is the Lyapunov-drift-minimizing one-slot rule for the
quadratic Lyapunov function ``L(Q) = 0.5 * sum_i Q_i^2``.

Two variants
------------
1. ``aoi`` (MaxWeight-AoI): the virtual queue tracks **age since last probe**.
   arrival = +1 every slot, service drains the queue when probed. This is the
   canonical age-based MaxWeight backlog: a node not probed for a long time has
   a large queue and is prioritized. Equivalent in spirit to Kadota's AoI
   MaxWeight applied to the probe channel.

2. ``vou`` (MaxWeight-VoU): the virtual queue **accumulates per-slot VoU**
   (value-of-update urgency) and is drained on service. This is the bare
   MaxWeight/Lyapunov analogue of CAW-VoU's urgency signal but WITHOUT the
   correlation credit and WITHOUT the closed-form fairness debt -- only the
   Lyapunov queue provides implicit fairness. Comparing it to CAW-VoU isolates
   the value added by correlation + debt over a plain drift-plus-penalty rule.

Queues are stored in ``state.extras`` so they persist across simulator steps,
and are drained inside :class:`MaxWeightTwoStagePolicy.step` after the simulator
applies the schedule (we drain in ``step`` because that is the only hook that
knows the realized probe/payload sets).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from probe_transmit import safety  # noqa: E402
from probe_transmit.channel import predict_success, predict_success_vec  # noqa: E402
from probe_transmit.policies import (  # noqa: E402
    Policy,
    SchedulerState,
    TwoStagePolicy,
    topk,
)

_PROBE_Q = "mw_probe_queue"
_PAY_Q = "mw_payload_queue"


def _probe_queue(state: SchedulerState) -> np.ndarray:
    q = state.extras.get(_PROBE_Q)
    if q is None or len(q) != state.n:
        q = np.zeros(state.n, dtype=float)
        state.extras[_PROBE_Q] = q
    return q


def _payload_queue(state: SchedulerState) -> np.ndarray:
    q = state.extras.get(_PAY_Q)
    if q is None or len(q) != state.n:
        q = np.zeros(state.n, dtype=float)
        state.extras[_PAY_Q] = q
    return q


def _vou_arrival(state: SchedulerState, h: int = 4) -> np.ndarray:
    """Per-slot VoU urgency arrival: tracking gap + empirically-priced safety risk.

    Identical functional form to ``VoUProbe`` so MaxWeight-VoU's *urgency input*
    matches CAW-VoU's, isolating the scheduling mechanism (Lyapunov queue vs
    correlation+debt) rather than the urgency definition.
    """
    mu, var = state.ar.forecast_stats(state.xh, state.age, h=h)
    track_gap = (mu - state.xh) ** 2 / (safety.RANGE ** 2)
    pvio = state.ar.empirical_safety_prob(
        mu, var, safe_min=safety.SAFE_MIN, safe_max=safety.SAFE_MAX
    )
    xh_safe = ((state.xh >= safety.SAFE_MIN) & (state.xh <= safety.SAFE_MAX)).astype(float)
    return track_gap + 6.0 * pvio * xh_safe


class MaxWeightProbe(Policy):
    """Probe stage: select top-B by virtual-queue backlog Q_i (MaxWeight)."""

    name = "maxweight_probe"

    def __init__(self, variant: str = "aoi", h: int = 4):
        assert variant in {"aoi", "vou"}
        self.variant = variant
        self.h = int(h)

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        b = min(state.b_probe, state.n)
        if b <= 0:
            return np.empty(0, dtype=int)
        q = _probe_queue(state)
        # Add the current slot's arrival to the backlog BEFORE selecting so the
        # decision reflects up-to-date work (drain happens post-schedule in step).
        if self.variant == "aoi":
            arrival = np.ones(state.n, dtype=float)
        else:  # vou
            arrival = _vou_arrival(state, self.h)
        score = q + arrival
        # Channel scaling keeps MaxWeight-VoU comparable to VoU/CAW-VoU, which
        # weight by delivery probability. For AoI we leave the queue unscaled
        # (canonical age MaxWeight).
        if self.variant == "vou":
            p_succ = np.clip(predict_success_vec(state.pi_bad, state.channel), 0.25, 1.0)
            score = p_succ * score
        return topk(score, b)


class MaxWeightPayload:
    """Payload stage: select top-B by virtual-queue backlog (MaxWeight)."""

    name = "maxweight_payload"

    def __init__(self, variant: str = "aoi", h: int = 4):
        assert variant in {"aoi", "vou"}
        self.variant = variant
        self.h = int(h)

    def select(self, state: SchedulerState, probe_set: np.ndarray,
               metadata: np.ndarray, mask: np.ndarray) -> np.ndarray:
        b = min(state.b_payload, state.n)
        if b <= 0:
            return np.empty(0, dtype=int)
        q = _payload_queue(state)
        if self.variant == "aoi":
            # Age-based backlog already tracked by simulator state.age; combine
            # with the payload queue for a consistent age MaxWeight.
            score = q + state.age.astype(float)
        else:  # vou -- use revealed metadata to refine urgency where probed
            mu, var = state.ar.forecast_stats(state.xh, state.age, h=self.h)
            est = np.where(mask, metadata, mu)
            track_gap = (est - state.xh) ** 2 / (safety.RANGE ** 2)
            var_eff = np.where(mask, 1e-6, var)
            pvio = state.ar.empirical_safety_prob(
                est, var_eff, safe_min=safety.SAFE_MIN, safe_max=safety.SAFE_MAX
            )
            xh_safe = ((state.xh >= safety.SAFE_MIN) & (state.xh <= safety.SAFE_MAX)).astype(float)
            arrival = track_gap + 6.0 * pvio * xh_safe
            p_succ = np.clip(predict_success_vec(state.pi_bad, state.channel), 0.25, 1.0)
            score = p_succ * (q + arrival)
        return topk(score, b)


class MaxWeightTwoStagePolicy(TwoStagePolicy):
    """TwoStagePolicy that updates MaxWeight virtual queues each step.

    Queue dynamics, applied AFTER the schedule is computed for this slot:

        Q_i <- max(Q_i + arrival_i - service_i_drain, 0)

    For the AoI variant, ``arrival = 1`` and a scheduled node's queue resets to
    0 (age cleared). For the VoU variant, ``arrival = VoU_i`` and a scheduled
    node's queue is drained to 0 (its accumulated urgency is "served"). Draining
    to 0 on service is the standard infinite-service-rate AoI/VoU MaxWeight
    convention used by Kadota et al.
    """

    def __init__(self, name: str, probe_policy: Policy, payload_policy,
                 variant: str = "aoi", h: int = 4):
        super().__init__(name=name, probe_policy=probe_policy,
                         payload_policy=payload_policy)
        self.variant = variant
        self.h = int(h)

    def step(self, state: SchedulerState, x_true: np.ndarray, *,
             metadata_noise: float, metadata_loss: float):
        pq = _probe_queue(state)
        yq = _payload_queue(state)
        # Arrivals for this slot (snapshot before selection mutates beliefs).
        if self.variant == "aoi":
            probe_arr = np.ones(state.n, dtype=float)
            pay_arr = np.ones(state.n, dtype=float)
        else:
            probe_arr = _vou_arrival(state, self.h)
            pay_arr = probe_arr.copy()
        # Accumulate arrivals into the persistent queues.
        pq += probe_arr
        yq += pay_arr

        probe_set, payload_set, metadata, mask = super().step(
            state, x_true,
            metadata_noise=metadata_noise, metadata_loss=metadata_loss,
        )

        # Drain served nodes to 0 (infinite-rate service).
        if len(probe_set) > 0:
            pq[np.asarray(probe_set, dtype=int)] = 0.0
        if len(payload_set) > 0:
            yq[np.asarray(payload_set, dtype=int)] = 0.0
        np.maximum(pq, 0.0, out=pq)
        np.maximum(yq, 0.0, out=yq)
        state.extras[_PROBE_Q] = pq
        state.extras[_PAY_Q] = yq
        return probe_set, payload_set, metadata, mask


def make_maxweight_policy(variant: str = "aoi", h: int = 4) -> MaxWeightTwoStagePolicy:
    label = {"aoi": "MaxWeight-AoI", "vou": "MaxWeight-VoU"}[variant]
    return MaxWeightTwoStagePolicy(
        name=label,
        probe_policy=MaxWeightProbe(variant=variant, h=h),
        payload_policy=MaxWeightPayload(variant=variant, h=h),
        variant=variant, h=h,
    )
