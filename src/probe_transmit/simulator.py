"""Two-stage probe-then-transmit simulator.

Runs a single window for a chosen policy. Records per-step diagnostics so the
research wiki can document not only summary metrics but also how often the
debt term changed decisions, how metadata mask varied, and what service-debt
backlogs were observed.
"""
from __future__ import annotations

from dataclasses import asdict
import numpy as np
import pandas as pd

from . import safety
from .channel import CHANNELS, ChannelParams, step_state, stationary_bad_belief, update_bad_belief
from .forecast import AR1Model
from .policies import SchedulerState, TwoStagePolicy, build


def run_window(
    *,
    data: np.ndarray,
    ar: AR1Model,
    channel: ChannelParams,
    policy: TwoStagePolicy,
    start: int,
    horizon: int,
    seed: int,
    b_probe: int,
    b_payload: int,
    metadata_noise: float = 0.0,
    metadata_loss: float = 0.0,
    debt_mode: str = "bounded",
) -> dict:
    rng = np.random.default_rng(seed)
    n = data.shape[1]
    end = start + horizon
    xh = data[start].copy()
    age = np.zeros(n, dtype=int)
    payload_counts = np.zeros(n)
    probe_counts = np.zeros(n)
    last_metadata = data[start].copy()
    last_metadata_age = np.zeros(n, dtype=int)
    pi_bad = stationary_bad_belief(channel)
    bad = False

    state = SchedulerState(
        n=n,
        b_probe=b_probe,
        b_payload=b_payload,
        rng=rng,
        ar=ar,
        channel=channel,
        xh=xh,
        age=age,
        payload_counts=payload_counts,
        probe_counts=probe_counts,
        total_payload_choices=0,
        total_probe_choices=0,
        pi_bad=pi_bad,
        last_metadata=last_metadata,
        last_metadata_age=last_metadata_age,
        debt_mode=debt_mode,
        probe_service_age=np.zeros(n, dtype=float),
        payload_service_age=np.zeros(n, dtype=float),
    )
    err_history = []
    losses = []
    missed = []
    metadata_seen = []
    payload_success_history = []
    debt_changes = 0
    payload_steps = 0

    for k in range(start + 1, end):
        x_true = data[k]
        probe_set, payload_set, metadata, mask = policy.step(
            state,
            x_true,
            metadata_noise=metadata_noise,
            metadata_loss=metadata_loss,
        )
        # Record probe usage
        for i in probe_set:
            state.probe_counts[i] += 1
            state.total_probe_choices += 1
        # Accumulating-debt bookkeeping: age-of-service for the probe stage
        # (slots since last probed). Served -> 0, others +1. Used only when
        # debt_mode == "accumulating"; harmless otherwise.
        if state.probe_service_age is not None and b_probe > 0:
            state.probe_service_age += 1.0
            if len(probe_set) > 0:
                state.probe_service_age[np.asarray(probe_set, dtype=int)] = 0.0
        metadata_seen.append(float(mask.mean()) if state.b_probe > 0 else 0.0)

        # Track whether payload set differs from a safety-only ranking (without debt).
        from .policies import metadata_safety_value
        values_only = metadata_safety_value(state, metadata, mask)
        safety_only_top = tuple(np.argsort(values_only)[::-1][: state.b_payload])
        actual_top = tuple(payload_set)
        if safety_only_top != actual_top:
            debt_changes += 1
        payload_steps += 1

        # Apply payload deliveries through the channel.
        state.age += 1
        for i in payload_set:
            state.payload_counts[i] += 1
            state.total_payload_choices += 1
            ok, bad = step_state(channel, rng, bad)
            payload_success_history.append(int(ok))
            state.pi_bad = update_bad_belief(state.pi_bad, ok, channel)
            if ok:
                state.xh[i] = x_true[i]
                state.age[i] = 0
        # Accumulating-debt bookkeeping: age-of-service for the payload stage
        # (slots since last selected for payload). Served -> 0, others +1.
        if state.payload_service_age is not None:
            state.payload_service_age += 1.0
            if len(payload_set) > 0:
                state.payload_service_age[np.asarray(payload_set, dtype=int)] = 0.0

        # Per-step metrics
        step_loss, step_missed, _ = safety.loss_terms(x_true, state.xh)
        losses.append(step_loss)
        missed.append(step_missed)
        err_history.append((state.xh - x_true).copy())

    err_arr = np.asarray(err_history)
    losses_arr = np.asarray(losses)
    missed_arr = np.asarray(missed)
    return {
        "policy": policy.name,
        "channel": channel.name,
        "seed": seed,
        "start": start,
        "horizon": horizon,
        "b_probe": b_probe,
        "b_payload": b_payload,
        "metadata_noise": metadata_noise,
        "metadata_loss": metadata_loss,
        "debt_mode": debt_mode,
        "rmse_mean": float(np.mean(np.sqrt(np.mean(err_arr ** 2, axis=0)))),
        "loss_mean": float(losses_arr.mean()),
        "missed_violation_pct": float(100 * missed_arr.mean()),
        "metadata_seen_frac_mean": float(np.mean(metadata_seen)),
        "debt_changed_decision_pct": float(100 * debt_changes / max(payload_steps, 1)),
        "payload_success_pct": float(100 * np.mean(payload_success_history)) if payload_success_history else 100.0,
        "payload_fairness_jain": safety.jain(state.payload_counts),
        "probe_fairness_jain": safety.jain(state.probe_counts) if state.b_probe > 0 else 1.0,
        "max_age": float(state.age.max()),
        "avg_age": float(state.age.mean()),
        "max_payload_debt": float(np.max(np.maximum(state.payload_target_share() * state.total_payload_choices - state.payload_counts, 0.0)) / max(state.total_payload_choices, 1)),
    }


def make_policies(b_probe: int) -> list[TwoStagePolicy]:
    rules = ["active", "random", "round_robin", "max_aoi"] if b_probe > 0 else ["active"]
    return [build(name) for name in rules]
