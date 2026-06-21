"""Two-stage probe-then-transmit policies.

Each policy implements two functions:

- ``probe(state)`` returns a list of probed loop indices. Decides Stage 1.
- ``payload(state, probe_set, metadata)`` returns the chosen payload set.

Policies share the same scheduler state object so that the simulator does not
need to know which policy is running.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import safety
from .channel import ChannelParams, predict_success, predict_success_vec
from .forecast import AR1Model


@dataclass
class SchedulerState:
    n: int
    b_probe: int
    b_payload: int
    rng: np.random.Generator
    ar: AR1Model
    channel: ChannelParams
    xh: np.ndarray
    age: np.ndarray
    payload_counts: np.ndarray
    probe_counts: np.ndarray
    total_payload_choices: int
    total_probe_choices: int
    pi_bad: np.ndarray
    last_metadata: np.ndarray
    last_metadata_age: np.ndarray
    extras: dict = field(default_factory=dict)
    # --- fairness debt model ---------------------------------------------
    # "bounded":      deployed normalized deficit service_debt() in [0,1] (default,
    #                 backward compatible). Soft fairness, no hard starvation bound.
    # "accumulating": age-of-service surrogate analyzed in Theorem 1 (05_theory.tex).
    #                 Grows without bound while a node waits, so it carries the hard
    #                 closed-form starvation deadline. Normalized by the target share
    #                 (age * B/N) so it lives on the same ~[0,1] scale as the bounded
    #                 deficit; this is an exact reparametrization w' = w*(B/N) of the
    #                 theorem rule and leaves the hard bound valid.
    debt_mode: str = "bounded"
    probe_service_age: np.ndarray | None = None
    payload_service_age: np.ndarray | None = None

    def payload_target_share(self) -> float:
        return self.b_payload / max(self.n, 1)

    def probe_target_share(self) -> float:
        return self.b_probe / max(self.n, 1)

    def fairness_debt(self, stage: str) -> np.ndarray:
        """Return the per-node fairness debt vector for the requested stage,
        dispatching on :attr:`debt_mode`.

        - ``bounded``      -> :func:`service_debt` (deployed deficit, in [0,1]).
        - ``accumulating`` -> :func:`accumulating_debt` (age-of-service surrogate
          from Theorem 1), normalized by the target share so its scale matches
          the bounded deficit and the existing ``w_debt`` weights stay meaningful.
        """
        if stage == "probe":
            counts, total = self.probe_counts, self.total_probe_choices
            share = self.probe_target_share()
            age = self.probe_service_age
        elif stage == "payload":
            counts, total = self.payload_counts, self.total_payload_choices
            share = self.payload_target_share()
            age = self.payload_service_age
        else:
            raise ValueError(f"unknown fairness stage {stage!r}")
        if self.debt_mode == "accumulating":
            if age is None:
                age = np.zeros(self.n, dtype=float)
            return accumulating_debt(age, share)
        return service_debt(counts, total, share)


# ---------------- shared scoring primitives ----------------------------------

def service_debt(counts: np.ndarray, total_choices: int, target_share: float) -> np.ndarray:
    if total_choices <= 0 or target_share <= 0:
        return np.zeros_like(counts, dtype=float)
    share = counts / max(total_choices, 1)
    return np.maximum(target_share - share, 0.0) / max(target_share, 1e-9)


def accumulating_debt(service_age: np.ndarray, target_share: float) -> np.ndarray:
    """Age-of-service fairness debt (the model analyzed in Theorem 1).

    ``service_age[i]`` is the number of slots since node ``i`` was last served in
    this stage (reset to 0 on service, +1 otherwise). Unlike the bounded deficit,
    it grows without bound while a node waits, which is exactly what lets the
    debt term eventually dominate any bounded urgency and yields the hard
    closed-form starvation deadline of Theorem 1.

    We multiply by ``target_share`` (= B/N) so a node served at the round-robin
    rate sits near debt 1, matching the scale of :func:`service_debt`. This is a
    pure reparametrization ``w' = w * target_share`` of the theorem's selection
    rule and therefore preserves the bound (with ``w`` replaced by ``w'``).
    """
    if target_share <= 0:
        return np.zeros_like(service_age, dtype=float)
    return np.asarray(service_age, dtype=float) * target_share


def gateway_pessimistic_value(state: SchedulerState) -> np.ndarray:
    mu, var = state.ar.forecast_stats(state.xh, state.age, h=4)
    sd = np.sqrt(var)
    high_state = mu + 1.64 * sd
    low_state = mu - 1.64 * sd
    high_margin = np.abs(high_state - safety.SAFE_MAX)
    low_margin = np.abs(low_state - safety.SAFE_MIN)
    pseudo = np.where(high_margin < low_margin, high_state, low_state)
    pseudo_loss, _, _ = safety.loss_terms(pseudo, state.xh)
    return pseudo_loss + 0.05 * np.log1p(state.age)


def belief_uncertainty(state: SchedulerState) -> np.ndarray:
    mu, var = state.ar.forecast_stats(state.xh, state.age, h=4)
    sd = np.sqrt(var) / safety.RANGE
    pvio = state.ar.empirical_safety_prob(mu, var, safe_min=safety.SAFE_MIN, safe_max=safety.SAFE_MAX)
    return 1.5 * sd ** 2 + 4.0 * pvio + 0.10 * np.log1p(state.age)


def metadata_safety_value(state: SchedulerState, metadata: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Estimate per-loop safety value using metadata when available, else forecast."""
    mu, var = state.ar.forecast_stats(state.xh, state.age, h=4)
    proxy = np.where(mask, metadata, mu)
    cur, _, _ = safety.loss_terms(proxy, state.xh)
    values = np.zeros(state.n)
    for i in range(state.n):
        cand = state.xh.copy()
        cand[i] = proxy[i]
        nxt, _, _ = safety.loss_terms(proxy, cand)
        values[i] = float(np.sum(cur - nxt))
    if not mask.all():
        gw = gateway_pessimistic_value(state)
        values = values + np.where(mask, 0.0, 0.4 * safety.normalize_unit(gw))
    return values


# ---------------- policy interface ------------------------------------------

@dataclass
class PolicyResult:
    name: str
    probe_score_used: bool
    payload_score_used: bool


class Policy:
    name: str = "base"

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        raise NotImplementedError

    def select_payload(self, state: SchedulerState, probe_set: np.ndarray, metadata: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.empty(0, dtype=int)
    return np.argsort(scores)[::-1][:k]


# ---------------- baseline probe rules --------------------------------------

class RandomProbe(Policy):
    name = "random_probe"

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        return state.rng.choice(state.n, size=min(state.b_probe, state.n), replace=False)


class RoundRobinProbe(Policy):
    name = "round_robin_probe"

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        cursor = state.extras.setdefault("rr_probe_cursor", 0)
        idx = [(cursor + j) % state.n for j in range(min(state.b_probe, state.n))]
        state.extras["rr_probe_cursor"] = (cursor + max(state.b_probe, 1)) % max(state.n, 1)
        return np.asarray(idx, dtype=int)


class MaxAoIProbe(Policy):
    name = "max_aoi_probe"

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        return topk(state.age.astype(float), min(state.b_probe, state.n))


class AoIIProbe(Policy):
    """Age of Incorrect Information (AoII) probe rule.

    Classic AoII = age * indicator(estimate != true). The gateway does not know
    the true state before probing, so we use a documented proxy that keeps the
    AoII spirit (penalise *stale AND likely-wrong* loops) while staying
    computable from belief state only:

        AoII_proxy_i = age_i * incorrectness_i

    where ``incorrectness_i`` blends two belief signals:

    - forecast drift: |mu_i(after age) - xh_i| / RANGE, i.e. how far the
      gateway's held value xh has likely drifted from the AR(1) forecast. This
      is the expected magnitude of the error indicator.
    - violation probability: P(x_i outside [SAFE_MIN, SAFE_MAX]) from the
      empirical safety model, capturing semantic "incorrectness" that matters
      for safety even when the numeric drift is small.

    The two are combined as ``drift + w_vio * pvio`` so a loop that is either
    numerically stale or near a safety threshold accrues AoII. This recovers the
    pure AoII metric in the limit where incorrectness -> indicator(error).
    """

    name = "aoii_probe"

    def __init__(self, w_vio: float = 1.0, h: int = 4):
        self.w_vio = float(w_vio)
        self.h = int(h)

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        mu, var = state.ar.forecast_stats(state.xh, state.age, h=self.h)
        drift = np.abs(mu - state.xh) / safety.RANGE
        pvio = state.ar.empirical_safety_prob(
            mu, var, safe_min=safety.SAFE_MIN, safe_max=safety.SAFE_MAX
        )
        incorrectness = drift + self.w_vio * pvio
        aoii = state.age.astype(float) * incorrectness
        return topk(aoii, min(state.b_probe, state.n))


class VoUProbe(Policy):
    """Value of Update (VoU) probe rule.

    VoU_i = expected reduction in cost if loop i is updated now. The simulator
    cost is ``track + safety`` (see ``safety.loss_terms``). If the gateway keeps
    the held estimate xh_i, the expected per-step cost under the AR(1) belief is

        C_keep_i = E[(x - xh_i)^2] / RANGE^2 + lambda * P(x violates, xh safe)
                 = ((mu_i - xh_i)^2 + var_i) / RANGE^2
                   + lambda * pvio_i * I(xh_i in safe band).

    If instead loop i were perfectly refreshed (xh_i := x_i), the residual
    tracking cost collapses toward var_i / RANGE^2 (one-step-ahead uncertainty)
    and the *stale-while-safe* miss term is removed. Hence

        VoU_i = C_keep_i - C_refresh_i
              = (mu_i - xh_i)^2 / RANGE^2 + lambda * pvio_i * I(xh_i safe),

    scaled by the channel success probability because an update only lands if
    the payload delivery succeeds. We probe the loops with the highest VoU so
    Stage 2 can act on the most cost-reducing refreshes.
    """

    name = "vou_probe"

    def __init__(self, lambda_safety: float = 6.0, h: int = 4):
        self.lambda_safety = float(lambda_safety)
        self.h = int(h)

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        mu, var = state.ar.forecast_stats(state.xh, state.age, h=self.h)
        track_gap = (mu - state.xh) ** 2 / (safety.RANGE ** 2)
        pvio = state.ar.empirical_safety_prob(
            mu, var, safe_min=safety.SAFE_MIN, safe_max=safety.SAFE_MAX
        )
        xh_safe = ((state.xh >= safety.SAFE_MIN) & (state.xh <= safety.SAFE_MAX)).astype(float)
        vou = track_gap + self.lambda_safety * pvio * xh_safe
        p_succ = np.clip(predict_success_vec(state.pi_bad, state.channel), 0.25, 1.0)
        return topk(p_succ * vou, min(state.b_probe, state.n))


class UrgencyProbe(Policy):
    """Urgency probe rule: prioritise loops closest to a safety threshold.

    Urgency_i measures how near the gateway's belief about loop i is to a safety
    boundary. Two complementary signals are combined:

    - margin urgency: ``margin_to_safety(mu_i)`` -> 1.0 when the forecast mean
      sits at a threshold, 0.0 when it is comfortably mid-band.
    - tail urgency: ``pvio_i`` -> probability the loop is actually outside the
      safe band given forecast uncertainty.

    Urgency_i = max(margin_urgency_i, tail_urgency_i). Using the max keeps a
    loop urgent if *either* its expected value is near the edge or its
    uncertainty tail crosses the boundary. A small staleness nudge breaks ties
    toward loops the gateway has not refreshed recently.
    """

    name = "urgency_probe"

    def __init__(self, w_stale: float = 0.05, h: int = 4):
        self.w_stale = float(w_stale)
        self.h = int(h)

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        mu, var = state.ar.forecast_stats(state.xh, state.age, h=self.h)
        margin_urgency = safety.margin_to_safety(mu)
        tail_urgency = state.ar.empirical_safety_prob(
            mu, var, safe_min=safety.SAFE_MIN, safe_max=safety.SAFE_MAX
        )
        urgency = np.maximum(margin_urgency, tail_urgency)
        score = urgency + self.w_stale * safety.normalize_unit(state.age.astype(float))
        return topk(score, min(state.b_probe, state.n))


class WhittleVoIProbe(Policy):
    """Whittle Index for Probing via Lagrangian Relaxation (POMDP / VoI).

    Implements the design from `docs/theory_deepmind_breakthrough.md`:

    1. Compute prior belief (mu, var) per loop using AR(1) gateway state.
    2. Compute the per-loop Payload Index
           I2(b) = P_succ * E_x[((x - xh)/RANGE)^2 + lambda * I(x violates, xh safe)]
       on the prior belief; sort descending.
    3. Set adaptive Lagrange multiplier nu_2 at position (B_payload + 1).
    4. Approximate the expectation over the future probe observation y_i with
       Gauss-Hermite quadrature of order ``deg`` (default 5):
           y_{i,k} = mu_i + z_k * sqrt(2 * (sigma_i^2 + sigma_noise^2))
           E_y[max(I2(b(y)) - nu_2, 0)] ~= (1/sqrt(pi)) * sum_k w_k * max(...)
    5. Probe Index = V_probe - V_no_probe; pick top B_probe.

    Bayesian belief update inside the quadrature uses the scalar Kalman gain
        K = sigma^2 / (sigma^2 + sigma_noise^2),
    giving posterior mean mu' = mu + K (y - mu) and variance (1 - K) sigma^2.
    """

    name = "whittle_voi_probe"

    def __init__(self, lambda_safety: float = 6.0, deg: int = 5,
                 metadata_noise_std: float = 0.0):
        self.lambda_safety = float(lambda_safety)
        self.deg = int(deg)
        self.metadata_noise_var = float(metadata_noise_std) ** 2
        nodes, weights = np.polynomial.hermite.hermgauss(self.deg)
        self.gh_nodes = nodes
        self.gh_weights = weights
        self.gh_norm = 1.0 / float(np.sqrt(np.pi))

    @staticmethod
    def _erf_vec(z: np.ndarray) -> np.ndarray:
        try:
            from scipy.special import erf as _erf  # type: ignore
            return _erf(z)
        except Exception:  # pragma: no cover - fallback only
            from math import erf
            return np.vectorize(erf)(z)

    def _payload_index(self, mu: np.ndarray, var: np.ndarray,
                       xh: np.ndarray, p_succ: float,
                       ar_model=None) -> np.ndarray:
        """E[risk(x, xh)] for x ~ N(mu, var), scaled by channel success.
        
        v2 FIX: Use empirical_safety_prob instead of Gaussian erf to better
        capture heavy-tailed greenhouse temperature excursions.
        """
        var_safe = np.maximum(var, 1e-12)
        track_term = ((mu - xh) ** 2 + var_safe) / (safety.RANGE ** 2)
        
        # v2: Use empirical tail probability if AR model available
        if ar_model is not None and hasattr(ar_model, 'empirical_safety_prob'):
            p_violate = ar_model.empirical_safety_prob(
                mu, var_safe, safe_min=safety.SAFE_MIN, safe_max=safety.SAFE_MAX
            )
        else:
            # Fallback to Gaussian erf
            sd = np.sqrt(var_safe)
            sqrt2 = np.sqrt(2.0)
            z_lo = (safety.SAFE_MIN - mu) / (sd * sqrt2)
            z_hi = (safety.SAFE_MAX - mu) / (sd * sqrt2)
            cdf_lo = 0.5 * (1.0 + self._erf_vec(z_lo))
            cdf_hi = 0.5 * (1.0 + self._erf_vec(z_hi))
            p_violate = np.clip(cdf_lo + (1.0 - cdf_hi), 0.0, 1.0)
        
        xh_safe = ((xh >= safety.SAFE_MIN) & (xh <= safety.SAFE_MAX)).astype(float)
        return p_succ * (track_term + self.lambda_safety * p_violate * xh_safe)

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        mu, var = state.ar.forecast_stats(state.xh, state.age, h=4)
        p_succ = np.clip(predict_success_vec(state.pi_bad, state.channel), 0.25, 1.0)

        # Step 2: Payload Index on prior belief (v2: use empirical_safety_prob).
        i2_prior = self._payload_index(mu, var, state.xh, p_succ, ar_model=state.ar)

        # Step 3: adaptive Lagrange multiplier nu_2 = (B_payload + 1)-th score.
        if state.b_payload <= 0:
            nu_2 = float(np.max(i2_prior)) + 1.0
        elif state.b_payload >= state.n:
            nu_2 = 0.0
        else:
            sorted_desc = np.sort(i2_prior)[::-1]
            nu_2 = float(sorted_desc[state.b_payload])

        v_no_probe = np.maximum(i2_prior - nu_2, 0.0)

        # Step 4: Gauss-Hermite quadrature for V_probe.
        y_var = var + self.metadata_noise_var
        K = var / np.maximum(y_var, 1e-12)
        post_var = np.maximum((1.0 - K) * var, 0.0)
        y_sd = np.sqrt(np.maximum(y_var, 1e-12))
        sqrt2 = np.sqrt(2.0)

        v_probe = np.zeros(state.n, dtype=float)
        for k_idx in range(self.deg):
            z_k = float(self.gh_nodes[k_idx])
            w_k = float(self.gh_weights[k_idx])
            y_k = mu + z_k * sqrt2 * y_sd
            mu_post = mu + K * (y_k - mu)
            i2_post = self._payload_index(mu_post, post_var, state.xh, p_succ, ar_model=state.ar)
            v_probe += w_k * np.maximum(i2_post - nu_2, 0.0)
        v_probe *= self.gh_norm

        probe_index = v_probe - v_no_probe
        return topk(probe_index, min(state.b_probe, state.n))


class WhittleVoUShield(Policy):
    """Idea 2: Whittle/VoI ranking with VoU-style empirical violation shield.
    
    Two-tier candidate selection:
    1. Shield set (priority): loops with high empirical violation probability
    2. Whittle fill: remaining budget filled by descending Whittle VoI index
    
    This hybrid preserves Whittle's RMSE edge while importing VoU's winning
    safety mechanism (the source of its loss_mean victory).
    """
    
    name = "whittle_vou_shield"
    
    def __init__(self, lambda_safety: float = 15.0, shield_threshold: float = 0.3,
                 shield_budget_fraction: float = 0.5, deg: int = 5,
                 metadata_noise_std: float = 0.0):
        self.lambda_safety = float(lambda_safety)
        self.shield_threshold = float(shield_threshold)
        self.shield_budget_fraction = float(shield_budget_fraction)
        self.deg = int(deg)
        self.metadata_noise_var = float(metadata_noise_std) ** 2
        nodes, weights = np.polynomial.hermite.hermgauss(self.deg)
        self.gh_nodes = nodes
        self.gh_weights = weights
        self.gh_norm = 1.0 / float(np.sqrt(np.pi))
    
    def _payload_index_empirical(self, mu: np.ndarray, var: np.ndarray,
                                  xh: np.ndarray, p_succ: float,
                                  ar_model) -> np.ndarray:
        """Payload index using empirical safety probability."""
        var_safe = np.maximum(var, 1e-12)
        track_term = ((mu - xh) ** 2 + var_safe) / (safety.RANGE ** 2)
        p_violate = ar_model.empirical_safety_prob(
            mu, var_safe, safe_min=safety.SAFE_MIN, safe_max=safety.SAFE_MAX
        )
        xh_safe = ((xh >= safety.SAFE_MIN) & (xh <= safety.SAFE_MAX)).astype(float)
        return p_succ * (track_term + self.lambda_safety * p_violate * xh_safe)
    
    def select_probe(self, state: SchedulerState) -> np.ndarray:
        mu, var = state.ar.forecast_stats(state.xh, state.age, h=4)
        p_succ = np.clip(predict_success_vec(state.pi_bad, state.channel), 0.25, 1.0)
        
        # Step 1: Identify shield set (high violation probability loops)
        pvio = state.ar.empirical_safety_prob(
            mu, var, safe_min=safety.SAFE_MIN, safe_max=safety.SAFE_MAX
        )
        shield_mask = pvio >= self.shield_threshold
        shield_indices = np.where(shield_mask)[0]
        
        # Step 2: Compute Whittle VoI index for all loops
        i2_prior = self._payload_index_empirical(mu, var, state.xh, p_succ, state.ar)
        
        # Adaptive Lagrange multiplier
        if state.b_payload <= 0:
            nu_2 = float(np.max(i2_prior)) + 1.0
        elif state.b_payload >= state.n:
            nu_2 = 0.0
        else:
            sorted_desc = np.sort(i2_prior)[::-1]
            nu_2 = float(sorted_desc[min(state.b_payload, len(sorted_desc)-1)])
        
        v_no_probe = np.maximum(i2_prior - nu_2, 0.0)
        
        # Gauss-Hermite quadrature for V_probe
        y_var = var + self.metadata_noise_var
        K = var / np.maximum(y_var, 1e-12)
        post_var = np.maximum((1.0 - K) * var, 0.0)
        y_sd = np.sqrt(np.maximum(y_var, 1e-12))
        sqrt2 = np.sqrt(2.0)
        
        v_probe = np.zeros(state.n, dtype=float)
        for k_idx in range(self.deg):
            z_k = float(self.gh_nodes[k_idx])
            w_k = float(self.gh_weights[k_idx])
            y_k = mu + z_k * sqrt2 * y_sd
            mu_post = mu + K * (y_k - mu)
            i2_post = self._payload_index_empirical(mu_post, post_var, state.xh, p_succ, state.ar)
            v_probe += w_k * np.maximum(i2_post - nu_2, 0.0)
        v_probe *= self.gh_norm
        
        whittle_index = v_probe - v_no_probe
        
        # Step 3: Two-tier selection
        max_shield = int(self.shield_budget_fraction * state.b_probe)
        if len(shield_indices) > 0:
            shield_pvio = pvio[shield_indices]
            shield_order = np.argsort(-shield_pvio)
            shield_selected = shield_indices[shield_order[:min(max_shield, len(shield_indices))]]
        else:
            shield_selected = np.array([], dtype=int)
        
        remaining_budget = state.b_probe - len(shield_selected)
        if remaining_budget > 0:
            whittle_masked = whittle_index.copy()
            if len(shield_selected) > 0:
                whittle_masked[shield_selected] = -np.inf
            whittle_fill = topk(whittle_masked, remaining_budget)
        else:
            whittle_fill = np.array([], dtype=int)
        
        selected = np.concatenate([shield_selected, whittle_fill]).astype(int)
        return selected[:state.b_probe]


class ActiveProbe(Policy):
    """Active probing rule with three tunable modes.

    The probe score combines three signals computed from gateway-side state:

    - ``risk``: belief uncertainty plus near-threshold hazard.
    - ``debt``: probe-channel service debt to avoid permanently un-probing loops.
    - ``staleness``: how long ago the gateway last received metadata for that loop.

    The ``mode`` argument selects a documented preset. ``custom`` exposes raw
    weights for ablation studies.
    """

    name = "active_probe"

    PRESETS = {
        "safety_first": {"w_risk": 1.00, "w_debt": 0.50, "w_stale": 0.80},
        "balanced":      {"w_risk": 0.70, "w_debt": 0.55, "w_stale": 0.80},
        "debt_first":    {"w_risk": 0.40, "w_debt": 1.00, "w_stale": 0.70},
    }

    def __init__(self, mode: str = "balanced", *, w_risk: float | None = None,
                 w_debt: float | None = None, w_stale: float | None = None):
        if mode == "custom":
            assert None not in (w_risk, w_debt, w_stale), "custom mode needs explicit weights"
        else:
            preset = self.PRESETS[mode]
            w_risk = preset["w_risk"] if w_risk is None else w_risk
            w_debt = preset["w_debt"] if w_debt is None else w_debt
            w_stale = preset["w_stale"] if w_stale is None else w_stale
        self.mode = mode
        self.w_risk = float(w_risk)
        self.w_debt = float(w_debt)
        self.w_stale = float(w_stale)

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        risk = belief_uncertainty(state)
        debt = state.fairness_debt("probe")
        stale = state.last_metadata_age.astype(float)
        score = (
            self.w_risk * safety.normalize_unit(risk)
            + self.w_debt * debt
            + self.w_stale * safety.normalize_unit(stale)
        )
        return topk(score, min(state.b_probe, state.n))


# ---------------- payload rule ----------------------------------------------

class DebtAwarePayload:
    name = "debt_aware_payload"

    def __init__(self, V: float = 1.0, w_residual: float = 0.10, w_probed_danger: float = 2.0):
        self.V = V
        self.w_residual = w_residual
        self.w_probed_danger = w_probed_danger  # bonus for probed + near-threshold

    def select(self, state: SchedulerState, probe_set: np.ndarray, metadata: np.ndarray, mask: np.ndarray) -> np.ndarray:
        values = metadata_safety_value(state, metadata, mask)
        debt = state.fairness_debt("payload")
        reliability = np.clip(predict_success_vec(state.pi_bad, state.channel), 0.25, 1.0)
        score = reliability * self.V * safety.normalize_unit(values) + debt
        residual = safety.normalize_unit(((metadata - state.xh) / safety.RANGE) ** 2)
        score = score - self.w_residual * residual
        # KEY FIX: strongly prioritize probed sensors that reveal danger (near/beyond threshold)
        # This ensures probe information is actually USED for payload decision
        danger = safety.margin_to_safety(metadata)  # 1.0 = at threshold, 0.0 = safe
        probed_danger_bonus = mask.astype(float) * danger * self.w_probed_danger
        score = score + probed_danger_bonus
        return topk(score, min(state.b_payload, state.n))


class ConstrainedAdaptiveGreedyPayload:
    """Adaptive greedy batch selector with feasibility checks and a safety shield."""

    name = "constrained_adaptive_greedy_payload"

    def __init__(self, V: float = 1.0, w_debt: float = 0.75, w_age: float = 0.20,
                 diversity_penalty: float = 0.10, cooldown: int = 0,
                 shield_pvio: float = 0.35, shield_margin: float = 0.85):
        self.V = float(V)
        self.w_debt = float(w_debt)
        self.w_age = float(w_age)
        self.diversity_penalty = float(diversity_penalty)
        self.cooldown = int(cooldown)
        self.shield_pvio = float(shield_pvio)
        self.shield_margin = float(shield_margin)

    def _base_score(self, state: SchedulerState, metadata: np.ndarray, mask: np.ndarray) -> np.ndarray:
        values = metadata_safety_value(state, metadata, mask)
        debt = state.fairness_debt("payload")
        age = safety.normalize_unit(state.age.astype(float))
        reliability = np.clip(predict_success_vec(state.pi_bad, state.channel), 0.25, 1.0)
        return reliability * self.V * safety.normalize_unit(values) + self.w_debt * debt + self.w_age * age

    def _shield_candidates(self, state: SchedulerState, metadata: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mu, var = state.ar.forecast_stats(state.xh, state.age, h=4)
        proxy = np.where(mask, metadata, mu)
        pvio = state.ar.empirical_safety_prob(mu, var, safe_min=safety.SAFE_MIN, safe_max=safety.SAFE_MAX)
        margin_risk = safety.margin_to_safety(proxy)
        return np.flatnonzero((pvio >= self.shield_pvio) | (margin_risk >= self.shield_margin))

    def select(self, state: SchedulerState, probe_set: np.ndarray, metadata: np.ndarray, mask: np.ndarray) -> np.ndarray:
        k = min(state.b_payload, state.n)
        if k <= 0:
            return np.empty(0, dtype=int)
        score = self._base_score(state, metadata, mask)
        last_served = state.extras.setdefault("cag_last_served", np.full(state.n, -10**9, dtype=int))
        slot = int(state.extras.setdefault("cag_slot", 0))
        feasible = np.ones(state.n, dtype=bool)
        if self.cooldown > 0:
            feasible &= (slot - last_served) >= self.cooldown

        selected: list[int] = []
        remaining = set(np.flatnonzero(feasible).tolist())
        while remaining and len(selected) < k:
            best_i = None
            best_gain = -np.inf
            for i in remaining:
                similarity = float(np.mean(np.abs(metadata[i] - metadata[selected]) <= 0.5)) if selected else 0.0
                gain = float(score[i] - self.diversity_penalty * similarity)
                if gain > best_gain or (np.isclose(gain, best_gain) and (best_i is None or i < best_i)):
                    best_i = i
                    best_gain = gain
            selected.append(int(best_i))
            remaining.remove(int(best_i))

        shield = [int(i) for i in self._shield_candidates(state, metadata, mask) if feasible[i]]
        for i in sorted(shield, key=lambda j: score[j], reverse=True):
            if i in selected:
                continue
            if len(selected) < k:
                selected.append(i)
                continue
            replaceable = [j for j in selected if j not in shield]
            if not replaceable:
                break
            victim = min(replaceable, key=lambda j: score[j])
            selected[selected.index(victim)] = i

        out = np.asarray(selected[:k], dtype=int)
        last_served[out] = slot
        state.extras["cag_slot"] = slot + 1
        state.extras["cag_last_score"] = score.copy()
        state.extras["cag_last_shield"] = np.asarray(shield, dtype=int)
        return out


# ---------------- top-level two-stage policy --------------------------------

@dataclass
class TwoStagePolicy:
    name: str
    probe_policy: Policy
    payload_policy: DebtAwarePayload

    def step(self, state: SchedulerState, x_true: np.ndarray, *, metadata_noise: float, metadata_loss: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Stage 1: probe
        probe_set = self.probe_policy.select_probe(state)
        # Reveal metadata for probed loops, with noise and channel loss.
        metadata = state.xh.copy()
        mask = np.zeros(state.n, dtype=bool)
        for i in probe_set:
            if state.rng.random() < (1.0 - metadata_loss):
                noise = state.rng.normal(0.0, metadata_noise) if metadata_noise > 0 else 0.0
                metadata[i] = x_true[i] + noise
                mask[i] = True
                state.last_metadata[i] = metadata[i]
                state.last_metadata_age[i] = 0
        state.last_metadata_age += 1
        # Stage 2: payload
        payload_set = self.payload_policy.select(state, probe_set, metadata, mask)
        return probe_set, payload_set, metadata, mask


# ---------------- factory ---------------------------------------------------

def build(name: str, **kwargs) -> TwoStagePolicy:
    if name in {"cag", "constrained_adaptive_greedy"}:
        payload = ConstrainedAdaptiveGreedyPayload(
            V=kwargs.get("V", 1.0),
            cooldown=kwargs.get("cooldown", 0),
            shield_pvio=kwargs.get("shield_pvio", 0.35),
            shield_margin=kwargs.get("shield_margin", 0.85),
        )
    else:
        payload = DebtAwarePayload(V=kwargs.get("V", 1.0))
    if name == "active":
        probe = ActiveProbe(mode=kwargs.get("mode", "balanced"))
        label = f"two_stage::active_{kwargs.get('mode', 'balanced')}"
    elif name in {"cag", "constrained_adaptive_greedy"}:
        probe = ActiveProbe(mode=kwargs.get("mode", "safety_first"))
        label = "two_stage::constrained_adaptive_greedy"
    elif name == "active_safety":
        probe = ActiveProbe(mode="safety_first")
        label = "two_stage::active_safety_first"
    elif name == "active_balanced":
        probe = ActiveProbe(mode="balanced")
        label = "two_stage::active_balanced"
    elif name == "active_debt":
        probe = ActiveProbe(mode="debt_first")
        label = "two_stage::active_debt_first"
    elif name == "random":
        probe = RandomProbe()
        label = "two_stage::random"
    elif name == "round_robin":
        probe = RoundRobinProbe()
        label = "two_stage::round_robin"
    elif name == "max_aoi":
        probe = MaxAoIProbe()
        label = "two_stage::max_aoi"
    elif name in {"aoii", "aoii_probe"}:
        probe = AoIIProbe(
            w_vio=kwargs.get("w_vio", 1.0),
            h=kwargs.get("h", 4),
        )
        label = "two_stage::aoii"
    elif name in {"vou", "vou_probe"}:
        probe = VoUProbe(
            lambda_safety=kwargs.get("lambda_safety", 6.0),
            h=kwargs.get("h", 4),
        )
        label = "two_stage::vou"
    elif name in {"urgency", "urgency_probe"}:
        probe = UrgencyProbe(
            w_stale=kwargs.get("w_stale", 0.05),
            h=kwargs.get("h", 4),
        )
        label = "two_stage::urgency"
    elif name in {"whittle_voi", "whittle_voi_probe"}:
        probe = WhittleVoIProbe(
            lambda_safety=kwargs.get("lambda_safety", 6.0),
            deg=kwargs.get("deg", 5),
            metadata_noise_std=kwargs.get("metadata_noise_std", 0.0),
        )
        label = "two_stage::whittle_voi"
    elif name in {"whittle_vou_shield", "voi_shield"}:
        probe = WhittleVoUShield(
            lambda_safety=kwargs.get("lambda_safety", 15.0),
            shield_threshold=kwargs.get("shield_threshold", 0.3),
            shield_budget_fraction=kwargs.get("shield_budget_fraction", 0.5),
            deg=kwargs.get("deg", 5),
            metadata_noise_std=kwargs.get("metadata_noise_std", 0.0),
        )
        label = "two_stage::whittle_vou_shield"
    else:
        raise ValueError(f"unknown probe rule {name}")
    return TwoStagePolicy(name=label, probe_policy=probe, payload_policy=payload)
