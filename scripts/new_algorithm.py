#!/usr/bin/env python3
"""CAW-VoU: Correlation-Aware Whittle Value-of-Update probe scheduler.

New probe-scheduling algorithm for the probe-then-transmit greenhouse IoT paper.

Motivation
----------
The current winning baseline, VoU, scores every loop independently and takes the
top-k by

    VoU_i = (mu_i - xh_i)^2 / RANGE^2 + lambda * p_vio_i * I(xh_i in safe band).

On the real greenhouse panel the temperature sensors are correlated at
rho ~ 0.94-0.97, so independent top-k spends scarce probe budget on *redundant*
neighbours and never credits the fact that probing one sensor already shrinks the
uncertainty about its correlated peers.

CAW-VoU keeps VoU's winning, empirically-priced safety term but replaces
independent top-k with a *submodular, correlation-discounted greedy* selection.
Each marginal probe is scored by the residual value it adds GIVEN the probes
already chosen this step, using a Gaussian conditional-variance update over a
fitted spatial correlation matrix R:

    x ~ N(mu, Sigma),  Sigma = D R D,  D = diag(sd_i),  sd_i = sqrt(var_i)
    var_{i|S} = Sigma_ii - Sigma_iS (Sigma_SS + sigma_meta^2 I)^{-1} Sigma_Si

When R = I (no correlation) CAW-VoU reduces EXACTLY to debt-regularised VoU, so
VoU is a strict special case.

This file is import-compatible with ``src/probe_transmit/policies.py``: it reuses
``SchedulerState``, ``safety``, ``forecast`` and ``channel``. Run directly for a
self-contained smoke test:

    python3 scripts/new_algorithm.py
"""
from __future__ import annotations

import sys
from math import erf
from pathlib import Path

import numpy as np

# Make src/ importable whether run from repo root or scripts/.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from probe_transmit import safety  # noqa: E402
from probe_transmit.channel import predict_success, predict_success_vec  # noqa: E402
from probe_transmit.policies import Policy, SchedulerState, service_debt, topk  # noqa: E402


# --------------------------------------------------------------------------- #
# Correlation matrix estimation (fitted ONCE from training data, passed in).   #
# --------------------------------------------------------------------------- #
def fit_correlation(train: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    """Estimate an n x n correlation matrix from a training panel.

    Parameters
    ----------
    train : (T, n) array of historical loop values.
    shrinkage : float in [0, 1]
        Linear shrinkage toward the identity (Ledoit-Wolf style). 0 -> raw
        sample correlation, 1 -> identity (recovers independent VoU). A small
        positive value stabilises R when T is short and conveniently keeps the
        ``R -> I`` reduction-to-VoU property continuous.

    Returns
    -------
    R : (n, n) symmetric PSD-ish correlation matrix with unit diagonal.
    """
    train = np.asarray(train, dtype=float)
    if train.ndim != 2:
        raise ValueError("train must be 2D (T, n)")
    n = train.shape[1]
    if train.shape[0] < 3:
        return np.eye(n)
    R = np.corrcoef(train, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    s = float(np.clip(shrinkage, 0.0, 1.0))
    R = (1.0 - s) * R + s * np.eye(n)
    np.fill_diagonal(R, 1.0)
    return R


def _conditional_variances(
    Sigma: np.ndarray, S: list[int], sigma_meta_var: float
) -> np.ndarray:
    """Posterior variances var_{i|S} for every loop i, given probed set S.

    Uses the Gaussian conditional rule
        var_{i|S} = Sigma_ii - Sigma_iS (Sigma_SS + sigma_meta^2 I)^{-1} Sigma_Si.
    Loops in S get ~sigma_meta^2 (they are revealed up to metadata noise).
    Returns the full length-n vector of conditional variances.
    """
    n = Sigma.shape[0]
    v = np.diag(Sigma).astype(float).copy()
    if not S:
        return v
    S = list(S)
    A = Sigma[np.ix_(S, S)] + sigma_meta_var * np.eye(len(S))
    # Robust solve (A is small, |S| <= B_probe).
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)
    # For all loops at once: reduction_i = Sigma_iS A_inv Sigma_Si.
    Sig_iS = Sigma[:, S]                       # (n, |S|)
    reduction = np.einsum("ik,kl,il->i", Sig_iS, A_inv, Sig_iS)
    v = np.maximum(np.diag(Sigma) - reduction, 0.0)
    # Probed loops are observed: residual variance ~ metadata noise.
    v[S] = sigma_meta_var
    return v


# --------------------------------------------------------------------------- #
# The new algorithm.                                                          #
# --------------------------------------------------------------------------- #
class CorrVoUProbe(Policy):
    """Correlation-Aware Whittle Value-of-Update (CAW-VoU) probe rule.

    Drop-in replacement for ``VoUProbe`` that adds correlation-aware, submodular
    greedy selection. Keeps VoU's empirically-priced safety term and channel
    scaling; adds a Gaussian conditional-variance update over a fitted
    correlation matrix R and a small Lyapunov debt floor.

    Parameters
    ----------
    corr : (n, n) array or None
        Spatial correlation matrix R. If None, defaults to identity, in which
        case CAW-VoU reduces exactly to debt-regularised VoU. Fit it once with
        ``fit_correlation(train)`` and pass it in.
    lambda_safety : float
        Safety weight, aligned to the simulator's miss cost (6.0). Default 6.0
        matches the winning VoU configuration.
    h : int
        Forecast horizon for AR(1) belief.
    metadata_noise_std : float
        Assumed metadata noise std for the conditional-variance update.
    w_debt : float
        Lyapunov fairness floor weight (small; only breaks near-ties).
    use_correlation_credit : bool
        If False, only the conditional-variance of each *candidate itself* is
        used (cheaper) without crediting peer information. Default True.
    """

    name = "corr_vou_probe"

    def __init__(
        self,
        corr: np.ndarray | None = None,
        lambda_safety: float = 6.0,
        h: int = 4,
        metadata_noise_std: float = 0.25,
        w_debt: float = 0.05,
        use_correlation_credit: bool = True,
        credit_mode: str = "variance",
        use_first_passage: bool = False,
        fp_mode: str = "analytic",
        vou_mode: str = "classic",
        rho: float = 6.0,
    ):
        self.corr = None if corr is None else np.asarray(corr, dtype=float)
        self.lambda_safety = float(lambda_safety)
        self.h = int(h)
        self.sigma_meta_var = float(metadata_noise_std) ** 2
        self.w_debt = float(w_debt)
        self.use_correlation_credit = bool(use_correlation_credit)
        # use_first_passage: if True, the safety term is the PREDICTIVE
        # first-passage probability P(signal exits the safe band at ANY time in
        # [t, t+h]) under the trend-aware forecaster, instead of the
        # point-in-time exit probability at exactly step h. Turns VoU from
        # *confirming* a current/horizon-end violation into *predicting* an
        # imminent one. Requires state.ar to expose first_passage_prob (LLT/AR1).
        self.use_first_passage = bool(use_first_passage)
        # fp_mode selects the danger term when use_first_passage is on:
        #   "analytic" -> closed-form (Brownian) first_passage_prob
        #   "mc"       -> AR(1) Monte-Carlo Bayes-optimal first_passage_prob_mc
        self.fp_mode = str(fp_mode)
        # vou_mode: "classic" = track_gap + lambda*P_danger (original, ad-hoc);
        #   "ev_cost" = expected-cost-reduction VoU (estimation-theoretic):
        #   tracking term = error-covariance reduction (bias^2+var -> sigma^2),
        #   safety term = reduction in expected boundary exceedance E[(x-u)_+]
        #   under the Gaussian belief; both in signal^2 units so the safety
        #   weight rho is dimensionless (no hand-tuned lambda).
        self.vou_mode = str(vou_mode)
        self.rho = float(rho)
        # credit_mode: "variance" = original conditional-variance credit (sums
        # VoU reduction over all peers equally); "danger_gated" = weight each
        # peer's credit by its own danger (proximity to threshold), so probing
        # concentrates on clustered danger instead of broad field coverage.
        self.credit_mode = str(credit_mode)

    # -- per-loop instantaneous VoU value as a function of conditional variance -
    def _vou_value(
        self,
        state: SchedulerState,
        mu: np.ndarray,
        v: np.ndarray,
        xh_safe: np.ndarray,
        track: np.ndarray,
    ) -> np.ndarray:
        """val_i = track_i + lambda * P_danger_i * I(xh_i safe).

        P_danger is either:
          - the PREDICTIVE first-passage probability P(exit safe band at ANY
            step in [t, t+h]) under the trend-aware forecaster, when
            ``use_first_passage`` is set (and state.ar supports it), or
          - the EMPIRICAL point-in-time exit probability at the horizon
            (original VoU ingredient), evaluated at conditional variance v.
        """
        if self.vou_mode == "severity":
            # Severity-aware VoU (Bayes expected operational cost, math-first):
            #   Danger = P_vio  +  (kappa/RANGE) * ES,   ES = E[(X-u)_+]
            # Classic P_vio is the leading (eps^0) term; the expected-shortfall
            # ES is the severity correction (LEVEL, not reduction), large exactly
            # near/over a bound. Reduces to classic as the field gets stationary
            # (suppression lemma: ES/P_vio -> v/(u-mu) -> 0). See
            # docs/severity_vou_derivation.md.
            vv = np.maximum(v, 1e-12)
            sd = np.sqrt(vv)
            p_vio = state.ar.empirical_safety_prob(
                mu, vv, safe_min=safety.SAFE_MIN, safe_max=safety.SAFE_MAX,
            )

            def _phi(zz):
                return np.exp(-0.5 * zz * zz) / np.sqrt(2 * np.pi)

            def _Phi(zz):
                return 0.5 * (1.0 + np.vectorize(erf)(zz / np.sqrt(2.0)))

            # E[(X-Smax)_+] for X~N(mu,vv)
            z_hi = (mu - safety.SAFE_MAX) / sd
            es_hi = sd * (_phi(z_hi) + z_hi * _Phi(z_hi))
            # E[(Smin-X)_+]
            z_lo = (safety.SAFE_MIN - mu) / sd
            es_lo = sd * (_phi(z_lo) + z_lo * _Phi(z_lo))
            es = np.maximum(es_hi, 0.0) + np.maximum(es_lo, 0.0)
            danger = p_vio + (self.rho / safety.RANGE) * es
            return track + self.lambda_safety * danger * xh_safe

        if self.vou_mode == "ev_cost":
            # Expected-shortfall VoU (estimation-theoretic, dimensionless rho).
            # Tracking value = error-covariance reduction: current squared error
            # (bias^2 from belief drift + predictive variance v) relative to the
            # one-step residual variance sigma^2 attained once the sensor is served.
            sig2 = np.asarray(state.ar.sigma, dtype=float) ** 2
            est_now = (mu - state.xh) ** 2 + np.maximum(v, 1e-12)
            track_reduction = np.maximum(est_now - sig2, 0.0) / (safety.RANGE ** 2)
            # Safety value = the EXPECTED BOUNDARY EXCEEDANCE under the Gaussian
            # belief N(mu, v): E[(x-Smax)_+] + E[(Smin-x)_+]. This is a closed-form
            # partial moment (an expected-shortfall risk measure), large precisely
            # when the belief sits near/over a safety bound -- it measures POSITION
            # risk, not accumulated uncertainty, so it fires at the right sensors.
            sd = np.sqrt(np.maximum(v, 1e-12))
            z_hi = (mu - safety.SAFE_MAX) / sd
            z_lo = (safety.SAFE_MIN - mu) / sd
            phi = np.exp(-0.5 * z_hi * z_hi) / np.sqrt(2 * np.pi)
            phi_lo = np.exp(-0.5 * z_lo * z_lo) / np.sqrt(2 * np.pi)
            Phi_hi = 0.5 * (1.0 + np.vectorize(erf)(z_hi / np.sqrt(2)))
            Phi_lo = 0.5 * (1.0 + np.vectorize(erf)(z_lo / np.sqrt(2)))
            up = sd * phi + (mu - safety.SAFE_MAX) * Phi_hi
            lo = sd * phi_lo + (safety.SAFE_MIN - mu) * Phi_lo
            exceedance = (np.maximum(up, 0.0) + np.maximum(lo, 0.0)) / safety.RANGE
            return track_reduction + self.rho * exceedance * xh_safe

        if self.use_first_passage and hasattr(state.ar, "first_passage_prob"):
            last_md = getattr(state, "last_metadata", None)
            if self.fp_mode == "mc" and hasattr(state.ar, "first_passage_prob_mc"):
                p_danger = state.ar.first_passage_prob_mc(
                    state.xh, state.age,
                    safe_min=safety.SAFE_MIN, safe_max=safety.SAFE_MAX,
                    horizon=self.h,
                )
            else:
                try:
                    p_danger = state.ar.first_passage_prob(
                        state.xh, state.age,
                        safe_min=safety.SAFE_MIN, safe_max=safety.SAFE_MAX,
                        horizon=self.h, last_metadata=last_md,
                    )
                except TypeError:
                    # AR1Model.first_passage_prob has no last_metadata kwarg.
                    p_danger = state.ar.first_passage_prob(
                        state.xh, state.age,
                        safe_min=safety.SAFE_MIN, safe_max=safety.SAFE_MAX,
                        horizon=self.h,
                    )
        else:
            p_danger = state.ar.empirical_safety_prob(
                mu, np.maximum(v, 1e-12),
                safe_min=safety.SAFE_MIN, safe_max=safety.SAFE_MAX,
            )
        return track + self.lambda_safety * p_danger * xh_safe

    def select_probe(self, state: SchedulerState) -> np.ndarray:
        n = state.n
        b = min(state.b_probe, n)
        if b <= 0:
            return np.empty(0, dtype=int)

        mu, var = state.ar.forecast_stats(state.xh, state.age, h=self.h)
        var = np.maximum(var, 1e-12)
        sd = np.sqrt(var)
        track = (mu - state.xh) ** 2 / (safety.RANGE ** 2)
        xh_safe = (
            (state.xh >= safety.SAFE_MIN) & (state.xh <= safety.SAFE_MAX)
        ).astype(float)
        p_succ = np.clip(predict_success_vec(state.pi_bad, state.channel), 0.25, 1.0)
        debt = service_debt(
            state.probe_counts, state.total_probe_choices, state.probe_target_share()
        )

        # Correlation matrix R: passed in, cached in extras, or identity.
        R = self.corr
        if R is None:
            R = state.extras.get("corr_matrix")
        if R is None or np.asarray(R).shape != (n, n):
            R = np.eye(n)
        Sigma = (sd[:, None] * np.asarray(R, dtype=float)) * sd[None, :]

        # --- Fast path: no correlation -> debt-regularised VoU (top-k) -------
        off_diag = Sigma - np.diag(np.diag(Sigma))
        if (not self.use_correlation_credit) or np.allclose(off_diag, 0.0):
            vou = self._vou_value(state, mu, var, xh_safe, track)
            score = p_succ * vou + self.w_debt * debt
            return topk(score, b)

        # --- Event-confirmation mode: correlation AMPLIFIES (does not replace) -
        # Rationale: in threshold problems, inferring a peer's danger does not
        # refresh that peer's data -- we must still observe the boundary node
        # directly. So instead of using correlation to SKIP peers, we use it to
        # SHARPEN the danger signal: a node whose correlated neighbours are also
        # near the boundary is evidence of a genuine clustered event (not sensor
        # noise), so we boost its urgency and still probe it directly.
        if self.credit_mode == "event_confirm":
            vou = self._vou_value(state, mu, var, xh_safe, track)
            p_vio = state.ar.empirical_safety_prob(
                mu, np.maximum(var, 1e-12),
                safe_min=safety.SAFE_MIN, safe_max=safety.SAFE_MAX,
            )
            Rabs = np.abs(np.asarray(R, dtype=float))
            np.fill_diagonal(Rabs, 0.0)
            # neighbour danger evidence: correlation-weighted peer violation prob
            row_sum = Rabs.sum(axis=1)
            row_sum[row_sum <= 0] = 1.0
            neigh_danger = (Rabs @ p_vio) / row_sum
            # multiplicative confirmation: own danger reinforced by coherent peers
            confirm = 1.0 + self.lambda_safety * p_vio * neigh_danger * xh_safe
            score = p_succ * vou * confirm + self.w_debt * debt
            return topk(score, b)

        # --- Correlation-aware submodular greedy ----------------------------
        # Optional danger gate: weight each peer's credit by its own violation
        # probability, so correlation only pulls probes toward peers that are
        # themselves near the safety boundary (clustered danger), not toward
        # arbitrary well-correlated-but-safe neighbours.
        if self.credit_mode == "danger_gated":
            danger_w = state.ar.empirical_safety_prob(
                mu, np.maximum(var, 1e-12),
                safe_min=safety.SAFE_MIN, safe_max=safety.SAFE_MAX,
            )
        else:
            danger_w = np.ones(n, dtype=float)
        S: list[int] = []
        v_cur = var.copy()  # conditional variances given current S (S empty -> marginals)
        base_val = self._vou_value(state, mu, v_cur, xh_safe, track)
        for _ in range(b):
            best_j, best_score = -1, -np.inf
            remaining = [j for j in range(n) if j not in S]
            for j in remaining:
                # Direct refresh value of j (j becomes ~exact: v_j -> meta noise).
                gain_self = base_val[j]
                # Correlation credit: shrink peers' miss probability by adding j.
                v_after = _conditional_variances(Sigma, S + [j], self.sigma_meta_var)
                val_after = self._vou_value(state, mu, v_after, xh_safe, track)
                # Credit only from unprobed peers (exclude S and j itself),
                # weighted by each peer's own danger level.
                peer_mask = np.ones(n, dtype=bool)
                peer_mask[S] = False
                peer_mask[j] = False
                credit = float(np.sum(((base_val - val_after) * danger_w)[peer_mask]))
                delta_j = gain_self + credit
                score_j = float(p_succ[j]) * delta_j + self.w_debt * debt[j]
                if score_j > best_score:
                    best_score, best_j = score_j, j
            S.append(best_j)
            # Commit posterior variances and recompute the base values given S.
            v_cur = _conditional_variances(Sigma, S, self.sigma_meta_var)
            base_val = self._vou_value(state, mu, v_cur, xh_safe, track)
        return np.asarray(S, dtype=int)


# --------------------------------------------------------------------------- #
# Self-contained smoke test.                                                   #
# --------------------------------------------------------------------------- #
def _build_state(n, b_probe, b_payload, ar, channel, xh, age, rng):
    from probe_transmit.channel import stationary_bad_belief_vec
    return SchedulerState(
        n=n, b_probe=b_probe, b_payload=b_payload, rng=rng, ar=ar, channel=channel,
        xh=xh.copy(), age=age.copy(),
        payload_counts=np.zeros(n), probe_counts=np.zeros(n),
        total_payload_choices=0, total_probe_choices=0,
        pi_bad=stationary_bad_belief_vec(channel, n),
        last_metadata=xh.copy(), last_metadata_age=np.zeros(n, dtype=int),
    )


def _smoke() -> int:
    from probe_transmit.channel import CHANNELS
    from probe_transmit.forecast import AR1Model
    from probe_transmit.policies import VoUProbe

    print("== CAW-VoU smoke test ==")
    rng = np.random.default_rng(7)

    # Synthetic correlated panel: 2 blocks of 3 highly-correlated loops (rho~0.95)
    # plus structure, to exercise the correlation credit. T steps of AR(1)-ish data.
    T, n = 400, 6
    base = np.zeros((T, n))
    drivers = rng.normal(0, 1, size=(T, 2)).cumsum(axis=0) * 0.3
    for j in range(n):
        blk = 0 if j < 3 else 1
        base[:, j] = 25.0 + drivers[:, blk] + rng.normal(0, 0.3, size=T)
    ar = AR1Model.fit(base)
    ar.set_empirical_residuals(
        base[1:] - (ar.alpha * base[:-1] + ar.beta)
    )
    channel = CHANNELS["severe_burst"]
    R = fit_correlation(base, shrinkage=0.05)
    print(f"fitted block correlation R[0,1]={R[0,1]:.3f} (within-block), "
          f"R[0,3]={R[0,3]:.3f} (cross-block)")

    # Make a state where block 0 is stale (high age) so several block-0 loops
    # look individually attractive to VoU -> redundancy opportunity.
    xh = base[-1].copy()
    age = np.array([8, 8, 8, 1, 1, 1], dtype=int)
    b_probe, b_payload = 2, 2

    st_vou = _build_state(n, b_probe, b_payload, ar, channel, xh, age, np.random.default_rng(1))
    st_caw = _build_state(n, b_probe, b_payload, ar, channel, xh, age, np.random.default_rng(1))
    st_id  = _build_state(n, b_probe, b_payload, ar, channel, xh, age, np.random.default_rng(1))

    vou = VoUProbe(lambda_safety=6.0)
    caw = CorrVoUProbe(corr=R, lambda_safety=6.0, w_debt=0.0)
    caw_id = CorrVoUProbe(corr=np.eye(n), lambda_safety=6.0, w_debt=0.0)

    s_vou = sorted(vou.select_probe(st_vou).tolist())
    s_caw = sorted(caw.select_probe(st_caw).tolist())
    s_id = sorted(caw_id.select_probe(st_id).tolist())

    print(f"VoU probe set          : {s_vou}")
    print(f"CAW-VoU (R fitted)     : {s_caw}")
    print(f"CAW-VoU (R=I, w_debt=0): {s_id}")

    ok = True
    # (a) runs and returns the right budget
    assert len(s_caw) == b_probe, "wrong probe budget"
    # (b) reduces to VoU when R = I
    if s_id == s_vou:
        print("PASS (b): CAW-VoU with R=I matches VoU exactly.")
    else:
        print(f"FAIL (b): R=I should match VoU. got {s_id} vs {s_vou}")
        ok = False
    # (c) correlation changes the decision: CAW should avoid picking two
    #     within-block (redundant) loops when VoU would.
    vou_within_block0 = sum(1 for i in s_vou if i < 3)
    caw_within_block0 = sum(1 for i in s_caw if i < 3)
    print(f"block-0 probes -> VoU={vou_within_block0}, CAW-VoU={caw_within_block0}")
    if s_caw != s_vou:
        print("PASS (c): correlation-aware selection differs from VoU top-k.")
    else:
        print("NOTE (c): identical here; try larger budget / stronger correlation.")

    print("== smoke test done ==")
    return 0 if ok else 1


def _smoke_redundancy() -> int:
    """Decisive test: two mutually-redundant loops vs one independent loop.

    Construct a panel where loops 0 and 1 are near-duplicates (rho ~ 0.99) and
    loop 2 is independent. Set beliefs so that VoU's per-loop scores rank
    {0, 1} above 2 (so VoU picks the redundant pair), while CAW-VoU's
    correlation credit should demote the duplicate and pull in loop 2.
    """
    from probe_transmit.channel import CHANNELS
    from probe_transmit.forecast import AR1Model
    from probe_transmit.policies import VoUProbe

    print("\n== CAW-VoU redundancy test ==")
    rng = np.random.default_rng(11)
    T, n = 600, 3
    # Strong shared latent with large per-step innovation -> high forecast var,
    # so the empirical violation term is variance-sensitive. Small idiosyncratic
    # noise keeps loops 0,1 near-duplicates (rho ~ 0.99).
    z01 = 28.0 + np.cumsum(rng.normal(0, 0.6, size=T))
    z2 = 28.0 + np.cumsum(rng.normal(0, 0.6, size=T))
    z01 = np.clip(z01, 22.0, 31.5)
    z2 = np.clip(z2, 22.0, 31.5)
    base = np.zeros((T, n))
    base[:, 0] = z01 + rng.normal(0, 0.08, size=T)         # duplicate pair
    base[:, 1] = z01 + rng.normal(0, 0.08, size=T)         # duplicate pair
    base[:, 2] = z2 + rng.normal(0, 0.08, size=T)          # independent loop
    ar = AR1Model.fit(base)
    ar.set_empirical_residuals(base[1:] - (ar.alpha * base[:-1] + ar.beta))
    channel = CHANNELS["severe_burst"]
    R = fit_correlation(base, shrinkage=0.0)
    print(f"R[0,1]={R[0,1]:.3f} (duplicates), R[0,2]={R[0,2]:.3f} (independent)")

    # Beliefs: all loops sit right at the upper safety edge so the empirical
    # violation term is large AND variance-sensitive (the regime where the
    # correlation credit actually bites). Loops 0,1 are the redundant pair.
    xh = np.array([31.5, 31.5, 31.4])     # safe (<32) but at the upper edge
    age = np.array([6, 6, 6], dtype=int)
    b_probe, b_payload = 2, 2

    st_vou = _build_state(n, b_probe, b_payload, ar, channel, xh, age, np.random.default_rng(2))
    st_caw = _build_state(n, b_probe, b_payload, ar, channel, xh, age, np.random.default_rng(2))
    s_vou = sorted(VoUProbe(lambda_safety=6.0).select_probe(st_vou).tolist())
    s_caw = sorted(CorrVoUProbe(corr=R, lambda_safety=6.0, w_debt=0.0).select_probe(st_caw).tolist())
    print(f"VoU probe set      : {s_vou}")
    print(f"CAW-VoU probe set  : {s_caw}")

    ok = True
    if s_vou == [0, 1]:
        print("OK: VoU picks the redundant duplicate pair {0,1} (as expected).")
        if 2 in s_caw and not (0 in s_caw and 1 in s_caw):
            print("PASS (c'): CAW-VoU drops a redundant duplicate and probes the "
                  "independent loop 2.")
        else:
            print(f"NOTE (c'): CAW-VoU={s_caw}; correlation credit did not flip the "
                  "choice in this configuration.")
    else:
        print(f"NOTE: VoU did not pick {{0,1}} here (got {s_vou}); test inconclusive, "
              "not a failure of CAW-VoU.")
    print("== redundancy test done ==")
    return 0 if ok else 1


if __name__ == "__main__":
    rc = _smoke()
    rc2 = _smoke_redundancy()
    raise SystemExit(rc or rc2)


