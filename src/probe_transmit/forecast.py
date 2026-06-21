"""Forecasting models for safety-aware scheduling.

This module provides two forecasters that share a common interface
(``forecast_stats`` + ``empirical_safety_prob``) so policies can be swapped
without code changes, plus the predictive machinery that upgrades the value of
update (VoU) from *confirmation* of a current violation to *prediction* of a
future one:

  - ``AR1Model``         : the original mean-reverting AR(1) forecaster.
  - ``LocalLinearTrendModel`` : a level+damped-slope linear-Gaussian state-space
                           model (a.k.a. damped-trend / Holt) whose multi-step
                           predictive mean carries the drift ("momentum") of a
                           sensor toward a danger threshold, and whose variance
                           grows with the horizon. Estimated by a per-sensor
                           Kalman filter.

Both expose ``first_passage_prob`` and ``forecast_path`` so a policy can ask the
predictive question: *what is the probability the signal leaves the safe band at
ANY time within the horizon H?* (first-passage), rather than only *at exactly
step h?* (point-in-time). The first-passage probability uses the closed-form
hitting-time law of a drifted Brownian motion (Bachelier-Levy / inverse-Gaussian
boundary-crossing formula), which is an upper bound on the discrete-time crossing
probability and is therefore conservative for safety monitoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np

_SQRT2 = sqrt(2.0)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """Vectorized standard-normal CDF via erf (no SciPy dependency)."""
    z = np.asarray(z, dtype=float)
    # erf is scalar in math; vectorize once.
    return 0.5 * (1.0 + np.vectorize(erf)(z / _SQRT2))


def _one_sided_first_passage(
    gap: np.ndarray, drift: np.ndarray, diff_var_per_step: np.ndarray, horizon: int
) -> np.ndarray:
    """P(drifted BM starting ``gap`` below a barrier reaches it within ``horizon``).

    Continuous-time approximation: X_t = x0 + drift * t + sigma * W_t, barrier at
    distance ``gap`` >= 0 above x0 (so a positive drift pushes toward it). With
    a = gap, mu = drift, sigma^2 = diff_var_per_step, T = horizon, the hitting
    probability has the closed form (Bachelier-Levy):

        P = Phi((mu*T - a)/(sigma*sqrt(T)))
            + exp(2*mu*a / sigma^2) * Phi(-(mu*T + a)/(sigma*sqrt(T))).

    Numerically guarded; returns 0 when the barrier is effectively unreachable
    and the exponential term is clipped to avoid overflow.
    """
    gap = np.maximum(np.asarray(gap, dtype=float), 0.0)
    mu = np.asarray(drift, dtype=float)
    var = np.maximum(np.asarray(diff_var_per_step, dtype=float), 1e-12)
    sigma = np.sqrt(var)
    T = float(max(horizon, 1))
    sig_sqrtT = sigma * sqrt(T) + 1e-12

    z1 = (mu * T - gap) / sig_sqrtT
    term1 = _norm_cdf(z1)

    # exp(2*mu*gap/var) can overflow when drift pushes hard toward the barrier;
    # clip the exponent. The product with the (tiny) second CDF stays in [0,1].
    expo = np.clip(2.0 * mu * gap / var, -50.0, 50.0)
    z2 = -(mu * T + gap) / sig_sqrtT
    term2 = np.exp(expo) * _norm_cdf(z2)

    return np.clip(term1 + term2, 0.0, 1.0)


def first_passage_prob(
    mu0: np.ndarray,
    drift: np.ndarray,
    diff_var_per_step: np.ndarray,
    *,
    safe_min: float,
    safe_max: float,
    horizon: int,
) -> np.ndarray:
    """Probability the signal exits [safe_min, safe_max] within ``horizon`` steps.

    Combines the two one-sided barrier-crossing probabilities (toward safe_min
    and toward safe_max) with a union bound -- an upper bound on the true
    two-sided first-passage probability, hence conservative (safety-favouring).

    ``mu0``  : current best estimate per sensor (path start).
    ``drift``: per-step predictive drift (slope) per sensor.
    ``diff_var_per_step`` : per-step innovation variance per sensor.
    """
    mu0 = np.asarray(mu0, dtype=float)
    # Upper barrier: distance up to safe_max, drift as-is (positive drift -> toward it).
    p_hi = _one_sided_first_passage(safe_max - mu0, drift, diff_var_per_step, horizon)
    # Lower barrier: reflect coordinates so the barrier sits "above" at distance
    # (mu0 - safe_min) with drift sign flipped.
    p_lo = _one_sided_first_passage(mu0 - safe_min, -drift, diff_var_per_step, horizon)
    # Union bound on the two disjoint-ish crossing events.
    return np.clip(p_hi + p_lo, 0.0, 1.0)


@dataclass
class AR1Model:
    """Mean-reverting AR(1) forecaster (original model)."""

    alpha: np.ndarray
    beta: np.ndarray
    sigma: np.ndarray
    residuals: np.ndarray | None = None

    @classmethod
    def fit(cls, train: np.ndarray) -> "AR1Model":
        n = train.shape[1]
        alpha = np.zeros(n)
        beta = np.zeros(n)
        sigma = np.zeros(n)
        for j in range(n):
            x_prev = train[:-1, j]
            x_curr = train[1:, j]
            cov = np.cov(x_prev, x_curr, ddof=1)[0, 1]
            var = np.var(x_prev, ddof=1) + 1e-9
            alpha[j] = cov / var
            beta[j] = float(np.mean(x_curr - alpha[j] * x_prev))
            resid = x_curr - alpha[j] * x_prev - beta[j]
            sigma[j] = float(np.std(resid, ddof=1) + 1e-9)
        return cls(alpha=alpha, beta=beta, sigma=sigma)

    def set_empirical_residuals(self, residuals: np.ndarray) -> None:
        self.residuals = residuals

    def forecast_stats(self, xh: np.ndarray, ages: np.ndarray, h: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance after `min(ages, h)` steps with last value `xh`."""
        steps = np.minimum(ages.astype(int), h)
        mu = xh.copy()
        var = np.zeros_like(xh)
        for s in range(int(steps.max(initial=0))):
            active = steps > s
            mu[active] = self.alpha[active] * mu[active] + self.beta[active]
            var[active] = (self.alpha[active] ** 2) * var[active] + self.sigma[active] ** 2
        return mu, var

    def forecast_path(self, xh: np.ndarray, ages: np.ndarray, horizon: int = 4):
        """Return (mu_H, drift, diff_var_per_step) summarising the H-step forecast.

        For AR(1) the "drift" is the average per-step change of the predictive
        mean over the horizon (mean-reversion gives a typically small drift),
        and diff_var_per_step is the one-step innovation variance sigma^2.
        Lets AR(1) also be queried via first_passage_prob for a fair ablation.
        """
        mu_H, _ = self.forecast_stats(xh, np.full_like(ages, horizon), h=horizon)
        drift = (mu_H - xh) / float(max(horizon, 1))
        diff_var = self.sigma ** 2
        return mu_H, drift, diff_var

    def first_passage_prob(self, xh, ages, *, safe_min, safe_max, horizon=4):
        mu0, drift, diff_var = self.forecast_path(xh, ages, horizon)
        return first_passage_prob(
            xh, drift, diff_var, safe_min=safe_min, safe_max=safe_max, horizon=horizon
        )

    def first_passage_prob_mc(self, xh, ages, *, safe_min, safe_max, horizon=4,
                              n_mc=256, rng=None):
        """Bayes-optimal first-passage under the FITTED AR(1) + empirical-residual
        model: Monte-Carlo rollout of x_{t+1}=alpha*x_t+beta+eps with resampled
        empirical residuals; fraction of paths exiting [safe_min,safe_max] at ANY
        step within the horizon. This is the correct generative-model object (no
        Brownian/Gaussian boundary approximation). Vectorized over sensors.
        """
        xh = np.asarray(xh, dtype=float)
        N = xh.shape[0]
        if rng is None:
            rng = np.random.default_rng(7)
        if self.residuals is not None:
            flat = self.residuals.flatten()
            draw = lambda shape: rng.choice(flat, size=shape, replace=True)
        else:
            sig = self.sigma
            draw = lambda shape: rng.normal(0.0, 1.0, size=shape) * sig[None, :]
        paths = np.repeat(xh[None, :], n_mc, axis=0)
        hit = np.zeros((n_mc, N), dtype=bool)
        a = self.alpha[None, :]; b = self.beta[None, :]
        for _ in range(int(max(horizon, 1))):
            eps = draw((n_mc, N))
            paths = a * paths + b + eps
            hit |= (paths < safe_min) | (paths > safe_max)
        return hit.mean(axis=0)

    def empirical_safety_prob(self, mu: np.ndarray, var: np.ndarray, *, safe_min: float, safe_max: float) -> np.ndarray:
        if self.residuals is None:
            std = np.sqrt(var) + 1e-9
            z_lo = (safe_min - mu) / std
            z_hi = (safe_max - mu) / std
            cdf_lo = _norm_cdf(z_lo)
            cdf_hi = _norm_cdf(z_hi)
            return np.clip(cdf_lo + (1.0 - cdf_hi), 0.0, 1.0)
        std = np.sqrt(var) + 1e-9
        rng = np.random.default_rng(2026)
        sample = rng.choice(self.residuals.flatten(), size=512, replace=True)
        sample = sample / (np.std(sample) + 1e-9)
        draws = mu[:, None] + std[:, None] * sample[None, :]
        out = np.mean((draws < safe_min) | (draws > safe_max), axis=1)
        return np.clip(out, 0.0, 1.0)


@dataclass
class LocalLinearTrendModel:
    """Damped local-linear-trend (DLT / damped-Holt) linear-Gaussian state-space
    forecaster, estimated per sensor by a Kalman filter.

    Per-sensor latent state s_t = [level l_t, slope b_t]^T evolves as

        l_t = l_{t-1} + phi * b_{t-1} + eta_t,     eta_t ~ N(0, q_level)
        b_t = phi * b_{t-1}           + zeta_t,    zeta_t ~ N(0, q_slope)

    observation  y_t = l_t + eps_t,                eps_t ~ N(0, r_obs).

    The damping ``phi`` in (0, 1] geometrically discounts the slope so noise
    cannot drive an explosive (false-alarm) trend; phi=1 recovers the classical
    local linear trend, phi<1 a mean-reverting drift. The tau-step-ahead
    predictive mean is

        E[y_{t+tau}] = l_t + b_t * (phi + phi^2 + ... + phi^tau),

    which carries the *momentum* of a sensor toward a threshold -- the
    ingredient AR(1) lacks. Predictive variance grows with tau via the
    propagated state covariance plus observation noise.
    """

    # Per-sensor parameters (shape (N,)).
    phi: np.ndarray
    q_level: np.ndarray
    q_slope: np.ndarray
    r_obs: np.ndarray
    # Cached per-sensor steady-state filtered covariance of the state (2x2 each),
    # used to seed the predictive variance recursion. Shape (N, 2, 2).
    P_filt: np.ndarray
    residuals: np.ndarray | None = None

    @classmethod
    def fit(cls, train: np.ndarray, phi: float = 0.92) -> "LocalLinearTrendModel":
        """Estimate per-sensor noise variances by method-of-moments on the
        first/second differences, then run a short Kalman pass to cache the
        steady-state filtered covariance.

        - r_obs   ~ measurement noise, from the high-frequency component.
        - q_slope ~ slope innovation, from the variance of second differences.
        - q_level ~ level innovation, residual of first differences after slope.
        ``phi`` is the damping factor (default 0.92): strong enough to track real
        drift, damped enough to suppress noise-induced spurious trends.
        """
        T, n = train.shape
        d1 = np.diff(train, axis=0)            # first differences  ~ slope + noise
        d2 = np.diff(train, n=2, axis=0)       # second differences ~ slope innov + noise

        # Robust per-sensor scale estimates (MAD-based to resist outliers).
        def _var_mad(x):
            med = np.median(x, axis=0)
            mad = np.median(np.abs(x - med), axis=0) * 1.4826
            return (mad ** 2) + 1e-9

        var_d1 = _var_mad(d1)
        var_d2 = _var_mad(d2)
        # Var(d2) = q_slope + 2*q_level + 6*r_obs (for phi~1); Var(d1) = q_level + 2*r_obs.
        # Solve a conservative split: attribute most high-freq energy to r_obs.
        r_obs = np.maximum(0.25 * var_d1, 1e-6)
        q_level = np.maximum(var_d1 - 2.0 * r_obs, 1e-6)
        q_slope = np.maximum(0.1 * np.maximum(var_d2 - 2.0 * q_level - 6.0 * r_obs, 0.0), 1e-7)

        phi_arr = np.full(n, float(phi))

        # Run a short Kalman filter to a steady-state filtered covariance per sensor.
        P_filt = np.zeros((n, 2, 2))
        for j in range(n):
            F = np.array([[1.0, phi_arr[j]], [0.0, phi_arr[j]]])
            Q = np.array([[q_level[j], 0.0], [0.0, q_slope[j]]])
            H = np.array([[1.0, 0.0]])
            R = np.array([[r_obs[j]]])
            P = np.eye(2) * var_d1[j]
            x = np.array([train[0, j], 0.0])
            burn = min(T, 300)
            for t in range(1, burn):
                # Predict
                x = F @ x
                P = F @ P @ F.T + Q
                # Update
                S = H @ P @ H.T + R
                K = (P @ H.T) / S
                innov = train[t, j] - (H @ x)[0]
                x = x + (K.flatten() * innov)
                P = (np.eye(2) - K @ H) @ P
            P_filt[j] = P

        return cls(phi=phi_arr, q_level=q_level, q_slope=q_slope, r_obs=r_obs, P_filt=P_filt)

    def set_empirical_residuals(self, residuals: np.ndarray) -> None:
        self.residuals = residuals

    # -- predictive machinery -------------------------------------------------

    def _slope_from_recent(self, xh, last_metadata, ages):
        """Estimate current slope b_t per sensor from the held estimate vs the
        last received metadata, damped by phi^age (older info -> weaker slope).
        A cheap, robust surrogate for the full filtered slope that needs no
        per-step state carried in SchedulerState.
        """
        age = np.maximum(ages.astype(float), 1.0)
        raw_slope = (xh - last_metadata) / age
        return raw_slope * (self.phi ** np.minimum(age, 10.0))

    def forecast_path(self, xh, ages, horizon=4, last_metadata=None):
        """Return (mu_H, drift, diff_var_per_step).

        mu_H  : H-step predictive mean WITH trend (level + damped slope sum).
        drift : per-step predictive drift = (mu_H - xh)/H.
        diff_var_per_step : average per-step predictive innovation variance.
        """
        if last_metadata is None:
            last_metadata = xh
        slope = self._slope_from_recent(xh, last_metadata, ages)
        # Damped geometric sum: sum_{k=1..H} phi^k.
        phi = self.phi
        H = int(max(horizon, 1))
        geo = np.zeros_like(xh)
        pk = np.ones_like(xh)
        for _ in range(H):
            pk = pk * phi
            geo = geo + pk
        mu_H = xh + slope * geo
        drift = (mu_H - xh) / float(H)
        # Predictive variance per step: level innovation + propagated slope
        # innovation + observation noise, averaged over the horizon.
        diff_var = self.q_level + self.q_slope * (geo / H) + self.r_obs
        return mu_H, drift, diff_var

    def forecast_stats(self, xh, ages, h: int = 4, last_metadata=None):
        """Predictive mean and (cumulative) variance at horizon h, trend-aware."""
        mu_H, drift, diff_var = self.forecast_path(xh, ages, h, last_metadata)
        H = int(max(h, 1))
        var = diff_var * H  # cumulative predictive variance over the horizon
        return mu_H, var

    def first_passage_prob(self, xh, ages, *, safe_min, safe_max, horizon=4, last_metadata=None):
        mu0 = xh
        _, drift, diff_var = self.forecast_path(xh, ages, horizon, last_metadata)
        return first_passage_prob(
            mu0, drift, diff_var, safe_min=safe_min, safe_max=safe_max, horizon=horizon
        )

    def empirical_safety_prob(self, mu, var, *, safe_min, safe_max):
        """Point-in-time exit probability at the horizon (Gaussian or empirical
        residual mixture), kept for interface compatibility / ablation."""
        if self.residuals is None:
            std = np.sqrt(var) + 1e-9
            cdf_lo = _norm_cdf((safe_min - mu) / std)
            cdf_hi = _norm_cdf((safe_max - mu) / std)
            return np.clip(cdf_lo + (1.0 - cdf_hi), 0.0, 1.0)
        std = np.sqrt(var) + 1e-9
        rng = np.random.default_rng(2026)
        sample = rng.choice(self.residuals.flatten(), size=512, replace=True)
        sample = sample / (np.std(sample) + 1e-9)
        draws = mu[:, None] + std[:, None] * sample[None, :]
        out = np.mean((draws < safe_min) | (draws > safe_max), axis=1)
        return np.clip(out, 0.0, 1.0)
