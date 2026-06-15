"""AR(1) forecaster with empirical residual quantiles for safety calibration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AR1Model:
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

    def empirical_safety_prob(self, mu: np.ndarray, var: np.ndarray, *, safe_min: float, safe_max: float) -> np.ndarray:
        if self.residuals is None:
            std = np.sqrt(var) + 1e-9
            from math import erf, sqrt
            erf_vec = np.vectorize(erf)
            z_lo = (safe_min - mu) / (std * sqrt(2.0))
            z_hi = (safe_max - mu) / (std * sqrt(2.0))
            cdf_lo = 0.5 * (1.0 + erf_vec(z_lo))
            cdf_hi = 0.5 * (1.0 + erf_vec(z_hi))
            return np.clip(cdf_lo + (1.0 - cdf_hi), 0.0, 1.0)
        # Empirical residual mixture: scale residuals by sqrt(var) per loop.
        std = np.sqrt(var) + 1e-9
        # Sample empirical residuals once per call, then re-use for each loop.
        rng = np.random.default_rng(2026)
        sample = rng.choice(self.residuals.flatten(), size=512, replace=True)
        sample = sample / (np.std(sample) + 1e-9)
        # Vectorized over loops (behavior-identical to the per-loop version):
        # draws_{i,k} = mu_i + std_i * sample_k.
        draws = mu[:, None] + std[:, None] * sample[None, :]
        out = np.mean((draws < safe_min) | (draws > safe_max), axis=1)
        return np.clip(out, 0.0, 1.0)
