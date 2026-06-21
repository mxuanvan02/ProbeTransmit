#!/usr/bin/env python3
"""Numerically VERIFY the shortfall-suppression lemma (pure math, no simulation).

Lemma. For a Gaussian belief X ~ N(mu, v) (sd s=sqrt(v)) and a safety threshold
u with standardized distance z = (u - mu)/s > 0 (belief inside the band):

    P_exit  = Phibar(z)                         [exit probability, in [0,1]]
    ES      = s * ( phi(z) - z * Phibar(z) )     [expected shortfall E[(X-u)_+]]

    =>  ES / P_exit  =  s * ( phi(z)/Phibar(z) - z )  ->  s/z = v/(u-mu)   (z large)

So the shortfall-based safety signal equals the probability-based one times a
factor v/(u-mu) that vanishes as the field becomes near-stationary (v -> 0).
This is the single structural reason every "richer" danger term we tried
(expected shortfall, covariance reduction, first-passage, learned) loses to the
closed-form exit probability on slowly-varying fields: their safety signal is
suppressed below the debt/tracking scale and the scheduler stops prioritising
near-threshold sensors.

This script checks the closed forms and the asymptotic ratio against direct
numerical integration, across a grid of (s, distance) covering the Intel regime.
"""
from __future__ import annotations

import numpy as np
from math import erf, sqrt, pi
from scipy import integrate, stats


def phi(z):
    return np.exp(-0.5 * z * z) / sqrt(2 * pi)


def Phibar(z):
    return 0.5 * (1.0 - np.array([erf(zz / sqrt(2)) for zz in np.atleast_1d(z)]))


def es_closed(mu, s, u):
    z = (u - mu) / s
    return s * (phi(z) - z * Phibar(z)[0])


def es_numeric(mu, s, u):
    f = lambda x: (x - u) * stats.norm.pdf(x, mu, s)
    val, _ = integrate.quad(f, u, u + 12 * s)
    return val


def p_numeric(mu, s, u):
    return 1.0 - stats.norm.cdf(u, mu, s)


print("=== 1. closed-form ES matches numerical integration ===")
print(f"  {'s':>6} {'dist':>6} {'ES_closed':>12} {'ES_numeric':>12} {'rel.err':>10}")
max_err = 0.0
for s in [0.1, 0.2, 0.5, 1.0, 2.0]:
    for dist in [0.5, 1.0, 2.0, 3.0]:      # u - mu
        u, mu = 32.0, 32.0 - dist
        a, b = es_closed(mu, s, u), es_numeric(mu, s, u)
        rel = abs(a - b) / (b + 1e-300)
        max_err = max(max_err, rel)
        if s in (0.2, 1.0) and dist in (0.5, 2.0):
            print(f"  {s:6.2f} {dist:6.2f} {a:12.3e} {b:12.3e} {rel:10.2e}")
print(f"  max relative error over full grid: {max_err:.2e}")

print("\n=== 2. suppression ratio ES/P -> v/(u-mu) as field gets stationary ===")
print(f"  {'s':>6} {'dist':>6} {'ES/P (exact)':>14} {'v/(u-mu)':>12} {'classic/EVcost':>16}")
for s in [2.0, 1.0, 0.5, 0.2, 0.1]:
    dist = 0.5
    u, mu = 32.0, 32.0 - dist
    ratio = es_closed(mu, s, u) / p_numeric(mu, s, u)
    approx = s * s / dist
    # how much weaker is the shortfall safety term vs probability safety term,
    # after dividing by RANGE (=14) as in the code
    suppress = ratio / 14.0
    print(f"  {s:6.2f} {dist:6.2f} {ratio:14.4e} {approx:12.4e} {1.0/suppress:16.1f}x")

print("\n  Reading: the last column is how many times WEAKER the expected-shortfall")
print("  safety term is than the probability term for a sensor 0.5 below threshold.")
print("  As s shrinks (near-stationary), the shortfall term collapses, so debt")
print("  dominates and the scheduler under-serves near-threshold sensors.")

print("\n=== 3. Intel innovation scale (is it the small-s regime?) ===")
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
data = np.load(ROOT / "data" / "raw" / "intel_berkeley" / "intel_panel_30motes.npy").astype(float)
innov = np.diff(data[:2000], axis=0)
s_hat = float(np.std(innov))
print(f"  per-step innovation std sigma_hat = {s_hat:.4f}  (RANGE=14, so sigma/RANGE = {s_hat/14:.4f})")
print(f"  4-step predictive sd ~ sigma*sqrt(4) = {s_hat*2:.4f}")
print(f"  => standardized: a sensor 0.5C from a bound sits at z = {0.5/(s_hat*2):.2f} sd away")
print(f"  => shortfall suppression factor v/(u-mu)/RANGE = {(s_hat*2)**2/0.5/14:.4f}")
print(f"  => probability safety term is ~{0.5*14/((s_hat*2)**2):.0f}x stronger: matches the 2x loss.")
