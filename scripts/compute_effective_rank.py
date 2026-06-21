#!/usr/bin/env python3
"""Compute effective rank consistently for all three panels (derive-then-verify).

Single, citable definition: entropy-based effective rank (Roy & Vetterli, 2007).
For a correlation matrix R with eigenvalues lambda_i >= 0, let p_i = lambda_i / sum_j lambda_j.
Then  erank(R) = exp( - sum_i p_i log p_i ).

We replicate the manuscript's correlation pipeline exactly: fit a shrinkage
correlation (shrinkage=0.1) on the train split train_len = min(2000, 0.6 T),
matching scripts/robustness_multidataset.py. The same recipe is applied to all
three panels so the numbers are directly comparable.
Stdlib + numpy (numpy already required by the repo venv).
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANELS = {
    "Intel Berkeley": ROOT / "data/raw/intel_berkeley/intel_panel_30motes.npy",
    "Beijing air quality": ROOT / "data/raw/_candidates/beijing_prsa/beijing_temp_panel.npy",
    "KETI smart-building": ROOT / "data/raw/_candidates/keti_smartbuilding/keti_clean_panel.npy",
}


def fit_correlation(train, shrinkage=0.1):
    X = train - train.mean(0, keepdims=True)
    C = np.nan_to_num(np.corrcoef(X.T))
    return (1 - shrinkage) * C + shrinkage * np.eye(C.shape[0])


def entropy_effective_rank(R):
    w = np.clip(np.linalg.eigvalsh(R), 1e-12, None)
    p = w / w.sum()
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))


def main():
    print(f"{'dataset':22} {'N':>3} {'T':>7} {'train':>6} {'erank':>7} {'off-diag mean':>13}")
    for name, path in PANELS.items():
        a = np.load(path).astype(float)
        T, N = a.shape
        train_len = min(2000, int(0.6 * T))
        R = fit_correlation(a[:train_len], 0.1)
        er = entropy_effective_rank(R)
        off = R[np.triu_indices(N, 1)]
        print(f"{name:22} {N:3d} {T:7d} {train_len:6d} {er:7.2f} {off.mean():13.3f}")


if __name__ == "__main__":
    main()
