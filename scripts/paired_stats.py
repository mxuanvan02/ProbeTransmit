#!/usr/bin/env python3
"""Paired statistics over per-window results from Cycle 1 sweeps.

Reads the raw per-window CSV produced by cycle1_bprobe_sweep.py and computes
paired deltas between probe rules at each (bprobe_frac, V_param) cell.

Reports for each ordered pair (rule_a, rule_b) and metric in {rmse_mean,
loss_mean, missed_violation_pct}:

- mean delta (rule_a - rule_b),
- 95% bootstrap confidence interval,
- Wilcoxon signed-rank statistic and p-value (one-sided rule_a < rule_b),
- effect size (Cohen's dz).
"""
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 5000, seed: int = 2026) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(deltas) == 0:
        return float("nan"), float("nan")
    boots = rng.choice(deltas, size=(n_boot, len(deltas)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def wilcoxon_one_sided(deltas: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank for H0 median(d)>=0 vs H1 median(d)<0.

    Implemented locally to avoid scipy as a hard dependency.
    Uses asymptotic normal approximation with continuity correction.
    """
    deltas = deltas[deltas != 0]
    n = len(deltas)
    if n == 0:
        return float("nan"), float("nan")
    abs_d = np.abs(deltas)
    ranks = pd.Series(abs_d).rank(method="average").to_numpy()
    signs = np.sign(deltas)
    w_plus = float(np.sum(ranks[signs > 0]))
    w_minus = float(np.sum(ranks[signs < 0]))
    w_min = min(w_plus, w_minus)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w == 0:
        return float("nan"), float("nan")
    z = (w_min - mean_w + 0.5) / np.sqrt(var_w)
    # one-sided p-value for "rule_a smaller than rule_b" (negative deltas)
    from math import erf, sqrt
    p_one_sided = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    if w_plus < w_minus:
        # negative deltas dominate, support the alternative -> small p
        return w_min, float(p_one_sided)
    else:
        return w_min, float(1.0 - p_one_sided)


def cohens_dz(deltas: np.ndarray) -> float:
    if len(deltas) < 2:
        return float("nan")
    s = float(np.std(deltas, ddof=1))
    if s == 0:
        return float("nan")
    return float(np.mean(deltas) / s)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired stats for Cycle 1 sweep")
    p.add_argument("--input", required=True, help="raw CSV from cycle1_bprobe_sweep.py")
    p.add_argument("--metrics", nargs="+", default=["rmse_mean", "loss_mean", "missed_violation_pct"])
    p.add_argument("--baseline", default="random",
                   help="probe rule used as the comparison baseline (rule_b)")
    p.add_argument("--out", default=None, help="optional output CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.input)
    needed = {"window_id", "bprobe_frac", "V_param", "probe_rule"}
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"input CSV missing columns: {missing}")
    rules = sorted(raw["probe_rule"].unique().tolist())
    if args.baseline not in rules:
        raise ValueError(f"baseline {args.baseline} not in {rules}")

    rows = []
    for (frac, V), group in raw.groupby(["bprobe_frac", "V_param"]):
        wide = group.pivot_table(
            index="window_id",
            columns="probe_rule",
            values=args.metrics,
            aggfunc="mean",
        )
        for metric in args.metrics:
            for rule in rules:
                if rule == args.baseline:
                    continue
                if (metric, rule) not in wide.columns:
                    continue
                a = wide[(metric, rule)].to_numpy(dtype=float)
                b = wide[(metric, args.baseline)].to_numpy(dtype=float)
                deltas = a - b
                deltas = deltas[~np.isnan(deltas)]
                if len(deltas) == 0:
                    continue
                ci_lo, ci_hi = bootstrap_ci(deltas)
                wstat, pval = wilcoxon_one_sided(deltas)
                rows.append({
                    "bprobe_frac": frac,
                    "V_param": V,
                    "metric": metric,
                    "rule_a": rule,
                    "rule_b": args.baseline,
                    "n": len(deltas),
                    "mean_delta": float(np.mean(deltas)),
                    "median_delta": float(np.median(deltas)),
                    "ci95_lo": ci_lo,
                    "ci95_hi": ci_hi,
                    "wilcoxon_stat": wstat,
                    "wilcoxon_p_one_sided": pval,
                    "cohens_dz": cohens_dz(deltas),
                    "wins_pct": float(100 * np.mean(deltas < 0)),
                })

    out = pd.DataFrame(rows)
    print(out.sort_values(["bprobe_frac", "metric", "rule_a"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
