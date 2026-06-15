#!/usr/bin/env python3
"""Fair 30-window 3-way comparison: Whittle-heuristic vs CAW-VoU vs Whittle-exact.

All three probe policies run on the IDENTICAL 30 Intel-Berkeley windows used by
``whittle_comparison_30windows.csv`` (same select_starts(.., n_windows=30),
same per-window seeds ``50260 + window_id * 17``, same channel ``severe_burst``,
same budgets B_probe=B_payload=4, same horizon=240). This makes every baseline
paired at 30 windows so significance is computed on the same footing.

Whittle-exact solves a belief-MDP per node, so it is ~300x slower than the
heuristic. We parallelize across windows with multiprocessing to keep wall time
manageable (each worker runs all three policies on one window).

Usage:
    python scripts/whittle_exact_30windows.py [--windows 30] [--horizon 240]
                                              [--n-grid 16] [--vi-iter 50]
                                              [--workers 6]

Outputs:
    docs/whittle_exact_comparison_30windows.csv   (per-window, per-policy)
    console: means, 95% CI, Wilcoxon paired p, effect size, win/loss/tie, tails
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
    _HAVE_WILCOXON = True
except Exception:
    _HAVE_WILCOXON = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))  # for the policies/ package

from probe_transmit.channel import CHANNELS  # noqa: E402
from probe_transmit.data import select_starts  # noqa: E402
from probe_transmit.forecast import AR1Model  # noqa: E402
from probe_transmit.simulator import run_window  # noqa: E402
from probe_transmit.policies import TwoStagePolicy, DebtAwarePayload  # noqa: E402
from new_algorithm import CorrVoUProbe, fit_correlation  # noqa: E402
from policies.whittle_baseline import WhittleProbe, WhittlePayload  # noqa: E402
from policies.whittle_exact import WhittleExactProbe, WhittleExactPayload  # noqa: E402

# Globals populated once per worker process via the initializer.
_DATA = None
_AR = None
_R = None
_CHANNEL = None
_NGRID = 16
_VIITER = 50


def _init_worker(arr_path: str, n_grid: int, vi_iter: int):
    global _DATA, _AR, _R, _CHANNEL, _NGRID, _VIITER
    _DATA = np.load(arr_path)
    train = _DATA[:2000]
    _AR = AR1Model.fit(train)
    _AR.set_empirical_residuals(train[1:] - (train[:-1] * _AR.alpha + _AR.beta))
    _R = fit_correlation(train, shrinkage=0.1)
    _CHANNEL = CHANNELS["severe_burst"]
    _NGRID = n_grid
    _VIITER = vi_iter


def _timed_run_window(**kwargs) -> dict:
    t0 = time.perf_counter()
    m = run_window(**kwargs)
    dt = time.perf_counter() - t0
    horizon = kwargs["horizon"]
    m["runtime_ms_per_step"] = float(1000.0 * dt / max(horizon - 1, 1))
    return m


def _build_policies():
    """Fresh policy instances (stateful solvers must not be shared)."""
    return [
        ("Whittle-heuristic",
         TwoStagePolicy("Whittle-heuristic", WhittleProbe(), WhittlePayload())),
        ("CAW-VoU",
         TwoStagePolicy("CAW-VoU",
                        CorrVoUProbe(corr=_R, lambda_safety=6.0, w_debt=0.05),
                        DebtAwarePayload(V=1.0))),
        ("Whittle-exact",
         TwoStagePolicy("Whittle-exact",
                        WhittleExactProbe(lambda_safety=6.0, n_grid=_NGRID,
                                          vi_max_iter=_VIITER),
                        DebtAwarePayload(V=1.0))),
    ]


def _run_one_window(args):
    wi, start, horizon, b_probe, b_payload = args
    seed = 50260 + wi * 17
    out = []
    for name, pol in _build_policies():
        m = _timed_run_window(
            data=_DATA, ar=_AR, channel=_CHANNEL, policy=pol,
            start=start, horizon=horizon, seed=seed,
            b_probe=b_probe, b_payload=b_payload,
        )
        out.append({
            "window_id": wi,
            "seed": seed,
            "policy": name,
            "loss": float(m["loss_mean"]),
            "rmse": float(m["rmse_mean"]),
            "missed_vio": float(m["missed_violation_pct"]),
            "runtime": float(m["runtime_ms_per_step"]),
        })
    return out


def run(n_windows, horizon, b_probe, b_payload, n_grid, vi_iter, workers):
    arr_path = ROOT / "data" / "raw" / "intel_berkeley" / "intel_panel_30motes.npy"
    data_len = len(np.load(arr_path, mmap_mode="r"))
    starts = select_starts(data_len, horizon, n_windows)

    print(f"Fair 30-window 3-way: 30 sensors, {b_probe}/{b_payload} slots, "
          f"{n_windows} windows, horizon {horizon}")
    print(f"Whittle-exact: belief-MDP grid={n_grid}, value-iter={vi_iter}; "
          f"workers={workers}\n")
    print(f"Start indices (n_windows={n_windows}): {list(starts)}\n")

    jobs = [(wi, int(start), horizon, b_probe, b_payload)
            for wi, start in enumerate(starts)]

    t0 = time.perf_counter()
    rows = []
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(arr_path), n_grid, vi_iter),
    ) as ex:
        futs = {ex.submit(_run_one_window, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            elapsed = time.perf_counter() - t0
            print(f"  [{done}/{len(jobs)}] window {futs[fut]} done "
                  f"(elapsed {elapsed:.1f}s)")
    print(f"\nTotal wall time: {time.perf_counter() - t0:.1f}s")
    df = pd.DataFrame(rows).sort_values(["window_id", "policy"]).reset_index(drop=True)
    return df


def _paired(df, a, b, metric, alt):
    xa = df[df.policy == a].sort_values("window_id")[metric].to_numpy()
    xb = df[df.policy == b].sort_values("window_id")[metric].to_numpy()
    d = xa - xb
    if np.allclose(d, 0):
        return float("nan"), float("nan"), (0, 0, len(d))
    # Rank-biserial effect size for Wilcoxon signed-rank.
    nz = d[d != 0]
    ranks = pd.Series(np.abs(nz)).rank().to_numpy()
    rpos = ranks[nz > 0].sum()
    rneg = ranks[nz < 0].sum()
    tot = rpos + rneg
    rbc = (rpos - rneg) / tot if tot > 0 else float("nan")
    wins = int((d < 0).sum())   # a beats b (lower is better)
    losses = int((d > 0).sum())
    ties = int((d == 0).sum())
    if not _HAVE_WILCOXON:
        return float("nan"), rbc, (wins, losses, ties)
    try:
        _, p = wilcoxon(d, alternative=alt)
    except ValueError:
        p = float("nan")
    return p, rbc, (wins, losses, ties)


def _ci95(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return 0.0
    return 1.96 * x.std(ddof=1) / np.sqrt(n)


def summarize(df, out_csv):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    order = ["Whittle-heuristic", "CAW-VoU", "Whittle-exact"]
    present = [p for p in order if p in df.policy.unique()]
    metrics = ["loss", "rmse", "missed_vio", "runtime"]

    print("\n" + "=" * 86)
    print("FAIR 30-WINDOW 3-WAY COMPARISON (mean +/- 95% CI)")
    print("=" * 86)
    hdr = f"{'Policy':<20}{'Loss':>18}{'RMSE':>16}{'Missed-Vio%':>16}{'Runtime(ms)':>16}"
    print(hdr)
    print("-" * 86)
    for p in present:
        sub = df[df.policy == p]
        cells = []
        for mt in metrics:
            cells.append(f"{sub[mt].mean():.5f}+-{_ci95(sub[mt]):.5f}")
        print(f"{p:<20}{cells[0]:>18}{cells[1]:>16}{cells[2]:>16}{cells[3]:>16}")

    # Runtime ratios
    print("\n--- Runtime ratios (relative to Whittle-heuristic) ---")
    base = df[df.policy == "Whittle-heuristic"]["runtime"].mean()
    for p in present:
        rt = df[df.policy == p]["runtime"].mean()
        print(f"  {p:<20}: {rt:8.3f} ms  ({rt / base:6.1f}x heuristic)")
    if "Whittle-exact" in present and "CAW-VoU" in present:
        ex = df[df.policy == "Whittle-exact"]["runtime"].mean()
        cw = df[df.policy == "CAW-VoU"]["runtime"].mean()
        print(f"  Whittle-exact / CAW-VoU runtime ratio: {ex / cw:.1f}x")

    # Paired stats: CAW-VoU vs Whittle-exact and vs heuristic
    print("\n--- Wilcoxon paired tests (n=30 windows) ---")
    pairs = [("CAW-VoU", "Whittle-exact"), ("CAW-VoU", "Whittle-heuristic"),
             ("Whittle-exact", "Whittle-heuristic")]
    for a, b in pairs:
        if a not in present or b not in present:
            continue
        print(f"\n  {a}  vs  {b}:")
        for mt in metrics:
            # two-sided p plus directional context via win/loss
            p2, rbc, (w, l, t) = _paired(df, a, b, mt, "two-sided")
            mean_ratio = (df[df.policy == a][mt].mean()
                          / df[df.policy == b][mt].mean()
                          if df[df.policy == b][mt].mean() != 0 else float("nan"))
            print(f"    {mt:<11}: p(2s)={p2:.4f}  rank-biserial={rbc:+.3f}  "
                  f"win/loss/tie(={a} lower)={w}/{l}/{t}  "
                  f"mean_ratio({a}/{b})={mean_ratio:.3f}")

    # Tail metrics for CAW-VoU vs Whittle-exact (loss, rmse, missed_vio)
    print("\n--- Tail metrics (per-window distribution) ---")
    tail_qs = [("P50", 50), ("P90", 90), ("P99", 99), ("max", 100)]
    for mt in ["loss", "rmse", "missed_vio"]:
        print(f"  {mt}:")
        for p in present:
            x = df[df.policy == p][mt].to_numpy()
            vals = []
            for label, q in tail_qs:
                v = np.percentile(x, q)
                vals.append(f"{label}={v:.5f}")
            print(f"    {p:<20} " + "  ".join(vals))

    print(f"\nSaved per-window metrics to {out_csv}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=240)
    ap.add_argument("--b-probe", type=int, default=4)
    ap.add_argument("--b-payload", type=int, default=4)
    ap.add_argument("--n-grid", type=int, default=16)
    ap.add_argument("--vi-iter", type=int, default=50)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    df = run(args.windows, args.horizon, args.b_probe, args.b_payload,
             args.n_grid, args.vi_iter, args.workers)
    out_csv = ROOT / "docs" / "whittle_exact_comparison_30windows.csv"
    summarize(df, out_csv)


if __name__ == "__main__":
    main()
