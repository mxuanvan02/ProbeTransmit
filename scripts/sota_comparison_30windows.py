#!/usr/bin/env python3
"""SOTA baselines vs CAW-VoU: 30-window paired comparison on Intel Berkeley.

Runs eight probe-then-transmit policies on the IDENTICAL 30 Intel-Berkeley
windows used by ``whittle_exact_comparison_30windows.csv``:

    same select_starts(.., n_windows=30), same per-window seed 50260 + wi*17,
    same channel ``severe_burst``, same N=30, B_probe=B_payload=4, horizon=240.

Policies
--------
1. CAW-VoU              (CorrVoUProbe + DebtAwarePayload)         [reference]
2. Whittle-heuristic   (WhittleProbe + WhittlePayload)
3. AoII-greedy         (genie AoII oracle)
4. AoII+debt           (genie AoII oracle + fairness floor)
5. MaxWeight-AoI       (Lyapunov age queue)
6. MaxWeight-VoU       (Lyapunov VoU queue, no corr/debt)
7. VoI-greedy          (posterior-variance reduction)
8. VoI+debt            (VoI + fairness floor)

(AoII+debt+threshold is also run and logged but kept out of the headline table
since AoII+debt is the stronger AoII variant.)

Each policy runs on all 30 windows -> paired. Statistics vs CAW-VoU: mean +/-
95% CI, Wilcoxon signed-rank p, win/tie/loss, rank-biserial effect size.

Outputs
-------
    docs/sota_comparison_30windows.csv   (per-window, per-policy, full schema)
    console summary + the data the report (sota_comparison_report.md) needs.
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
sys.path.insert(0, str(ROOT))

from probe_transmit.channel import CHANNELS  # noqa: E402
from probe_transmit.data import select_starts  # noqa: E402
from probe_transmit.forecast import AR1Model  # noqa: E402
from probe_transmit.simulator import run_window  # noqa: E402
from probe_transmit.policies import TwoStagePolicy, DebtAwarePayload  # noqa: E402
from new_algorithm import CorrVoUProbe, fit_correlation  # noqa: E402
from policies.sota_recent_baselines import make_sota_policy  # noqa: E402
from policies.whittle_baseline import WhittleProbe, WhittlePayload  # noqa: E402
from policies.aoii_baseline import make_aoii_policy  # noqa: E402
from policies.maxweight_baseline import make_maxweight_policy  # noqa: E402
from policies.voi_baseline import make_voi_policy  # noqa: E402

# Headline policy order (CAW-VoU first as the reference).
POLICY_ORDER = [
    "CAW-VoU",
    "Whittle-heuristic",
    "AoII-greedy",
    "AoII+debt",
    "AoII+debt+threshold",
    "MaxWeight-AoI",
    "MaxWeight-VoU",
    "VoI-greedy",
    "VoI+debt",
    "OnlineWhittle-2024",
    "MultiChanWhittle-2023",
    "RiskAwareAoII-2026",
    "QAoI-Whittle-2024",
    "QVAoI-2024",
]

_DATA = None
_AR = None
_R = None
_CHANNEL = None


def _init_worker(arr_path: str):
    global _DATA, _AR, _R, _CHANNEL
    _DATA = np.load(arr_path)
    train = _DATA[:2000]
    _AR = AR1Model.fit(train)
    _AR.set_empirical_residuals(train[1:] - (train[:-1] * _AR.alpha + _AR.beta))
    _R = fit_correlation(train, shrinkage=0.1)
    _CHANNEL = CHANNELS["severe_burst"]


def _build_policies():
    """Fresh stateful policy instances per window (must not be shared)."""
    return [
        ("CAW-VoU",
         TwoStagePolicy("CAW-VoU",
                        CorrVoUProbe(corr=_R, lambda_safety=6.0, w_debt=0.05),
                        DebtAwarePayload(V=1.0))),
        ("Whittle-heuristic",
         TwoStagePolicy("Whittle-heuristic", WhittleProbe(), WhittlePayload())),
        ("AoII-greedy", make_aoii_policy("greedy")),
        ("AoII+debt", make_aoii_policy("debt")),
        ("AoII+debt+threshold", make_aoii_policy("debt_threshold")),
        ("MaxWeight-AoI", make_maxweight_policy("aoi")),
        ("MaxWeight-VoU", make_maxweight_policy("vou")),
        ("VoI-greedy", make_voi_policy("greedy")),
        ("VoI+debt", make_voi_policy("debt")),
        # Recent (2023-2026) SoTA baselines, as-published (gamma=0). Faithful
        # reimplementations of the published index rules; see
        # docs/sota_baselines_plan.md for verified arXiv IDs.
        ("OnlineWhittle-2024", make_sota_policy("online_whittle", gamma=0.0)),
        ("MultiChanWhittle-2023", make_sota_policy("mc_whittle", gamma=0.0)),
        ("RiskAwareAoII-2026", make_sota_policy("risk_aoii")),
        ("QAoI-Whittle-2024", make_sota_policy("qaoi_whittle")),
        ("QVAoI-2024", make_sota_policy("qvaoi")),
    ]


def _timed_run_window(**kwargs) -> dict:
    t0 = time.perf_counter()
    m = run_window(**kwargs)
    dt = time.perf_counter() - t0
    horizon = kwargs["horizon"]
    m["runtime_ms_per_step"] = float(1000.0 * dt / max(horizon - 1, 1))
    return m


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
            "overshoot_p90": float(m.get("overshoot_p90", 0.0)),
            "overshoot_p99": float(m.get("overshoot_p99", 0.0)),
            "runtime": float(m["runtime_ms_per_step"]),
            "max_age": float(m["max_age"]),
            "jain_probe": float(m["probe_fairness_jain"]),
            "jain_payload": float(m["payload_fairness_jain"]),
        })
    return out


def run(n_windows, horizon, b_probe, b_payload, workers):
    arr_path = ROOT / "data" / "raw" / "intel_berkeley" / "intel_panel_30motes.npy"
    data_len = len(np.load(arr_path, mmap_mode="r"))
    starts = select_starts(data_len, horizon, n_windows)

    print(f"SOTA 30-window paired: 30 sensors, {b_probe}/{b_payload} slots, "
          f"{n_windows} windows, horizon {horizon}, channel severe_burst")
    print(f"Start indices: {list(starts)}\n")

    jobs = [(wi, int(start), horizon, b_probe, b_payload)
            for wi, start in enumerate(starts)]

    t0 = time.perf_counter()
    rows = []
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(str(arr_path),)
    ) as ex:
        futs = {ex.submit(_run_one_window, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            print(f"  [{done}/{len(jobs)}] window {futs[fut]} done "
                  f"(elapsed {time.perf_counter() - t0:.1f}s)")
    print(f"\nTotal wall time: {time.perf_counter() - t0:.1f}s")
    df = pd.DataFrame(rows).sort_values(["window_id", "policy"]).reset_index(drop=True)
    return df


def _ci95(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return 0.0
    return 1.96 * x.std(ddof=1) / np.sqrt(n)


def _paired(df, a, b, metric):
    """Paired stats for policy a vs b on `metric` (lower is better)."""
    xa = df[df.policy == a].sort_values("window_id")[metric].to_numpy()
    xb = df[df.policy == b].sort_values("window_id")[metric].to_numpy()
    d = xa - xb
    wins = int((d < 0).sum())    # a better than b (lower)
    losses = int((d > 0).sum())
    ties = int((d == 0).sum())
    if np.allclose(d, 0):
        return float("nan"), float("nan"), (wins, losses, ties)
    nz = d[d != 0]
    ranks = pd.Series(np.abs(nz)).rank().to_numpy()
    rpos = ranks[nz > 0].sum()
    rneg = ranks[nz < 0].sum()
    tot = rpos + rneg
    rbc = (rpos - rneg) / tot if tot > 0 else float("nan")
    p = float("nan")
    if _HAVE_WILCOXON:
        try:
            _, p = wilcoxon(d, alternative="two-sided")
        except ValueError:
            p = float("nan")
    return p, rbc, (wins, losses, ties)


def summarize(df, out_csv):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    present = [p for p in POLICY_ORDER if p in df.policy.unique()]
    metrics = ["loss", "rmse", "missed_vio", "overshoot_p90", "overshoot_p99", "runtime"]

    print("\n" + "=" * 100)
    print("SOTA 30-WINDOW PAIRED COMPARISON (mean +/- 95% CI)")
    print("=" * 100)
    hdr = (f"{'Policy':<22}{'Loss':>20}{'RMSE':>16}"
           f"{'Missed-Vio%':>16}{'Runtime(ms)':>16}")
    print(hdr)
    print("-" * 100)
    for p in present:
        sub = df[df.policy == p]
        cells = [f"{sub[mt].mean():.6f}+-{_ci95(sub[mt]):.6f}" for mt in metrics]
        print(f"{p:<22}{cells[0]:>20}{cells[1]:>16}{cells[2]:>16}{cells[3]:>16}")

    print("\n--- Wilcoxon paired tests vs CAW-VoU (n=30, lower is better) ---")
    for b in present:
        if b == "CAW-VoU":
            continue
        print(f"\n  CAW-VoU vs {b}:")
        for mt in metrics:
            p2, rbc, (w, l, t) = _paired(df, "CAW-VoU", b, mt)
            mean_a = df[df.policy == "CAW-VoU"][mt].mean()
            mean_b = df[df.policy == b][mt].mean()
            ratio = mean_a / mean_b if mean_b != 0 else float("nan")
            print(f"    {mt:<11}: p={p2:.4f}  rank-biserial={rbc:+.3f}  "
                  f"CAW win/loss/tie={w}/{l}/{t}  "
                  f"mean_ratio(CAW/{b})={ratio:.3f}")

    print(f"\nSaved per-window metrics to {out_csv}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=240)
    ap.add_argument("--b-probe", type=int, default=4)
    ap.add_argument("--b-payload", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    df = run(args.windows, args.horizon, args.b_probe, args.b_payload, args.workers)
    out_csv = ROOT / "docs" / "sota_comparison_30windows.csv"
    summarize(df, out_csv)


if __name__ == "__main__":
    main()
