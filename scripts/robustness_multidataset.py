#!/usr/bin/env python3
"""Cross-dataset ROBUSTNESS re-run for CAW-VoU vs SOTA baselines.

Runs the IDENTICAL eight-policy probe-then-transmit comparison on TWO real,
public multi-sensor datasets that sit at opposite ends of the spatial-
correlation spectrum:

    * Beijing Multi-Site Air Quality (12 stations, TEMP)  -> UNIFORM-HIGH corr
      (eff. rank ~1.7, 0% pairs < 0.3).
    * KETI Smart-Building (40 office rooms, temperature)   -> BLOCK-CLUSTERED
      (eff. rank ~7.3, within-floor minus across-floor corr gap +0.15).

The point of the re-run is to show the scheduler's safety advantage is ROBUST
across both correlation regimes, not an artifact of one trace. To make the
"missed threshold violation" metric meaningful and COMPARABLE across processes
with very different physical ranges, the safety band for each dataset is the
empirical [2.5, 97.5] percentile band of that dataset (data-driven, reported
explicitly), giving ~5% violating samples in each.

Uniform params across both datasets:
    channel = severe_burst, B_probe = B_payload = 4, horizon = 240,
    n_windows = 30, per-window seed = 50260 + wi*17, AR(1) fit on first 2000
    (or 60% if shorter) samples, correlation shrinkage 0.1.

Outputs (one CSV per dataset, plus a combined summary JSON):
    docs/robustness_<dataset>.csv
    docs/robustness_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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

from probe_transmit import safety  # noqa: E402
from probe_transmit.channel import CHANNELS  # noqa: E402
from probe_transmit.data import select_starts  # noqa: E402
from probe_transmit.forecast import AR1Model  # noqa: E402
from probe_transmit.simulator import run_window  # noqa: E402
from probe_transmit.policies import TwoStagePolicy, DebtAwarePayload  # noqa: E402
from new_algorithm import CorrVoUProbe, fit_correlation  # noqa: E402
from policies.whittle_baseline import WhittleProbe, WhittlePayload  # noqa: E402
from policies.aoii_baseline import make_aoii_policy  # noqa: E402
from policies.maxweight_baseline import make_maxweight_policy  # noqa: E402
from policies.voi_baseline import make_voi_policy  # noqa: E402

POLICY_ORDER = [
    "CAW-VoU", "Whittle-heuristic", "AoII-greedy", "AoII+debt",
    "MaxWeight-AoI", "MaxWeight-VoU", "VoI-greedy", "VoI+debt",
]

DATASETS = {
    "beijing": {
        "panel": ROOT / "data" / "raw" / "_candidates" / "beijing_prsa" / "beijing_temp_panel.npy",
        "regime": "uniform-high correlation (12 stations, TEMP)",
    },
    "keti": {
        "panel": ROOT / "data" / "raw" / "_candidates" / "keti_smartbuilding" / "keti_clean_panel.npy",
        "regime": "block-clustered correlation (40 office rooms)",
    },
}


def _build_policies(R):
    return [
        ("CAW-VoU", TwoStagePolicy("CAW-VoU",
            CorrVoUProbe(corr=R, lambda_safety=6.0, w_debt=0.05),
            DebtAwarePayload(V=1.0))),
        ("Whittle-heuristic", TwoStagePolicy("Whittle-heuristic", WhittleProbe(), WhittlePayload())),
        ("AoII-greedy", make_aoii_policy("greedy")),
        ("AoII+debt", make_aoii_policy("debt")),
        ("MaxWeight-AoI", make_maxweight_policy("aoi")),
        ("MaxWeight-VoU", make_maxweight_policy("vou")),
        ("VoI-greedy", make_voi_policy("greedy")),
        ("VoI+debt", make_voi_policy("debt")),
    ]


def _ci95(x):
    x = np.asarray(x, float)
    n = len(x)
    return 1.96 * x.std(ddof=1) / np.sqrt(n) if n >= 2 else 0.0


def _paired(df, a, b, metric):
    xa = df[df.policy == a].sort_values("window_id")[metric].to_numpy()
    xb = df[df.policy == b].sort_values("window_id")[metric].to_numpy()
    d = xa - xb
    wins = int((d < 0).sum()); losses = int((d > 0).sum()); ties = int((d == 0).sum())
    p = float("nan")
    if _HAVE_WILCOXON and not np.allclose(d, 0):
        try:
            _, p = wilcoxon(d, alternative="two-sided")
        except ValueError:
            p = float("nan")
    return p, (wins, losses, ties)


def run_dataset(name, n_windows, horizon, b_probe, b_payload):
    cfg = DATASETS[name]
    data = np.load(cfg["panel"]).astype(float)
    T, N = data.shape
    train_len = min(2000, int(0.6 * T))

    # data-driven safety band: empirical [2.5, 97.5] pct over the whole trace
    lo = float(np.percentile(data, 2.5))
    hi = float(np.percentile(data, 97.5))
    safety.SAFE_MIN = lo
    safety.SAFE_MAX = hi
    safety.RANGE = hi - lo
    viol_pct = float(100 * ((data < lo) | (data > hi)).mean())

    train = data[:train_len]
    ar = AR1Model.fit(train)
    ar.set_empirical_residuals(train[1:] - (train[:-1] * ar.alpha + ar.beta))
    R = fit_correlation(train, shrinkage=0.1)
    off = R[np.triu_indices(N, 1)]

    starts = select_starts(T, horizon, n_windows)
    channel = CHANNELS["severe_burst"]

    print(f"\n=== {name.upper()} | {cfg['regime']} ===")
    print(f"T={T} N={N} train_len={train_len} SAFE=[{lo:.2f},{hi:.2f}] "
          f"violating={viol_pct:.2f}% | fitted R off-diag mean={off.mean():.3f} "
          f"std={off.std():.3f} frac<0.3={(off<0.3).mean():.2f}")

    rows = []
    t0 = time.perf_counter()
    for wi, start in enumerate(starts):
        seed = 50260 + wi * 17
        for pname, pol in _build_policies(R):
            m = run_window(data=data, ar=ar, channel=channel, policy=pol,
                           start=int(start), horizon=horizon, seed=seed,
                           b_probe=b_probe, b_payload=b_payload)
            rows.append({
                "dataset": name, "window_id": wi, "seed": seed, "policy": pname,
                "loss": float(m["loss_mean"]), "rmse": float(m["rmse_mean"]),
                "missed_vio": float(m["missed_violation_pct"]),
                "max_age": float(m["max_age"]),
                "jain_payload": float(m["payload_fairness_jain"]),
            })
        print(f"  [{wi+1}/{n_windows}] done (elapsed {time.perf_counter()-t0:.1f}s)")

    df = pd.DataFrame(rows).sort_values(["window_id", "policy"]).reset_index(drop=True)
    out_csv = ROOT / "docs" / f"robustness_{name}.csv"
    df.to_csv(out_csv, index=False)

    present = [p for p in POLICY_ORDER if p in df.policy.unique()]
    print(f"\n  {'Policy':<20}{'Loss':>18}{'Missed-Vio%':>16}{'RMSE':>14}")
    summ = {}
    for p in present:
        sub = df[df.policy == p]
        summ[p] = {
            "loss_mean": float(sub.loss.mean()), "loss_ci95": float(_ci95(sub.loss)),
            "missed_vio_mean": float(sub.missed_vio.mean()), "missed_vio_ci95": float(_ci95(sub.missed_vio)),
            "rmse_mean": float(sub.rmse.mean()), "rmse_ci95": float(_ci95(sub.rmse)),
        }
        print(f"  {p:<20}{sub.loss.mean():>10.5f}+-{_ci95(sub.loss):<7.5f}"
              f"{sub.missed_vio.mean():>9.3f}+-{_ci95(sub.missed_vio):<6.3f}"
              f"{sub.rmse.mean():>8.4f}+-{_ci95(sub.rmse):<6.4f}")

    print(f"\n  --- Wilcoxon vs CAW-VoU (n={n_windows}, lower=better) ---")
    pstats = {}
    for b in present:
        if b == "CAW-VoU":
            continue
        pstats[b] = {}
        for mt in ["loss", "missed_vio", "rmse"]:
            pv, (w, l, t) = _paired(df, "CAW-VoU", b, mt)
            ma = df[df.policy == "CAW-VoU"][mt].mean(); mb = df[df.policy == b][mt].mean()
            ratio = ma / mb if mb != 0 else float("nan")
            pstats[b][mt] = {"p": pv, "win": w, "loss": l, "tie": t, "ratio_caw_over_b": ratio}
        print(f"    vs {b:<18} loss p={pstats[b]['loss']['p']:.4f} "
              f"miss p={pstats[b]['missed_vio']['p']:.4f} "
              f"(CAW miss win/loss {pstats[b]['missed_vio']['win']}/{pstats[b]['missed_vio']['loss']})")

    return {
        "regime": cfg["regime"], "T": T, "N": N, "train_len": train_len,
        "safe_band": [lo, hi], "violating_pct": viol_pct,
        "R_offdiag": {"mean": float(off.mean()), "std": float(off.std()),
                      "frac_lt_0.3": float((off < 0.3).mean())},
        "csv": str(out_csv), "summary": summ, "wilcoxon_vs_caw": pstats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=240)
    ap.add_argument("--b-probe", type=int, default=4)
    ap.add_argument("--b-payload", type=int, default=4)
    ap.add_argument("--datasets", nargs="+", default=["beijing", "keti"])
    args = ap.parse_args()

    result = {"params": {"windows": args.windows, "horizon": args.horizon,
                         "b_probe": args.b_probe, "b_payload": args.b_payload,
                         "channel": "severe_burst", "seed_formula": "50260+wi*17",
                         "safety_band": "empirical [2.5,97.5] pct per dataset"}}
    for name in args.datasets:
        result[name] = run_dataset(name, args.windows, args.horizon, args.b_probe, args.b_payload)

    out_json = ROOT / "docs" / "robustness_summary.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nSaved combined summary to {out_json}")


if __name__ == "__main__":
    main()
