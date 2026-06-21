"""Head-to-head: CAW-VoU vs a modern DT+AoI synchronization baseline.

Same 30 Intel-Berkeley windows, same per-window seed (50260 + wi*17), same
severe_burst channel, same B_probe=B_payload=4 as the headline benchmark.
Adds the new DT+AoI (greedy) and DT+AoI+debt policies alongside CAW-VoU and
the two closest incumbents (VoU, MaxAoI) for context.

Question: does CAW-VoU's threshold-aware term beat a strong channel-aware
twin-AoI rule that has NO threshold term? If yes, the DT positioning is earned.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

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
from probe_transmit import policies as _pol  # noqa: E402
from policies.dt_aoi_baseline import make_dt_aoi_policy  # noqa: E402

_DATA = None
_AR = None
_R = None
_CHANNEL = None

POLICY_ORDER = ["CAW-VoU", "DT+AoI", "DT+AoI+debt", "VoU", "MaxAoI"]


def _init_worker(arr_path: str):
    global _DATA, _AR, _R, _CHANNEL
    _DATA = np.load(arr_path)
    train = _DATA[:2000]
    _AR = AR1Model.fit(train)
    _AR.set_empirical_residuals(train[1:] - (train[:-1] * _AR.alpha + _AR.beta))
    _R = fit_correlation(train, shrinkage=0.1)
    _CHANNEL = CHANNELS["severe_burst"]


def _build_policies():
    return [
        ("CAW-VoU",
         TwoStagePolicy("CAW-VoU",
                        CorrVoUProbe(corr=_R, lambda_safety=6.0, w_debt=0.05),
                        DebtAwarePayload(V=1.0))),
        ("DT+AoI", make_dt_aoi_policy("greedy")),
        ("DT+AoI+debt", make_dt_aoi_policy("debt")),
        ("VoU", _pol.build("vou", lambda_safety=6.0)),
        ("MaxAoI", _pol.build("max_aoi")),
    ]


def _run_one_window(args):
    wi, start, horizon, b_probe, b_payload = args
    seed = 50260 + wi * 17
    out = []
    for name, pol in _build_policies():
        m = run_window(
            data=_DATA, ar=_AR, channel=_CHANNEL, policy=pol,
            start=start, horizon=horizon, seed=seed,
            b_probe=b_probe, b_payload=b_payload,
        )
        out.append({
            "window_id": wi, "policy": name,
            "loss": m["loss_mean"], "rmse": m["rmse_mean"],
            "missed_vio": m["missed_violation_pct"],
        })
    return out


def _ci95(s):
    s = np.asarray(s, dtype=float)
    if len(s) < 2:
        return 0.0
    return 1.96 * s.std(ddof=1) / np.sqrt(len(s))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=240)
    ap.add_argument("--b-probe", type=int, default=4)
    ap.add_argument("--b-payload", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    arr_path = ROOT / "data" / "raw" / "intel_berkeley" / "intel_panel_30motes.npy"
    data_len = len(np.load(arr_path, mmap_mode="r"))
    starts = select_starts(data_len, args.horizon, args.windows)
    print(f"DT+AoI head-to-head: 30 sensors, {args.b_probe}/{args.b_payload} slots, "
          f"{args.windows} windows, horizon {args.horizon}, channel severe_burst")

    jobs = [(wi, int(s), args.horizon, args.b_probe, args.b_payload)
            for wi, s in enumerate(starts)]
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(str(arr_path),)) as ex:
        futs = {ex.submit(_run_one_window, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            rows.extend(fut.result())
    print(f"wall {time.perf_counter()-t0:.1f}s\n")

    df = pd.DataFrame(rows)
    out_csv = ROOT / "docs" / "dt_aoi_comparison_30windows.csv"
    df.sort_values(["window_id", "policy"]).to_csv(out_csv, index=False)

    print(f"{'policy':14s} {'loss':>10s} {'rmse':>9s} {'missed%':>9s}")
    for p in POLICY_ORDER:
        sub = df[df.policy == p]
        print(f"{p:14s} {sub.loss.mean():10.5f} {sub.rmse.mean():9.4f} "
              f"{sub.missed_vio.mean():9.4f}")

    # paired Wilcoxon CAW-VoU vs DT+AoI variants
    try:
        from scipy.stats import wilcoxon
        piv = df.pivot(index="window_id", columns="policy", values="loss")
        for opp in ["DT+AoI", "DT+AoI+debt"]:
            stat, p = wilcoxon(piv["CAW-VoU"], piv[opp], alternative="less")
            wins = int((piv["CAW-VoU"] < piv[opp]).sum())
            print(f"\nCAW-VoU vs {opp} (loss): win {wins}/{len(piv)}, "
                  f"Wilcoxon one-sided p={p:.2e}")
        # missed violations
        pivm = df.pivot(index="window_id", columns="policy", values="missed_vio")
        for opp in ["DT+AoI", "DT+AoI+debt"]:
            cawm = pivm["CAW-VoU"].mean(); om = pivm[opp].mean()
            red = 100 * (1 - cawm / om) if om > 0 else float("nan")
            print(f"missed%: CAW-VoU {cawm:.4f} vs {opp} {om:.4f} -> {red:.1f}% lower")
    except Exception as e:
        print("stats skipped:", e)
    print("DONE")


if __name__ == "__main__":
    main()
