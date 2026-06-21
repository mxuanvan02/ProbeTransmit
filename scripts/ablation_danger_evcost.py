#!/usr/bin/env python3
"""Danger-term ablation (ev-cost branch): classic vs estimation-theoretic VoU.

Reproduces the ev-cost rows of the danger-term ablation table in the manuscript.
Compares the deployed classic point-in-time exit-probability danger term against
an estimation-theoretic ev-cost VoU (error-covariance reduction + expected
shortfall, dimensionless rho, no hand-tuned lambda), on the 30 matched Intel
per-arm windows. Reports mean loss, missed-violation rate, and overshoot tails.
Run: python scripts/ablation_danger_evcost.py [n_windows]   (default 30).
"""
from __future__ import annotations

import sys, time
from pathlib import Path
import numpy as np

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

PATH = ROOT / "data" / "raw" / "intel_berkeley" / "intel_panel_30motes.npy"


def evaluate(factory, ar, data, ch, starts, horizon, b):
    L, M, P90, P99, MX = [], [], [], [], []
    for wi, start in enumerate(starts):
        m = run_window(data=data, ar=ar, channel=ch, policy=factory(),
                       start=int(start), horizon=horizon, seed=80260 + wi * 17,
                       b_probe=b, b_payload=b)
        L.append(m["loss_mean"]); M.append(m["missed_violation_pct"])
        P90.append(m["overshoot_p90"]); P99.append(m["overshoot_p99"]); MX.append(m["overshoot_max"])
    return np.mean(L), np.mean(M), np.mean(P90), np.mean(P99), np.mean(MX)


def main(n_windows=30, horizon=240, b=4):
    data = np.load(PATH).astype(float)
    T, N = data.shape
    train = data[:2000]
    ar = AR1Model.fit(train)
    ar.set_empirical_residuals(train[1:] - (train[:-1] * ar.alpha + ar.beta))
    R = fit_correlation(train, shrinkage=0.1)
    safety.SAFE_MIN, safety.SAFE_MAX, safety.RANGE = 18.0, 32.0, 14.0
    starts = select_starts(T, horizon, n_windows)
    ch = CHANNELS["severe_burst"]

    rows = [("CAW-VoU classic", lambda: TwoStagePolicy("c",
             CorrVoUProbe(corr=R, lambda_safety=6.0, w_debt=0.05), DebtAwarePayload(V=1.0)))]
    for k in [1.0, 3.0, 6.0]:
        rows.append((f"ev_cost rho={k}",
                     lambda k=k: TwoStagePolicy("e",
                       CorrVoUProbe(corr=R, lambda_safety=6.0, w_debt=0.05, vou_mode="ev_cost", rho=k),
                       DebtAwarePayload(V=1.0))))
    for k in [3.0, 6.0]:
        rows.append((f"severity rho={k}",
                     lambda k=k: TwoStagePolicy("s",
                       CorrVoUProbe(corr=R, lambda_safety=6.0, w_debt=0.05, vou_mode="severity", rho=k),
                       DebtAwarePayload(V=1.0))))

    t0 = time.perf_counter()
    print(f"=== INTEL per-arm | {n_windows} win | horizon={horizon} b={b} | mean + TAIL ===")
    print(f"  {'policy':18s} {'loss':>10s} {'missed%':>9s} {'over_P90':>10s} {'over_P99':>10s} {'over_max':>9s}")
    base = None
    for name, fac in rows:
        L, M, P90, P99, MX = evaluate(fac, ar, data, ch, starts, horizon, b)
        if base is None:
            base = L
        ratio = L / base if base else 1.0
        print(f"  {name:18s} {L:10.5f} {M:9.4f} {P90:10.4f} {P99:10.4f} {MX:9.4f}   x{ratio:.2f}")
    print(f"[{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
