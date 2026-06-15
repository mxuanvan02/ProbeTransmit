#!/usr/bin/env python3
"""Lightweight public smoke test for the ProbeTransmit release.

This test does not require third-party raw datasets. It verifies that the public
package imports correctly and that the two-stage scheduler can run on a small
simulated multivariate sensor panel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from probe_transmit.channel import CHANNELS  # noqa: E402
from probe_transmit.forecast import AR1Model  # noqa: E402
from probe_transmit.policies import build  # noqa: E402
from probe_transmit.simulator import run_window  # noqa: E402


def make_panel(t: int = 180, n: int = 12, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros((t, n), dtype=float)
    x[0] = rng.normal(24.0, 0.8, size=n)
    phases = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    for k in range(1, t):
        seasonal = 0.35 * np.sin(2.0 * np.pi * k / 48.0 + phases)
        shock = rng.normal(0.0, 0.08, size=n)
        x[k] = 0.94 * x[k - 1] + 0.06 * (24.0 + seasonal) + shock
    # Inject a short threshold-adjacent event so safety metrics are exercised.
    x[70:95, 3] += np.linspace(0.0, 9.0, 25)
    x[95:120, 3] += np.linspace(9.0, 0.0, 25)
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="run the lightweight public smoke test")
    ap.add_argument("--output", default="reports/verify_probe_transmit.json")
    args = ap.parse_args()

    data = make_panel()
    train = data[:80]
    ar = AR1Model.fit(train)
    policy = build("active", V=1.0, mode="balanced")
    result = run_window(
        data=data,
        ar=ar,
        channel=CHANNELS["burst"],
        policy=policy,
        start=80,
        horizon=60,
        seed=11,
        b_probe=3,
        b_payload=3,
        metadata_noise=0.05,
        metadata_loss=0.0,
        debt_mode="bounded",
    )
    checks = {
        "imports": True,
        "finite_loss": bool(np.isfinite(result["loss_mean"])),
        "finite_rmse": bool(np.isfinite(result["rmse_mean"])),
        "fairness_in_range": bool(0.0 <= result["payload_fairness_jain"] <= 1.0),
        "result": result,
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    print(f"WROTE {out}")
    return 0 if all(v for k, v in checks.items() if k != "result") else 1


if __name__ == "__main__":
    raise SystemExit(main())
