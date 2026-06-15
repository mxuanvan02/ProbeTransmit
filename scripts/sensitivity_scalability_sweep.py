#!/usr/bin/env python3
"""Sensitivity and scalability sweeps for the ProbeTransmit (CAW-VoU) paper.

Runs four sweeps on the Intel Berkeley panel (severe-burst Gilbert-Elliott
channel), comparing the proposed CAW-VoU scheduler against the correlation-
agnostic VoU baseline:

  1. N scalability   : N in {30, 50, 100}, B_probe = B_payload = 4.
  2. B_probe         : B_probe in {2, 4, 6, 8}, N = 30, B_payload = 4.
  3. lambda_safety   : lambda in {2, 4, 6, 8, 10}, N = 30, B = 4.
  4. channel loss    : packet_loss in {0.1, 0.2, 0.3, 0.4}, N = 30, B = 4.

For N > 30 the panel is expanded with documented virtual loops
(``expand_virtual_loops``); this is an N-loop scheduling stress test, not new
real sensors. Every configuration is evaluated over ``--n-windows`` paired
windows (same start/seed for CAW-VoU and VoU). We report mean and 95% CI.

Outputs (CSV) under code/data/processed/<sweep>/ and a tidy summary CSV under
code/docs/sensitivity_<sweep>.csv.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_transmit.channel import CHANNELS, ChannelParams
from probe_transmit.data import select_starts, expand_virtual_loops
from probe_transmit.forecast import AR1Model
from probe_transmit.simulator import run_window
from probe_transmit.policies import build, TwoStagePolicy, DebtAwarePayload
from new_algorithm import CorrVoUProbe, fit_correlation

PANEL = ROOT / "data" / "raw" / "intel_berkeley" / "intel_panel_30motes.npy"
PROC = ROOT / "data" / "processed"
DOCS = ROOT / "docs"


def ci95(x: np.ndarray) -> float:
    """Half-width of the 95% CI of the mean (normal approx, n-1 denom)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0
    return float(1.96 * np.std(x, ddof=1) / np.sqrt(n))


def make_loss_channel(base_loss_good: ChannelParams, packet_loss: float) -> ChannelParams:
    """A homogeneous-loss channel where p_ok = 1 - packet_loss in both states.

    Keeps the Gilbert-Elliott transition structure of ``severe_burst`` but lets
    us sweep an explicit average packet-loss rate. p_ok_bad is set to half of
    p_ok_good to preserve the good/bad asymmetry while hitting the target mean.
    """
    p_ok = 1.0 - float(packet_loss)
    return ChannelParams(
        name=f"loss_{packet_loss:.2f}",
        p_gb=base_loss_good.p_gb,
        p_bg=base_loss_good.p_bg,
        p_ok_good=float(np.clip(p_ok + 0.10, 0.0, 1.0)),
        p_ok_bad=float(np.clip(p_ok - 0.10, 0.0, 1.0)),
    )


@dataclass
class Cfg:
    N: int
    b_probe: int
    b_payload: int
    lambda_safety: float
    channel: ChannelParams
    metadata_loss: float


def build_data(N: int):
    base = np.load(PANEL)
    data = base if N <= base.shape[1] else expand_virtual_loops(base, N)
    train = data[:2000]
    ar = AR1Model.fit(train)
    ar.set_empirical_residuals(train[1:] - (train[:-1] * ar.alpha + ar.beta))
    R = fit_correlation(train, shrinkage=0.1)
    return data, ar, R


def run_cfg(cfg: Cfg, n_windows: int, horizon: int, label: str) -> list[dict]:
    data, ar, R = build_data(cfg.N)
    starts = select_starts(len(data), horizon, n_windows)
    rows: list[dict] = []
    for wi, start in enumerate(starts):
        seed = 60260 + wi * 17
        # VoU baseline.
        pol_vou = build("vou", lambda_safety=cfg.lambda_safety)
        m = run_window(
            data=data, ar=ar, channel=cfg.channel, policy=pol_vou,
            start=start, horizon=horizon, seed=seed,
            b_probe=cfg.b_probe, b_payload=cfg.b_payload,
            metadata_loss=cfg.metadata_loss,
        )
        m.update({"window_id": wi, "policy_name": "VoU", **_cfg_cols(cfg)})
        rows.append(m)
        # CAW-VoU (proposed).
        pol_caw = TwoStagePolicy(
            "CAW-VoU",
            CorrVoUProbe(corr=R, lambda_safety=cfg.lambda_safety, w_debt=0.05),
            DebtAwarePayload(V=1.0),
        )
        m = run_window(
            data=data, ar=ar, channel=cfg.channel, policy=pol_caw,
            start=start, horizon=horizon, seed=seed,
            b_probe=cfg.b_probe, b_payload=cfg.b_payload,
            metadata_loss=cfg.metadata_loss,
        )
        m.update({"window_id": wi, "policy_name": "CAW-VoU", **_cfg_cols(cfg)})
        rows.append(m)
    return rows


def _cfg_cols(cfg: Cfg) -> dict:
    return {
        "N": cfg.N, "b_probe": cfg.b_probe, "b_payload": cfg.b_payload,
        "lambda_safety": cfg.lambda_safety, "channel": cfg.channel.name,
        "metadata_loss": cfg.metadata_loss,
    }


def summarise(df: pd.DataFrame, sweep_var: str) -> pd.DataFrame:
    out = []
    for (val, pol), g in df.groupby([sweep_var, "policy_name"]):
        out.append({
            sweep_var: val,
            "policy": pol,
            "n_windows": len(g),
            "loss_mean": g["loss_mean"].mean(),
            "loss_ci95": ci95(g["loss_mean"].to_numpy()),
            "rmse_mean": g["rmse_mean"].mean(),
            "rmse_ci95": ci95(g["rmse_mean"].to_numpy()),
            "missed_vio_mean": g["missed_violation_pct"].mean(),
            "missed_vio_ci95": ci95(g["missed_violation_pct"].to_numpy()),
        })
    return pd.DataFrame(out).sort_values([sweep_var, "policy"]).reset_index(drop=True)


def save(df: pd.DataFrame, summary: pd.DataFrame, sweep: str) -> None:
    d = PROC / sweep
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / f"{sweep}_raw.csv", index=False)
    DOCS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(DOCS / f"sensitivity_{sweep}.csv", index=False)
    print(f"[{sweep}] wrote {d / (sweep + '_raw.csv')}")
    print(f"[{sweep}] wrote {DOCS / ('sensitivity_' + sweep + '.csv')}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


def sweep_N(n_windows: int, horizon: int) -> None:
    ch = CHANNELS["severe_burst"]
    rows: list[dict] = []
    for N in (30, 50, 100):
        t0 = time.time()
        rows += run_cfg(Cfg(N, 4, 4, 6.0, ch, 0.0), n_windows, horizon, f"N={N}")
        print(f"[scalability_N_sweep] N={N} done in {time.time()-t0:.1f}s")
    df = pd.DataFrame(rows)
    save(df, summarise(df, "N"), "scalability_N_sweep")


def sweep_bprobe(n_windows: int, horizon: int) -> None:
    ch = CHANNELS["severe_burst"]
    rows: list[dict] = []
    for b in (2, 4, 6, 8):
        t0 = time.time()
        rows += run_cfg(Cfg(30, b, 4, 6.0, ch, 0.0), n_windows, horizon, f"Bp={b}")
        print(f"[bprobe_sensitivity] B_probe={b} done in {time.time()-t0:.1f}s")
    df = pd.DataFrame(rows)
    save(df, summarise(df, "b_probe"), "bprobe_sensitivity")


def sweep_lambda(n_windows: int, horizon: int) -> None:
    ch = CHANNELS["severe_burst"]
    rows: list[dict] = []
    for lam in (2.0, 4.0, 6.0, 8.0, 10.0):
        t0 = time.time()
        rows += run_cfg(Cfg(30, 4, 4, lam, ch, 0.0), n_windows, horizon, f"lam={lam}")
        print(f"[lambda_sensitivity] lambda={lam} done in {time.time()-t0:.1f}s")
    df = pd.DataFrame(rows)
    save(df, summarise(df, "lambda_safety"), "lambda_sensitivity")


def sweep_channel(n_windows: int, horizon: int) -> None:
    base = CHANNELS["severe_burst"]
    rows: list[dict] = []
    for pl in (0.1, 0.2, 0.3, 0.4):
        ch = make_loss_channel(base, pl)
        t0 = time.time()
        rows += run_cfg(Cfg(30, 4, 4, 6.0, ch, 0.0), n_windows, horizon, f"pl={pl}")
        # tag the sweep variable explicitly (channel.name is unique per pl)
        for r in rows[-2 * n_windows:]:
            r["packet_loss"] = pl
        print(f"[channel_loss_sweep] packet_loss={pl} done in {time.time()-t0:.1f}s")
    df = pd.DataFrame(rows)
    save(df, summarise(df, "packet_loss"), "channel_loss_sweep")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-windows", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=240)
    ap.add_argument("--sweeps", nargs="+",
                    default=["N", "bprobe", "lambda", "channel"])
    args = ap.parse_args()
    print(f"[sweep] n_windows={args.n_windows} horizon={args.horizon} "
          f"sweeps={args.sweeps}")
    t_all = time.time()
    if "N" in args.sweeps:
        sweep_N(args.n_windows, args.horizon)
    if "bprobe" in args.sweeps:
        sweep_bprobe(args.n_windows, args.horizon)
    if "lambda" in args.sweeps:
        sweep_lambda(args.n_windows, args.horizon)
    if "channel" in args.sweeps:
        sweep_channel(args.n_windows, args.horizon)
    print(f"[sweep] ALL DONE in {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
