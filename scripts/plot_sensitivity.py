#!/usr/bin/env python3
"""Plot the four sensitivity/scalability sweeps for the ProbeTransmit paper.

Reads the tidy summaries written by ``sensitivity_scalability_sweep.py``
(code/docs/sensitivity_<sweep>.csv) and produces clean line plots with 95% CI
error bars, comparing CAW-VoU vs VoU.

Figures saved to manuscript/figures/sensitivity_*.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIGDIR = ROOT.parent / "manuscript" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

STYLE = {
    "CAW-VoU": {"color": "#1b6ca8", "marker": "o", "ls": "-"},
    "VoU": {"color": "#c44e52", "marker": "s", "ls": "--"},
}


def _line(ax, df, xvar, ymean, yci, ylabel, xlabel, title):
    for pol, g in df.groupby("policy"):
        g = g.sort_values(xvar)
        st = STYLE.get(pol, {})
        ax.errorbar(g[xvar], g[ymean], yerr=g[yci],
                    label=pol, capsize=3, markersize=6, linewidth=1.8, **st)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=True)


def plot_sweep(sweep, xvar, xlabel, panels, fname):
    path = DOCS / f"sensitivity_{sweep}.csv"
    if not path.exists():
        print(f"[skip] {path} not found")
        return
    df = pd.read_csv(path)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.0))
    if n == 1:
        axes = [axes]
    for ax, (ymean, yci, ylabel, title) in zip(axes, panels):
        _line(ax, df, xvar, ymean, yci, ylabel, xlabel, title)
    fig.tight_layout()
    out = FIGDIR / fname
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig] wrote {out}")


def main():
    # 1. N scalability: Loss, RMSE, Missed Vio vs N.
    plot_sweep(
        "scalability_N_sweep", "N", r"Network size $N$",
        [
            ("loss_mean", "loss_ci95", "Safety-critical loss", "Loss vs $N$"),
            ("rmse_mean", "rmse_ci95", "RMSE", "RMSE vs $N$"),
            ("missed_vio_mean", "missed_vio_ci95", "Missed violation (%)", "Missed Vio. vs $N$"),
        ],
        "sensitivity_scalability_N.png",
    )
    # 2. B_probe: Loss vs B_probe.
    plot_sweep(
        "bprobe_sensitivity", "b_probe", r"Probe budget $B_{\mathrm{probe}}$",
        [
            ("loss_mean", "loss_ci95", "Safety-critical loss", r"Loss vs $B_{\mathrm{probe}}$"),
            ("missed_vio_mean", "missed_vio_ci95", "Missed violation (%)", r"Missed Vio. vs $B_{\mathrm{probe}}$"),
        ],
        "sensitivity_bprobe.png",
    )
    # 3. lambda_safety: Loss, Missed Vio vs lambda.
    plot_sweep(
        "lambda_sensitivity", "lambda_safety", r"Safety weight $\lambda_{\mathrm{safety}}$",
        [
            ("loss_mean", "loss_ci95", "Safety-critical loss", r"Loss vs $\lambda_{\mathrm{safety}}$"),
            ("missed_vio_mean", "missed_vio_ci95", "Missed violation (%)", r"Missed Vio. vs $\lambda_{\mathrm{safety}}$"),
        ],
        "sensitivity_lambda.png",
    )
    # 4. channel loss: Loss vs packet_loss.
    plot_sweep(
        "channel_loss_sweep", "packet_loss", r"Packet-loss rate",
        [
            ("loss_mean", "loss_ci95", "Safety-critical loss", "Loss vs packet loss"),
            ("missed_vio_mean", "missed_vio_ci95", "Missed violation (%)", "Missed Vio. vs packet loss"),
        ],
        "sensitivity_channel_loss.png",
    )


if __name__ == "__main__":
    main()
