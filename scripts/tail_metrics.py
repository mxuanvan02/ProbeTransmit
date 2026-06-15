#!/usr/bin/env python3
"""Tail / worst-case reliability metrics: CAW-VoU vs heuristic Whittle.

Reads the existing 30-window paired comparison
(docs/whittle_comparison_30windows.csv) and computes the percentile / tail
panel used in the "Reliability under stress" subsection of the evaluation:
P50, P90, P99, max for Loss and Missed-violation, plus the worst-case ratios.

Also emits a CDF figure of per-window loss and missed-violation across the
30 windows (figures/tail_cdf.png).

Honest reporting: all numbers come straight from the paired per-window CSV.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIGS = ROOT.parent / "manuscript" / "figures"


def _pct(a, q):
    return float(np.percentile(np.asarray(a, float), q))


def tail_table(caw, wht, col):
    a, b = caw[col].to_numpy(), wht[col].to_numpy()
    rows = []
    for label, q in [("P50", 50), ("P90", 90), ("P99", 99)]:
        rows.append((label, _pct(a, q), _pct(b, q)))
    rows.append(("max", float(a.max()), float(b.max())))
    rows.append(("mean", float(a.mean()), float(b.mean())))
    return rows


def main():
    df = pd.read_csv(DOCS / "whittle_comparison_30windows.csv")
    caw = df[df.policy == "CAW-VoU"].sort_values("window_id").reset_index(drop=True)
    wht = df[df.policy == "Whittle"].sort_values("window_id").reset_index(drop=True)
    n = len(caw)
    print(f"Windows: {n} (paired)\n")

    out_rows = []
    for metric, col in [("Loss", "loss"), ("Missed-vio (%)", "missed_vio")]:
        print(f"=== {metric} ===")
        rows = tail_table(caw, wht, col)
        for label, cv, wv in rows:
            ratio = wv / cv if cv > 1e-12 else float("inf")
            print(f"  {label:5s} CAW={cv:.4g}  Whittle={wv:.4g}  ratio={ratio:.1f}x")
            out_rows.append({"metric": metric, "stat": label,
                             "CAW_VoU": cv, "Whittle": wv,
                             "ratio_Whittle_over_CAW": ratio})
        print()

    pd.DataFrame(out_rows).to_csv(DOCS / "tail_metrics_30windows.csv", index=False)
    print(f"saved {DOCS/'tail_metrics_30windows.csv'}")

    # ----- CDF figure -----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})

    def cdf(ax, a, b, xlabel, logx=False):
        for data, lab, col, ls in [(a, "CAW-VoU", "#1f4e79", "-"),
                                    (b, "Whittle", "#c00000", "--")]:
            xs = np.sort(np.asarray(data, float))
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.step(xs, ys, where="post", color=col, ls=ls, lw=1.6, label=lab)
        ax.set_xlabel(xlabel); ax.set_ylabel("Empirical CDF")
        if logx:
            ax.set_xscale("symlog", linthresh=1e-3)
        ax.legend(frameon=False, loc="lower right")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    cdf(ax1, caw.loss.values, wht.loss.values, "Safety-critical loss", logx=True)
    ax1.set_title("(a) Loss CDF (30 windows)", fontsize=9)
    cdf(ax2, caw.missed_vio.values, wht.missed_vio.values,
        "Missed-violation rate (%)", logx=True)
    ax2.set_title("(b) Missed-violation CDF (30 windows)", fontsize=9)
    fig.suptitle("Reliability under stress: tail behavior across 30 windows "
                 "(CAW-VoU vs.\\ heuristic Whittle)", fontsize=9.5)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "tail_cdf.png")
    plt.close(fig)
    print(f"saved {FIGS/'tail_cdf.png'}")


if __name__ == "__main__":
    main()
