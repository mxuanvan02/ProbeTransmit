#!/usr/bin/env python3
"""Debt-model ablation for ProbeTransmit (IoTJ 2026): deployed BOUNDED deficit vs
ACCUMULATING age-of-service debt.

Why this exists
---------------
Theorem 1 (Bounded Starvation, sections/05_theory.tex) proves a hard, closed-form
starvation deadline ONLY for the accumulating-debt model (age-of-service, grows
without bound while a node waits). The deployed scheduler uses the bounded
normalized deficit ``service_debt`` clipped to [0,1], which has NO horizon-
independent hard bound under adversarial urgency (see
``_probe_horizon_dep.py``: a fixed victim's worst gap grows ~linearly with system
age for every finite w). This script answers the deployment question the
reviewer will ask:

  "If the theorem needs accumulating debt but you ship bounded debt, do the two
   actually behave differently on the real task?"

We rerun the EXACT 30-window paired protocol of ``ablation_30windows.py`` (same
data, channel, seeds, budgets, horizon) for the full CAW-VoU policy under
``debt_mode='bounded'`` vs ``debt_mode='accumulating'`` and report a paired
comparison of safety/fairness/accuracy.

Honest reporting: numbers are whatever the real data produces. If the gap is
negligible, we recommend switching the default to accumulating debt so theory and
practice coincide; if accumulating debt hurts task performance, we keep bounded
as the deployed default and frame the theorem as the unbounded surrogate.

Output: docs/debt_mode_ablation_30windows.csv (per-window rows, 2 modes x 30).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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

# ---- Config: identical to ablation_30windows.py ----------------------------- #
DATA_PATH = ROOT / "data" / "raw" / "intel_berkeley" / "intel_panel_30motes.npy"
CHANNEL = CHANNELS["severe_burst"]
HORIZON = 240
N_WINDOWS = 30
TRAIN_LEN = 2000
SHRINKAGE = 0.1
LAMBDA_SAFETY = 6.0
W_DEBT = 0.05
B_PROBE = 4
B_PAYLOAD = 4
SEED_BASE = 50260
SEED_STEP = 17

MODES = ["bounded", "accumulating"]


def _mean_ci(x: np.ndarray, conf: float = 0.95) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(np.mean(x))
    if n < 2:
        return m, 0.0
    sem = float(np.std(x, ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(0.5 + conf / 2.0, df=n - 1))
    return m, tcrit * sem


def make_policy(R: np.ndarray) -> TwoStagePolicy:
    # Full CAW-VoU (corr + debt), the deployed configuration.
    return TwoStagePolicy(
        "caw_vou_full",
        CorrVoUProbe(corr=R, lambda_safety=LAMBDA_SAFETY, w_debt=W_DEBT,
                     use_correlation_credit=True),
        DebtAwarePayload(V=1.0),
    )


def run_ablation(data, ar, R):
    n = data.shape[1]
    starts = select_starts(len(data), HORIZON, N_WINDOWS)
    rows = []
    print(f"\n=== DEBT-MODE ABLATION 30 windows: N={n}, B_probe={B_PROBE}, "
          f"B_payload={B_PAYLOAD}, severe_burst, horizon={HORIZON} ===")
    for wi, start in enumerate(starts):
        seed = SEED_BASE + wi * SEED_STEP
        for mode in MODES:
            pol = make_policy(R)  # fresh policy; state carries debt_mode
            m = run_window(data=data, ar=ar, channel=CHANNEL, policy=pol,
                           start=start, horizon=HORIZON, seed=seed,
                           b_probe=B_PROBE, b_payload=B_PAYLOAD,
                           debt_mode=mode)
            m.update({"window_id": wi, "debt_mode": mode})
            rows.append(m)
        print(f"  window {wi+1}/{N_WINDOWS} done (seed={seed}, start={start})")
    return pd.DataFrame(rows)


def _rank_biserial(delta: np.ndarray) -> float:
    d = np.asarray(delta, dtype=float)
    d = d[d != 0.0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    total = ranks.sum()
    r_plus = ranks[d > 0].sum()
    r_minus = ranks[d < 0].sum()
    return float((r_plus - r_minus) / total)


def _paired_stats(df, variant, baseline, metric, key="debt_mode"):
    a = df[df[key] == variant].sort_values("window_id")[metric].to_numpy()
    b = df[df[key] == baseline].sort_values("window_id")[metric].to_numpy()
    delta = a - b  # variant - baseline
    win = int(np.sum(delta < 0))
    loss = int(np.sum(delta > 0))
    tie = int(np.sum(delta == 0))
    try:
        _, p_two = stats.wilcoxon(delta, alternative="two-sided")
    except ValueError:
        p_two = float("nan")
    return {
        "mean_delta": float(np.mean(delta)),
        "p_two_sided": float(p_two),
        "rank_biserial": _rank_biserial(delta),
        "win": win, "loss": loss, "tie": tie,
    }


def summarize(df):
    metrics = ["loss_mean", "missed_violation_pct", "rmse_mean", "max_age",
               "payload_fairness_jain", "probe_fairness_jain"]
    print("\n--- Per-mode mean +/- 95% CI (30 windows) ---")
    for mode in MODES:
        sub = df[df.debt_mode == mode]
        parts = []
        for mt in metrics:
            m, c = _mean_ci(sub[mt].to_numpy())
            parts.append(f"{mt}={m:.5g}+/-{c:.4g}")
        print(f"  {mode:13s} " + "  ".join(parts))

    baseline = "bounded"  # deployed default
    print(f"\n--- Paired tests: accumulating - {baseline} "
          f"(delta = accumulating - bounded) ---")
    for mt in metrics:
        s = _paired_stats(df, "accumulating", baseline, mt)
        print(f"    {mt:24s} dmean={s['mean_delta']:+.5g}  "
              f"p2={s['p_two_sided']:.4f}  rb={s['rank_biserial']:+.3f}  "
              f"W/L/T(acc lower)={s['win']}/{s['loss']}/{s['tie']}")

    print("\n--- RECOMMENDATION HEURISTIC ---")
    # safety + accuracy gap small AND fairness comparable -> switch default
    s_miss = _paired_stats(df, "accumulating", baseline, "missed_violation_pct")
    s_loss = _paired_stats(df, "accumulating", baseline, "loss_mean")
    s_rmse = _paired_stats(df, "accumulating", baseline, "rmse_mean")
    bnd_miss = df[df.debt_mode == "bounded"]["missed_violation_pct"].mean()
    acc_miss = df[df.debt_mode == "accumulating"]["missed_violation_pct"].mean()
    rel_miss = abs(acc_miss - bnd_miss) / max(bnd_miss, 1e-9)
    print(f"  missed-violation: bounded={bnd_miss:.4g}%  accumulating={acc_miss:.4g}%  "
          f"rel.gap={rel_miss*100:.1f}%  (p2={s_miss['p_two_sided']:.4f})")
    print(f"  loss_mean p2={s_loss['p_two_sided']:.4f}; rmse_mean p2={s_rmse['p_two_sided']:.4f}")
    sig = (s_miss["p_two_sided"] < 0.05) or (s_loss["p_two_sided"] < 0.05) or (s_rmse["p_two_sided"] < 0.05)
    if not sig and rel_miss < 0.10:
        print("  => Task performance statistically indistinguishable. SWITCH default to")
        print("     accumulating debt so the deployed scheduler matches Theorem 1 exactly.")
    else:
        print("  => Modes differ on the real task. KEEP bounded deficit as deployed default;")
        print("     frame the accumulating model as the analyzed unbounded surrogate and")
        print("     recommend it only where a contractual per-node deadline is required.")


def main():
    print("Loading REAL panel:", DATA_PATH)
    data = np.load(DATA_PATH)
    print("panel shape:", data.shape)
    train = data[:TRAIN_LEN]
    ar = AR1Model.fit(train)
    ar.set_empirical_residuals(train[1:] - (train[:-1] * ar.alpha + ar.beta))
    R = fit_correlation(train, shrinkage=SHRINKAGE)
    offdiag = R[~np.eye(R.shape[0], dtype=bool)]
    print(f"correlation R off-diagonal: mean={offdiag.mean():.3f}, "
          f"max={offdiag.max():.3f}, min={offdiag.min():.3f}")

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    df = run_ablation(data, ar, R)
    out = docs / "debt_mode_ablation_30windows.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved {out}")
    summarize(df)
    print("\nDONE.")


if __name__ == "__main__":
    main()
