#!/usr/bin/env python3
"""theory_validation.py

Numerical validation of the theoretical analysis of the debt-aware fairness
mechanism in CAW-VoU (ProbeTransmit, IoTJ 2026).

We validate two analytic results:

  Theorem 1 (Bounded starvation).  Under the *accumulating-debt* scheduler that
  selects, each slot, the B nodes of highest score
        score_i(t) = u_i(t) + w * d_i(t),     u_i in [0, V_max],
  with debt d_i = number of slots since node i was last selected (reset to 0 on
  service), the worst-case time between consecutive services of any node obeys
        T_max  <=  ceil(N / B) + ceil(V_max / w) + 1.
  The first term is the round-robin floor; the second is the urgency-induced slack.

  Theorem 2 (Fairness convergence).  As w -> infinity the policy reduces to
  longest-since-service-first (deficit round robin); per-node service counts then
  differ by at most one and the Jain fairness index obeys
        J  >=  1 / (1 + (N / (2 B T))^2)  ->  1.

We additionally contrast the *bounded normalized deficit* debt actually deployed
in policies.py (service_debt = max(target_share - share, 0)/target_share, in [0,1])
to make explicit that the hard guarantee belongs to the accumulating model while
the deployed surrogate is a soft approximation.

Outputs:
  - manuscript/figures/theory_validation.png
  - prints a numeric summary (used verbatim in 05_theory.tex)

No external repo modules required (numpy + matplotlib only).
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DIR = os.path.join(REPO, "manuscript", "figures")


# --------------------------------------------------------------------------- #
# Schedulers
# --------------------------------------------------------------------------- #
def simulate_accumulating(
    N: int,
    B: int,
    w: float,
    V_max: float,
    horizon: int,
    rng: np.random.Generator,
    adversarial: bool = True,
) -> dict:
    """Accumulating-debt scheduler (the *analyzed* model).

    debt = slots since last service (reset to 0 when served). Each slot select
    the B nodes of highest score u_i + w*debt_i. Track max inter-service gap.

    adversarial=True drives the worst case for Theorem 1: at every slot the
    node with the *largest* debt is assigned urgency 0 while all competitors get
    urgency V_max, maximally fighting the fairness term.
    """
    debt = np.zeros(N, dtype=float)          # slots since last service (=age)
    last_service = np.full(N, -1, dtype=int)  # slot index of last service
    max_gap = 0
    # service counts for Jain
    counts = np.zeros(N, dtype=int)

    for t in range(horizon):
        if adversarial:
            # worst case: starve the current most-deprived node
            u = np.full(N, V_max, dtype=float)
            victim = int(np.argmax(debt))
            u[victim] = 0.0
        else:
            u = rng.uniform(0.0, V_max, size=N)

        score = u + w * debt
        # top-B by score (ties broken by larger debt, then index -> stable)
        order = np.lexsort((np.arange(N), -debt, -score))
        selected = order[:B]

        # record inter-service gaps for newly served nodes
        for i in selected:
            if last_service[i] >= 0:
                gap = t - last_service[i]
                if gap > max_gap:
                    max_gap = gap
            last_service[i] = t
        counts[selected] += 1

        # debt update: served -> 0, others +1
        mask = np.zeros(N, dtype=bool)
        mask[selected] = True
        debt = np.where(mask, 0.0, debt + 1.0)

    # also account for nodes still unserved at the end (lower bound on gap)
    for i in range(N):
        if last_service[i] >= 0:
            tail = (horizon - 1) - last_service[i]
            if tail > max_gap:
                max_gap = tail

    bound = math.ceil(N / B) + math.ceil(V_max / w)
    mean_c = counts.mean()
    jain = (counts.sum() ** 2) / (N * np.sum(counts.astype(float) ** 2)) if counts.sum() > 0 else 0.0
    return {
        "max_gap": int(max_gap),
        "bound": int(bound),
        "rr_floor": math.ceil(N / B),
        "slack": math.ceil(V_max / w),
        "jain": float(jain),
        "count_spread": int(counts.max() - counts.min()),
        "mean_count": float(mean_c),
    }


def simulate_bounded_deficit(
    N: int,
    B: int,
    w: float,
    V_max: float,
    horizon: int,
    rng: np.random.Generator,
) -> dict:
    """Bounded normalized-deficit scheduler (the *deployed* surrogate).

    Mirrors service_debt() in policies.py:
        debt_i = max(target_share - counts_i/total, 0) / target_share  in [0,1]
        target_share = 1/N (each individual selection's fair share).
    score_i = u_i + w*debt_i, urgency uniform in [0, V_max].
    Reports Jain index of service counts (debt is bounded, so no hard gap bound).
    """
    counts = np.zeros(N, dtype=float)
    total = 0
    target_share = 1.0 / N
    last_service = np.full(N, -1, dtype=int)
    max_gap = 0
    # Persistent heterogeneous urgency profile: a minority of "hot" nodes carry
    # structurally higher urgency, reproducing the starvation regime that debt
    # must correct (cf. probe-Jain 0.33 -> 0.97 in the ablation).
    base_u = rng.uniform(0.0, V_max, size=N)
    base_u[: max(1, N // 5)] = V_max  # hot minority pinned near the top

    for t in range(horizon):
        if total > 0:
            share = counts / total
            debt = np.maximum(target_share - share, 0.0) / target_share
        else:
            debt = np.zeros(N)
        u = 0.7 * base_u + 0.3 * rng.uniform(0.0, V_max, size=N)
        score = u + w * debt
        order = np.lexsort((np.arange(N), -debt, -score))
        selected = order[:B]
        for i in selected:
            if last_service[i] >= 0:
                gap = t - last_service[i]
                if gap > max_gap:
                    max_gap = gap
            last_service[i] = t
        counts[selected] += 1
        total += B

    jain = (counts.sum() ** 2) / (N * np.sum(counts ** 2)) if counts.sum() > 0 else 0.0
    return {"jain": float(jain), "max_gap": int(max_gap)}


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def sweep_wdebt(N, B, V_max, horizon, seed=0):
    rng = np.random.default_rng(seed)
    ws = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
    emp, bnd, jain_acc, jain_bd = [], [], [], []
    for w in ws:
        r = simulate_accumulating(N, B, w, V_max, horizon, np.random.default_rng(seed), adversarial=True)
        b = simulate_bounded_deficit(N, B, w, V_max, horizon, np.random.default_rng(seed + 1))
        emp.append(r["max_gap"])
        bnd.append(r["bound"])
        jain_acc.append(r["jain"])
        jain_bd.append(b["jain"])
    # w=0 fairness baseline (no debt -> structural starvation), for annotation only
    j0 = simulate_bounded_deficit(N, B, 0.0, V_max, horizon, np.random.default_rng(seed + 1))["jain"]
    return ws, np.array(emp), np.array(bnd), np.array(jain_acc), np.array(jain_bd), float(j0)


def sweep_N(B, w, V_max, horizon, seed=0):
    Ns = np.array([10, 20, 30, 50, 80, 120])
    emp, bnd = [], []
    for N in Ns:
        r = simulate_accumulating(int(N), B, w, V_max, horizon, np.random.default_rng(seed), adversarial=True)
        emp.append(r["max_gap"])
        bnd.append(r["bound"])
    return Ns, np.array(emp), np.array(bnd)


def sweep_B(N, w, V_max, horizon, seed=0):
    Bs = np.array([2, 4, 6, 8, 10])
    emp, bnd = [], []
    for B in Bs:
        r = simulate_accumulating(N, int(B), w, V_max, horizon, np.random.default_rng(seed), adversarial=True)
        emp.append(r["max_gap"])
        bnd.append(r["bound"])
    return Bs, np.array(emp), np.array(bnd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--Vmax", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    V_max = args.Vmax
    H = args.horizon

    # --- Panel data ---------------------------------------------------------
    ws, emp_w, bnd_w, jain_acc, jain_bd, j0 = sweep_wdebt(N=30, B=4, V_max=V_max, horizon=H, seed=args.seed)
    Ns, emp_N, bnd_N = sweep_N(B=4, w=0.2, V_max=V_max, horizon=H, seed=args.seed)
    Bs, emp_B, bnd_B = sweep_B(N=30, w=0.2, V_max=V_max, horizon=H, seed=args.seed)

    # sanity: empirical must never exceed the analytic bound
    viol_w = int(np.sum(emp_w > bnd_w))
    viol_N = int(np.sum(emp_N > bnd_N))
    viol_B = int(np.sum(emp_B > bnd_B))

    # --- Figure -------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # (a) starvation vs w_debt
    ax[0].plot(ws, bnd_w, "s--", color="#c0392b", label=r"Bound $\lceil N/B\rceil+\lceil V_{\max}/w\rceil$")
    ax[0].plot(ws, emp_w, "o-", color="#2c3e50", label="Empirical max gap (adversarial)")
    ax[0].axhline(math.ceil(30 / 4), ls=":", color="#7f8c8d", label=r"Round-robin floor $\lceil N/B\rceil$")
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"Debt weight $w_{\mathrm{debt}}$")
    ax[0].set_ylabel("Max steps between services")
    ax[0].set_title(r"(a) Bounded starvation vs $w_{\mathrm{debt}}$ ($N{=}30,B{=}4$)")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].grid(alpha=0.3)

    # (b) starvation vs N and vs B (share axis via twin)
    ax[1].plot(Ns, bnd_N, "s--", color="#c0392b", label=r"Bound vs $N$")
    ax[1].plot(Ns, emp_N, "o-", color="#2c3e50", label=r"Empirical vs $N$")
    ax[1].set_xlabel(r"Network size $N$ ($B{=}4,w{=}0.2$)")
    ax[1].set_ylabel("Max steps between services")
    ax[1].set_title("(b) Linear-in-$N$ floor; bound holds")
    ax[1].legend(fontsize=8, loc="upper left")
    ax[1].grid(alpha=0.3)

    # (c) Jain convergence vs w
    ax[2].plot(ws, jain_acc, "o-", color="#27ae60", label="Jain (accumulating debt)")
    ax[2].plot(ws, jain_bd, "^-", color="#2980b9", label="Jain (bounded deficit, deployed)")
    ax[2].axhline(j0, ls="--", color="#e67e22", label=fr"No debt ($w{{=}}0$): Jain$={j0:.2f}$")
    ax[2].axhline(1.0, ls=":", color="#7f8c8d", label=r"Round-robin limit $\to 1$")
    ax[2].set_xscale("log")
    ax[2].set_ylim(0.0, 1.02)
    ax[2].set_xlabel(r"Debt weight $w_{\mathrm{debt}}$")
    ax[2].set_ylabel("Jain fairness index")
    ax[2].set_title(r"(c) Fairness convergence as $w_{\mathrm{debt}}\to\infty$")
    ax[2].legend(fontsize=8, loc="lower right")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "theory_validation.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"[fig] wrote {out}")

    # --- Numeric summary (for the manuscript) -------------------------------
    print("\n===== THEORY VALIDATION SUMMARY =====")
    print(f"V_max={V_max}, horizon={H}, seed={args.seed}")
    print("\n--- Theorem 1: bound vs empirical (w sweep, N=30, B=4) ---")
    for w, e, b in zip(ws, emp_w, bnd_w):
        print(f"  w={w:6.2f}  empirical_max_gap={e:4d}  bound={b:4d}  ok={e<=b}")
    print(f"  bound violations (w-sweep): {viol_w}")
    print("\n--- Theorem 1: N sweep (B=4, w=0.2) ---")
    for n, e, b in zip(Ns, emp_N, bnd_N):
        print(f"  N={n:4d}  empirical={e:4d}  bound={b:4d}  rr_floor={math.ceil(n/4)}  ok={e<=b}")
    print(f"  bound violations (N-sweep): {viol_N}")
    print("\n--- Theorem 1: B sweep (N=30, w=0.2) ---")
    for bb, e, b in zip(Bs, emp_B, bnd_B):
        print(f"  B={bb:3d}  empirical={e:4d}  bound={b:4d}  ok={e<=b}")
    print(f"  bound violations (B-sweep): {viol_B}")
    print("\n--- Theorem 2: Jain vs w (N=30, B=4) ---")
    print(f"  w=0.00 (no debt baseline)  Jain_bounded={j0:.4f}")
    for w, ja, jb in zip(ws, jain_acc, jain_bd):
        print(f"  w={w:6.2f}  Jain_accum={ja:.4f}  Jain_bounded={jb:.4f}")
    print(f"\n  Jain(accum) at w=0.05 -> {jain_acc[0]:.4f};  at w=50 -> {jain_acc[-1]:.4f}")
    print(f"  Jain(bounded) at w=0.05 -> {jain_bd[0]:.4f};  at w=50 -> {jain_bd[-1]:.4f}")
    total_viol = viol_w + viol_N + viol_B
    print(f"\nTOTAL BOUND VIOLATIONS ACROSS ALL SWEEPS: {total_viol} (expect 0)")
    print("=====================================")


if __name__ == "__main__":
    main()
