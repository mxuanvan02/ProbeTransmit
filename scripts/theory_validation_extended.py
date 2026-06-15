#!/usr/bin/env python3
"""theory_validation_extended.py

Extended numerical validation of the debt-aware starvation theory for
ProbeTransmit / CAW-VoU (IoTJ 2026). This supersedes the single-row
``theory_validation.py`` figure with a 2x3 panel that matches every theoretical
CLAIM in sections/05_theory.tex to a corresponding picture:

  Panel (a)  Two-sided tightness of the accumulating-debt bound (Thm 1 + 2).
             Empirical adversarial max-gap vs w with BOTH the upper bound
             U = ceil(N/B)+ceil(Vmax/w) and the matching lower bound U-2; the
             admissible band [U-2, U] is shaded. Closes the gap left by the old
             figure which drew only the upper bound.

  Panel (b)  Linear-in-N round-robin floor (Thm 1); bound stays tight as N grows.

  Panel (c)  Fairness convergence (Thm 3 / thm:fairness). Jain index of the
             deployed bounded-deficit scheduler rises 0.24 -> 1.0 as w grows;
             the accumulating variant is fair across the whole range. (Kept at
             slot (c) so existing Fig.~\\ref{...}c references stay valid.)

  Panel (d)  Deployed bounded-deficit starvation is age-proportional Theta(t0)
             (Prop 1 & 2). Worst-case victim wait vs system age t0 for several
             w>Vmax, overlaid with the analytic line tau_lb(t0) and the
             +O(ceil(N/B)) upper band.

  Panel (e)  Theory MEETS the real Intel Berkeley trace. Box distribution of
             per-window max service age over the 30 matched windows for the
             debt-bearing vs debt-free schedulers and bounded vs accumulating
             debt, with the analytic accumulating-debt deadline U as reference.

  Panel (f)  Worst-case vs typical (Scenario #3). Under stochastic urgency the
             realised gap sits far below the adversarial bound: the bound is a
             safety ceiling, not the common case. Ratio gap/U vs w.

Outputs:
  - manuscript/figures/theory_validation.png      (regenerated, 2x3)
  - prints a numeric summary used verbatim in 05_theory.tex / captions.

Real-trace inputs (already produced by the 30-window ablations; not recomputed):
  - code/docs/debt_mode_ablation_30windows.csv   (bounded vs accumulating)
  - code/docs/ablation_results_30windows.csv     (+debt vs -debt variants)
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
DOCS = os.path.join(REPO, "code", "docs")


# --------------------------------------------------------------------------- #
# Schedulers (numpy-only, scheduler-faithful)
# --------------------------------------------------------------------------- #
def sim_accumulating(N, B, w, V_max, horizon, rng, adversarial=True):
    """Accumulating-debt top-B scheduler. Returns worst inter-service gap + Jain."""
    debt = np.zeros(N)
    last = np.full(N, -1, dtype=int)
    counts = np.zeros(N, dtype=int)
    max_gap = 0
    for t in range(horizon):
        if adversarial:
            u = np.full(N, V_max)
            u[int(np.argmax(debt))] = 0.0
        else:
            u = rng.uniform(0.0, V_max, size=N)
        score = u + w * debt
        sel = np.lexsort((np.arange(N), -debt, -score))[:B]
        for i in sel:
            if last[i] >= 0:
                max_gap = max(max_gap, t - last[i])
            last[i] = t
        counts[sel] += 1
        m = np.zeros(N, dtype=bool)
        m[sel] = True
        debt = np.where(m, 0.0, debt + 1.0)
    for i in range(N):
        if last[i] >= 0:
            max_gap = max(max_gap, (horizon - 1) - last[i])
    jain = (counts.sum() ** 2) / (N * np.sum(counts.astype(float) ** 2)) if counts.sum() else 0.0
    return {"max_gap": int(max_gap), "jain": float(jain)}


def sim_bounded_deficit_jain(N, B, w, V_max, horizon, rng):
    """Deployed bounded normalized-deficit scheduler under a persistently
    heterogeneous urgency profile (a hot minority pinned at V_max). Returns the
    Jain index of cumulative service counts. Mirrors the old
    theory_validation.py construction so the reported 0.24->1.0 curve is stable."""
    counts = np.zeros(N, dtype=float)
    total = 0
    target_share = 1.0 / N
    base_u = rng.uniform(0.0, V_max, size=N)
    base_u[: max(1, N // 5)] = V_max  # hot minority pinned near the top
    for _ in range(horizon):
        if total > 0:
            share = counts / total
            debt = np.maximum(target_share - share, 0.0) / target_share
        else:
            debt = np.zeros(N)
        u = 0.7 * base_u + 0.3 * rng.uniform(0.0, V_max, size=N)
        score = u + w * debt
        sel = np.lexsort((np.arange(N), -debt, -score))[:B]
        counts[sel] += 1
        total += B
    return (counts.sum() ** 2) / (N * np.sum(counts ** 2)) if counts.sum() else 0.0


def sim_bounded_fixed_victim_aged(N, B, w, V_max, t0, seed=7, max_extra=400000):
    """Deployed bounded-deficit scheduler: warm up to age t0 (balanced), then run
    the strongest fixed-victim adversary and measure the victim's wait."""
    rng = np.random.default_rng(seed)
    counts = np.zeros(N, dtype=float)
    s_star = 1.0 / N
    for _ in range(t0):  # balanced warmup with fair random urgency
        total = counts.sum()
        debt = np.maximum(s_star - counts / total, 0.0) * N if total > 0 else np.zeros(N)
        u = rng.uniform(0.0, V_max, size=N)
        sel = np.lexsort((np.arange(N), -debt, -(u + w * debt)))[:B]
        counts[sel] += 1
    victim = int(np.argmin(counts))
    gap = 0
    for _ in range(max_extra):
        total = counts.sum()
        debt = np.maximum(s_star - counts / total, 0.0) * N
        u = np.full(N, V_max)
        u[victim] = 0.0
        sel = np.lexsort((np.arange(N), -debt, -(u + w * debt)))[:B]
        if victim in sel:
            break
        gap += 1
        counts[sel] += 1
    return gap


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #
def sweep_w(N, B, V_max, horizon, seed):
    ws = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
    emp_adv, emp_sto, U, jain_acc, jain_bd = [], [], [], [], []
    for w in ws:
        ra = sim_accumulating(N, B, w, V_max, horizon, np.random.default_rng(seed), adversarial=True)
        rs = sim_accumulating(N, B, w, V_max, horizon, np.random.default_rng(seed + 99), adversarial=False)
        jb = sim_bounded_deficit_jain(N, B, w, V_max, horizon, np.random.default_rng(seed + 1))
        emp_adv.append(ra["max_gap"])
        emp_sto.append(rs["max_gap"])
        jain_acc.append(ra["jain"])
        jain_bd.append(jb)
        U.append(math.ceil(N / B) + math.ceil(V_max / w))
    j0 = sim_bounded_deficit_jain(N, B, 0.0, V_max, horizon, np.random.default_rng(seed + 1))
    return (ws, np.array(emp_adv), np.array(emp_sto), np.array(U),
            np.array(jain_acc), np.array(jain_bd), float(j0))


def sweep_N(B, w, V_max, horizon, seed):
    Ns = np.array([10, 20, 30, 50, 80, 120])
    emp, U = [], []
    for N in Ns:
        r = sim_accumulating(int(N), B, w, V_max, horizon, np.random.default_rng(seed), adversarial=True)
        emp.append(r["max_gap"])
        U.append(math.ceil(N / B) + math.ceil(V_max / w))
    return Ns, np.array(emp), np.array(U)


def sweep_t0(N, B, V_max, ws, t0s, seed=7):
    out = {}
    for w in ws:
        out[w] = np.array([sim_bounded_fixed_victim_aged(N, B, w, V_max, int(t0), seed=seed) for t0 in t0s])
    return out


# --------------------------------------------------------------------------- #
# Real-trace loaders
# --------------------------------------------------------------------------- #
def load_real_trace():
    """Return dicts of per-window max_age arrays from the 30-window ablations."""
    import csv

    def col(path, key, fcol):
        vals = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                grp = row.get(fcol)
                if grp is None:
                    continue
                try:
                    vals.setdefault(grp, []).append(float(row[key]))
                except (ValueError, KeyError):
                    continue
        return vals

    debt_mode, variants = {}, {}
    p1 = os.path.join(DOCS, "debt_mode_ablation_30windows.csv")
    if os.path.exists(p1):
        debt_mode = col(p1, "max_age", "debt_mode")
    p2 = os.path.join(DOCS, "ablation_results_30windows.csv")
    if os.path.exists(p2):
        variants = col(p2, "max_age", "variant")
    return debt_mode, variants


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--Vmax", type=float, default=1.0)
    args = ap.parse_args()
    V, H, seed = args.Vmax, args.horizon, args.seed
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---- data --------------------------------------------------------------
    ws, adv, sto, U, jain_acc, jain_bd, j0 = sweep_w(30, 4, V, H, seed)
    Ns, empN, UN = sweep_N(4, 0.2, V, H, seed)
    t0s = np.array([250, 500, 1000, 2000, 4000, 8000])
    ws_dep = [1.5, 2.0, 5.0]
    dep = sweep_t0(30, 4, V, ws_dep, t0s, seed=seed)
    debt_mode, variants = load_real_trace()

    # ---- figure (2x3) ------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.0))
    a, b, c = ax[0, 0], ax[0, 1], ax[0, 2]
    d, e, f = ax[1, 0], ax[1, 1], ax[1, 2]

    # (a) two-sided tightness + stochastic overlay
    a.fill_between(ws, U - 2, U, color="#c0392b", alpha=0.12, label=r"Admissible band $[U-2,\,U]$")
    a.plot(ws, U, "s--", color="#c0392b", label=r"Upper bound $U=\lceil N/B\rceil+\lceil V_{\max}/w\rceil$")
    a.plot(ws, U - 2, "v:", color="#8e44ad", label=r"Lower bound $U-2$ (Thm.~2)")
    a.plot(ws, adv, "o-", color="#2c3e50", label="Empirical (adversarial)")
    a.plot(ws, sto, "D-", color="#16a085", alpha=0.85, label="Empirical (stochastic)")
    a.axhline(math.ceil(30 / 4), ls=":", color="#7f8c8d", lw=1)
    a.set_xscale("log")
    a.set_xlabel(r"Debt weight $w_{\mathrm{debt}}$")
    a.set_ylabel("Max steps between services")
    a.set_title(r"(a) Two-sided tightness ($N{=}30,B{=}4$)")
    a.legend(fontsize=7.5, loc="upper right")
    a.grid(alpha=0.3)

    # (b) linear-in-N floor
    b.fill_between(Ns, UN - 2, UN, color="#c0392b", alpha=0.12)
    b.plot(Ns, UN, "s--", color="#c0392b", label=r"Upper bound $U$")
    b.plot(Ns, UN - 2, "v:", color="#8e44ad", label=r"Lower bound $U-2$")
    b.plot(Ns, empN, "o-", color="#2c3e50", label="Empirical (adversarial)")
    b.set_xlabel(r"Network size $N$ ($B{=}4,\,w{=}0.2$)")
    b.set_ylabel("Max steps between services")
    b.set_title("(b) Linear-in-$N$ round-robin floor")
    b.legend(fontsize=8, loc="upper left")
    b.grid(alpha=0.3)

    # (c) Jain fairness convergence  (referenced as Fig...c in the text)
    c.plot(ws, jain_acc, "o-", color="#27ae60", label="Jain (accumulating debt)")
    c.plot(ws, jain_bd, "^-", color="#2980b9", label="Jain (bounded deficit, deployed)")
    c.axhline(j0, ls="--", color="#e67e22", label=fr"No debt ($w{{=}}0$): Jain$={j0:.2f}$")
    c.axhline(1.0, ls=":", color="#7f8c8d", label=r"Round-robin limit $\to 1$")
    c.set_xscale("log")
    c.set_ylim(0.0, 1.03)
    c.set_xlabel(r"Debt weight $w_{\mathrm{debt}}$")
    c.set_ylabel("Jain fairness index")
    c.set_title(r"(c) Fairness convergence as $w_{\mathrm{debt}}\to\infty$")
    c.legend(fontsize=7.5, loc="lower right")
    c.grid(alpha=0.3)

    # (d) deployed Theta(t0)
    colors = {1.5: "#2980b9", 2.0: "#e67e22", 5.0: "#27ae60"}
    rr = math.ceil(30 / 4)
    for w in ws_dep:
        tau_lb = t0s * (V / w) / (1 - V / w)
        d.plot(t0s, dep[w], "o-", color=colors[w], label=fr"Empirical $w={w}$")
        d.plot(t0s, tau_lb, "--", color=colors[w], alpha=0.6)
        d.fill_between(t0s, tau_lb, tau_lb + 3 * rr, color=colors[w], alpha=0.08)
    d.plot([], [], "k--", alpha=0.6, label=r"Analytic $\tau_{\mathrm{lb}}(t_0)$")
    d.plot([], [], color="k", alpha=0.15, lw=8, label=r"$+\,O(\lceil N/B\rceil)$ band")
    d.set_xlabel(r"System age $t_0$ (slots)")
    d.set_ylabel("Worst-case victim wait")
    d.set_title(r"(d) Deployed deficit is $\Theta(t_0)$ (Prop.~1,2)")
    d.legend(fontsize=8, loc="upper left")
    d.grid(alpha=0.3)

    # (e) real Intel trace: max_age distributions
    box_data, labels, box_colors = [], [], []
    order = [
        ("+debt -corr", "+debt", "#27ae60"),
        ("Full (corr+debt)", "Full", "#16a085"),
        ("+corr -debt", "$-$debt", "#c0392b"),
        ("VoU-only (R=I)", "VoU-only", "#e74c3c"),
    ]
    for key, lab, cc in order:
        if key in variants:
            box_data.append(variants[key])
            labels.append(lab)
            box_colors.append(cc)
    for key, lab, cc in [("bounded", "bounded", "#2980b9"), ("accumulating", "accum.", "#8e44ad")]:
        if key in debt_mode:
            box_data.append(debt_mode[key])
            labels.append(lab)
            box_colors.append(cc)
    if box_data:
        bp = e.boxplot(box_data, labels=labels, patch_artist=True, showmeans=True, widths=0.6)
        for patch, cc in zip(bp["boxes"], box_colors):
            patch.set_facecolor(cc)
            patch.set_alpha(0.45)
        U_dep = math.ceil(30 / 4) + math.ceil(1.0 / 0.05)
        e.axhline(U_dep, ls="--", color="#c0392b",
                  label=fr"Accum. deadline $U={U_dep}$ ($w{{=}}0.05$)")
        e.legend(fontsize=8, loc="upper left")
    e.set_ylabel(r"Per-window max service age $T_{\max}$ (slots)")
    e.set_title("(e) Theory meets the real Intel trace (30 windows)")
    e.grid(alpha=0.3, axis="y")
    plt.setp(e.get_xticklabels(), rotation=15, fontsize=8)

    # (f) worst-case vs typical: ratio gap/U
    ratio_adv = adv / np.maximum(U, 1)
    ratio_sto = sto / np.maximum(U, 1)
    f.plot(ws, ratio_adv, "o-", color="#2c3e50", label="Adversarial gap / $U$")
    f.plot(ws, ratio_sto, "D-", color="#16a085", label="Stochastic gap / $U$")
    f.axhline(1.0, ls="--", color="#c0392b", label=r"Bound $U$ (ceiling)")
    f.set_xscale("log")
    f.set_ylim(0.0, 1.1)
    f.set_xlabel(r"Debt weight $w_{\mathrm{debt}}$")
    f.set_ylabel(r"Realised gap / bound $U$")
    f.set_title("(f) Worst-case bound is a ceiling, not the typical case")
    f.legend(fontsize=8, loc="lower left")
    f.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "theory_validation.png")
    fig.savefig(out, dpi=165, bbox_inches="tight")
    print(f"[fig] wrote {out}")

    # ---- numeric summary ---------------------------------------------------
    print("\n===== EXTENDED THEORY VALIDATION SUMMARY =====")
    print(f"V_max={V}, horizon={H}, seed={seed}")

    print("\n--- (a) Two-sided tightness, accumulating (N=30,B=4) ---")
    band_ok = int(np.sum((adv < U - 2) | (adv > U)))
    for w, ee, es, u in zip(ws, adv, sto, U):
        print(f"  w={w:6.2f}  adv_gap={ee:4d}  in[U-2,U]=[{u-2},{u}]:{u-2<=ee<=u}"
              f"   stochastic_gap={es:4d}  (<<bound: {es < u})")
    print(f"  adversarial gaps outside [U-2,U]: {band_ok} (expect 0)")
    sto_ratio = float(np.mean(sto / np.maximum(U, 1)))
    print(f"  mean(stochastic_gap / U) = {sto_ratio:.3f}  (#3: typical << worst-case ceiling)")

    print("\n--- (b) N sweep (B=4,w=0.2) ---")
    for n, ee, u in zip(Ns, empN, UN):
        print(f"  N={n:4d}  empirical={ee:4d}  U={u:4d}  in_band={u-2<=ee<=u}")

    print("\n--- (c) Jain convergence (N=30,B=4) ---")
    print(f"  w=0.00 (no debt) Jain_bounded={j0:.4f}")
    for w, ja, jb in zip(ws, jain_acc, jain_bd):
        print(f"  w={w:6.2f}  Jain_accum={ja:.4f}  Jain_bounded={jb:.4f}")

    print("\n--- (d) Deployed Theta(t0): empirical vs tau_lb + 3*ceil(N/B) ---")
    for w in ws_dep:
        tau_lb = t0s * (V / w) / (1 - V / w)
        resid = dep[w] - tau_lb
        ok = int(np.sum(dep[w] > tau_lb + 3 * rr + 5))
        print(f"  w={w}: gaps={list(dep[w])}  residual(min/max)="
              f"({resid.min():.0f}/{resid.max():.0f})  over_band={ok}")

    print("\n--- (e) Real Intel trace max_age (30 windows) ---")
    for key in ["+debt -corr", "Full (corr+debt)", "+corr -debt", "VoU-only (R=I)"]:
        if key in variants:
            arr = np.array(variants[key])
            print(f"  variant {key:20s}: mean={arr.mean():6.2f} median={np.median(arr):6.1f} max={arr.max():6.0f}")
    for key in ["bounded", "accumulating"]:
        if key in debt_mode:
            arr = np.array(debt_mode[key])
            print(f"  debt_mode {key:18s}: mean={arr.mean():6.2f} median={np.median(arr):6.1f} max={arr.max():6.0f}")
    U_dep = math.ceil(30 / 4) + math.ceil(1.0 / 0.05)
    print(f"  analytic accumulating deadline at w=0.05: U={U_dep}")

    print("\n--- (f) worst-case vs typical ratios ---")
    print(f"  mean adversarial gap/U = {float(np.mean(ratio_adv)):.3f}")
    print(f"  mean stochastic  gap/U = {float(np.mean(ratio_sto)):.3f}")
    print("=====================================")


if __name__ == "__main__":
    main()
