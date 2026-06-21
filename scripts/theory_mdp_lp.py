#!/usr/bin/env python3
"""TN(a): Build the finite per-arm MDP from the Intel fit, solve the relaxed
single-arm LP for pi*, and verify Assumption 1 (aperiodic unichain) numerically.

This is the foundation for the asymptotic-optimality theorem (arXiv 2402.05689).
State  s = (belief-bin b in 0..Gb-1, channel c in {G,B}).
  - belief bin = quantized standardized distance-to-threshold z = (u-mu)/s_pred,
    which is the only belief summary the danger term depends on (Sec. theory).
    We bin z into Gb cells over [zmin, zmax]; smaller z = closer to bound.
  - channel c = independent Gilbert-Elliott mode (per-arm channel).
Action a in {0,1}: 1 = serve (probe+transmit) this slot.

Transitions:
  - channel: G-E kernel P_gg=1-p_gb, P_bb=1-p_bg (restless, independent of a).
  - belief: if served AND delivery ok (prob p_ok_c), reset to the "fresh" bin
    (largest z, safest); else age one AR(1) step -> z shrinks (more dangerous)
    because predictive sd grows. If served but delivery fails, same as not served.
Reward (maximize): r(s,a) = -(track(z) + lam * Pviol(z)). Serving cost folded
    into the budget constraint, not the reward (standard RMAB convention).

Outputs: LP value R_rel, pi*, and the unichain/aperiodicity certificate.
"""
import sys
import numpy as np
from pathlib import Path
from scipy.optimize import linprog
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_transmit import channel as ch  # noqa: E402
from probe_transmit import data as datamod  # noqa: E402

# ----- fit AR(1) on Intel to get alpha, innovation sd -----
def fit_ar1(series):
    x = series[:-1]
    y = series[1:]
    x = x - x.mean()
    y = y - y.mean()
    alpha = float(np.dot(x, y) / (np.dot(x, x) + 1e-12))
    resid = series[1:] - alpha * series[:-1] - (1 - alpha) * series[:-1].mean()
    sig = float(np.std(resid))
    return alpha, sig


def main():
    panel = np.load(ROOT / "data" / "raw" / "intel_berkeley" / "intel_panel_30motes.npy")
    # panel shape (T, N); fit a representative arm (median-variance mote)
    variances = panel.var(axis=0)
    j = int(np.argsort(variances)[len(variances) // 2])
    alpha, sig = fit_ar1(panel[:, j])
    rng = panel.max() - panel.min()
    print(f"[fit] AR(1) alpha={alpha:.4f}  innovation sd={sig:.4f}  range={rng:.2f}  mote={j}")

    # ----- MDP construction -----
    cp = ch.CHANNELS["severe_burst"]
    P_gg = 1 - cp.p_gb
    P_bb = 1 - cp.p_bg
    p_ok = {"G": cp.p_ok_good, "B": cp.p_ok_bad}
    print(f"[chan] P_gg={P_gg:.2f} P_bb={P_bb:.2f} p_ok_G={p_ok['G']:.2f} p_ok_B={p_ok['B']:.2f}")

    Gb = 12          # belief bins on z = distance-to-threshold in predictive sd
    H = 4            # forecast horizon (steps)
    zmin, zmax = 0.0, 8.0
    z_edges = np.linspace(zmin, zmax, Gb + 1)
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])

    # predictive sd after k ages (AR(1) variance accumulation), capped
    def pred_sd(age):
        # var_h = sig^2 * sum_{i=0}^{H-1} alpha^{2i}, then aged: more age -> larger
        base = sig * np.sqrt(np.sum(alpha ** (2 * np.arange(H))))
        return base * (1.0 + 0.15 * age)

    # danger + track as functions of bin (age encoded via current z)
    def track(z):
        return float(norm.sf(z))      # proxy: closeness pressure, in [0,1]
    def pviol(z):
        return float(norm.sf(z))      # exit prob ~ Phibar(z)

    lam = 6.0
    # "fresh" bin index = safest (largest z)
    fresh = Gb - 1

    states = [(b, c) for b in range(Gb) for c in ("G", "B")]
    Sidx = {s: i for i, s in enumerate(states)}
    nS = len(states)

    # belief aging: when not refreshed, z shrinks by one bin (toward danger)
    def age_bin(b):
        return max(0, b - 1)

    # reward
    def reward(s):
        b, c = s
        z = z_mid[b]
        return -(track(z) + lam * pviol(z))

    # transition: returns dict s'->prob given (s,a)
    def trans(s, a):
        b, c = s
        out = {}
        # channel evolves regardless of a
        cdst = {"G": {"G": P_gg, "B": 1 - P_gg}, "B": {"B": P_bb, "G": 1 - P_bb}}[c]
        # belief evolves
        if a == 1:
            ok = p_ok[c]
            # success -> reset to fresh; fail -> age
            belief_dst = {fresh: ok, age_bin(b): 1 - ok}
        else:
            belief_dst = {age_bin(b): 1.0}
        for bp, pb in belief_dst.items():
            for cp_, pc in cdst.items():
                out[(bp, cp_)] = out.get((bp, cp_), 0.0) + pb * pc
        return out

    # ----- build LP -----
    # vars y(s,a): index = 2*Sidx[s] + a
    nV = 2 * nS
    r = np.zeros(nV)
    for s in states:
        for a in (0, 1):
            r[2 * Sidx[s] + a] = reward(s)
    # maximize r.y  -> minimize -r.y
    c_obj = -r

    # balance constraints: for each s: sum_{s',a} y(s',a)P(s',a,s) - sum_a y(s,a) = 0
    A_eq = []
    b_eq = []
    for s in states:
        row = np.zeros(nV)
        # inflow
        for sp in states:
            for a in (0, 1):
                p = trans(sp, a).get(s, 0.0)
                if p:
                    row[2 * Sidx[sp] + a] += p
        # outflow
        for a in (0, 1):
            row[2 * Sidx[s] + a] -= 1.0
        A_eq.append(row); b_eq.append(0.0)
    # budget: sum_s y(s,1) = alpha
    row = np.zeros(nV)
    for s in states:
        row[2 * Sidx[s] + 1] = 1.0
    A_eq.append(row); b_eq.append(0.0)  # placeholder alpha set below
    # normalization: sum y = 1
    row = np.ones(nV)
    A_eq.append(row); b_eq.append(1.0)

    alpha_budget = 0.25
    b_eq[-2] = alpha_budget
    A_eq = np.array(A_eq); b_eq = np.array(b_eq)

    res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * nV, method="highs")
    assert res.success, res.message
    y = res.x.reshape(nS, 2)
    R_rel = -res.fun
    print(f"[LP] solved. R_rel (per-arm upper bound) = {R_rel:.4f}  budget alpha={alpha_budget}")

    # pi*(a|s)
    pi = np.zeros((nS, 2))
    for i in range(nS):
        tot = y[i].sum()
        pi[i] = y[i] / tot if tot > 1e-12 else np.array([0.5, 0.5])

    # build P^{pi*}
    Ppi = np.zeros((nS, nS))
    for s in states:
        i = Sidx[s]
        for a in (0, 1):
            wa = pi[i, a]
            if wa <= 0:
                continue
            for sp, p in trans(s, a).items():
                Ppi[i, Sidx[sp]] += wa * p
    # sanity rows sum to 1
    assert np.allclose(Ppi.sum(axis=1), 1.0), Ppi.sum(axis=1)

    # ----- unichain check: number of recurrent classes via strongly-connected -----
    # reachability closure
    reach = (Ppi > 1e-12).astype(int)
    R = reach.copy()
    for _ in range(nS):
        R = ((R @ reach) > 0).astype(int) | R
    # recurrent state: cannot leave its communicating class
    comm = (R & R.T).astype(bool)
    # count distinct recurrent classes: states that reach only within their comm set
    recurrent = []
    for i in range(nS):
        succ = set(np.where(R[i])[0])
        clique = set(np.where(comm[i])[0])
        if succ <= clique:   # all reachable states communicate back => recurrent
            recurrent.append(i)
    classes = []
    seen = set()
    for i in recurrent:
        if i in seen:
            continue
        cl = set(np.where(comm[i])[0]) & set(recurrent)
        classes.append(cl); seen |= cl
    n_classes = len(classes)
    print(f"[unichain] #recurrent classes = {n_classes}  -> {'UNICHAIN OK' if n_classes==1 else 'NOT unichain'}")

    # ----- aperiodicity: gcd of cycle lengths via self-loop in recurrent class -----
    rec = sorted(classes[0]) if classes else []
    diag_selfloop = any(Ppi[i, i] > 1e-9 for i in rec)
    print(f"[aperiodic] self-loop in recurrent class: {diag_selfloop} -> "
          f"{'APERIODIC OK' if diag_selfloop else 'check period'}")

    # stationary mu*
    if classes:
        sub = rec
        Psub = Ppi[np.ix_(sub, sub)]
        Psub = Psub / Psub.sum(axis=1, keepdims=True)
        evals, evecs = np.linalg.eig(Psub.T)
        k = int(np.argmin(np.abs(evals - 1.0)))
        mu = np.real(evecs[:, k]); mu = mu / mu.sum()
        print(f"[stationary] mu* support size {len(sub)}, min prob {mu.min():.4f}")

    # fraction of states where pi* is deterministic (LP-priority structure)
    detfrac = np.mean((pi.max(axis=1) > 0.999))
    print(f"[pi*] deterministic in {100*detfrac:.0f}% of states")

    np.savez(ROOT / "docs" / "mdp_lp_certificate.npz",
             alpha=alpha, sig=sig, R_rel=R_rel, Ppi=Ppi, pi=pi,
             n_classes=n_classes, aperiodic=diag_selfloop, alpha_budget=alpha_budget)
    print("[save] docs/mdp_lp_certificate.npz")
    print(f"\nSUMMARY: Assumption 1 {'HOLDS' if (n_classes==1 and diag_selfloop) else 'FAILS'} "
          f"(unichain={n_classes==1}, aperiodic={diag_selfloop})")


if __name__ == "__main__":
    main()
