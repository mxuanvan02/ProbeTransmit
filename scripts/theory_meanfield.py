#!/usr/bin/env python3
"""TN(b,c,d): N-arm simulation on the per-arm MDP to support the O(1/sqrt(N)) theorem.

Loads the LP certificate (pi*, P^pi*, R_rel) from theory_mdp_lp.py and simulates
the true N-arm constrained system (exactly B=floor(alpha N) served per slot):

  (b) Mean-field convergence: ||empirical state distribution - mu*||_1 -> 0 as N grows.
  (c) Index vs set-expansion agreement: fraction of slots where the deployed
      top-B index rule and the set-expansion (focus-set) policy pick the SAME
      arm set, plus the per-arm average-reward gap between them.
  (d) Optimality gap: R_rel - R_N(pi) vs 1/sqrt(N) for both policies; we check the
      gap shrinks at the predicted rate (linear in 1/sqrt(N)).

The deployed index = LP-priority order from pi* (states with pi*(serve)=1 first,
ties by the relaxed dual / fractional y*). The set-expansion policy follows
2402.05689: serve the top-B by the same priority but DETERMINISTICALLY tracks the
ideal mean-field action count, correcting drift -- here approximated by the
"follow-the-fractional-LP" rounding that keeps the served fraction at alpha.
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_transmit import channel as ch  # noqa: E402

rng = np.random.default_rng(20260619)

cert = np.load(ROOT / "docs" / "mdp_lp_certificate.npz")
alpha, sig = float(cert["alpha"]), float(cert["sig"])
R_rel = float(cert["R_rel"])
pi = cert["pi"]          # (nS,2)
Ppi = cert["Ppi"]        # (nS,nS)
alpha_budget = float(cert["alpha_budget"])

# Rebuild the same MDP structure (must match theory_mdp_lp.py)
from scipy.stats import norm  # noqa: E402
cp = ch.CHANNELS["severe_burst"]
P_gg, P_bb = 1 - cp.p_gb, 1 - cp.p_bg
p_ok = {"G": cp.p_ok_good, "B": cp.p_ok_bad}
Gb, H = 12, 4
zmin, zmax = 0.0, 8.0
z_edges = np.linspace(zmin, zmax, Gb + 1)
z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
fresh = Gb - 1
lam = 6.0
states = [(b, c) for b in range(Gb) for c in ("G", "B")]
Sidx = {s: i for i, s in enumerate(states)}
nS = len(states)
chan_idx = {"G": 0, "B": 1}

def reward_of(b):
    z = z_mid[b]
    return -(float(norm.sf(z)) + lam * float(norm.sf(z)))

# per-state serve-priority from pi*: prob of serving; deterministic-serve first
serve_prob = pi[:, 1]
priority = serve_prob.copy()        # higher = serve sooner
reward_vec = np.array([reward_of(b) for (b, c) in states])

def age_bin(b):
    return max(0, b - 1)

def step_arm(b, c_is_bad, served):
    """Advance one arm. Returns (b', bad')."""
    # channel G-E: stay-bad w.p. P_bb; G->B w.p. 1-P_gg
    if c_is_bad:
        bad2 = rng.random() < P_bb
    else:
        bad2 = rng.random() < (1 - P_gg)
    cmode = "B" if c_is_bad else "G"
    if served:
        ok = rng.random() < p_ok[cmode]
        b2 = fresh if ok else age_bin(b)
    else:
        b2 = age_bin(b)
    return b2, bad2

def state_index(b, bad):
    return Sidx[(b, "B" if bad else "G")]

def run_N(N, T=400, warm=100, policy="index"):
    B = max(1, int(np.floor(alpha_budget * N)))
    b = rng.integers(0, Gb, size=N)
    bad = rng.random(N) < 0.5
    emp = np.zeros(nS)
    rew_acc = 0.0
    cnt = 0
    for t in range(T):
        idx = np.array([state_index(b[i], bad[i]) for i in range(N)])
        pr = priority[idx]
        if policy == "index":
            served_set = np.argsort(-pr)[:B]
        else:  # set-expansion: round the fractional LP target per state class
            # serve to match alpha within each priority tier, deterministic fill
            order = np.argsort(-pr)
            served_set = order[:B]
            # set-expansion correction: if a tied tier straddles the budget,
            # prefer arms whose channel is Good (ideal-action tracking)
            if B < N:
                cutoff = pr[order[B - 1]]
                tie = np.where(np.isclose(pr, cutoff))[0]
                if len(tie) > 1:
                    good = [i for i in tie if not bad[i]]
                    keep = [i for i in order[:B] if not np.isclose(pr[i], cutoff)]
                    fill = (good + [i for i in tie if i not in good])[: B - len(keep)]
                    served_set = np.array(keep + list(fill))
        served = np.zeros(N, bool)
        served[served_set] = True
        if t >= warm:
            for i in range(N):
                emp[idx[i]] += 1
            rew_acc += reward_vec[idx].mean()
            cnt += 1
        nb = np.empty(N, int); nbad = np.empty(N, bool)
        for i in range(N):
            nb[i], nbad[i] = step_arm(b[i], bad[i], served[i])
        b, bad = nb, nbad
    emp = emp / emp.sum()
    return emp, rew_acc / cnt

def main():
    # mu* over recurrent states from Ppi
    evals, evecs = np.linalg.eig(Ppi.T)
    k = int(np.argmin(np.abs(evals - 1.0)))
    mu_star = np.real(evecs[:, k]); mu_star = np.clip(mu_star, 0, None); mu_star /= mu_star.sum()

    Ns = [10, 20, 50, 100, 200, 500, 1000]
    SEEDS = 32
    print(f"{'N':>5} {'||emp-mu*||_1':>14} {'R_idx':>10} {'R_se':>10} "
          f"{'gap_idx':>10} {'gap_se':>10} {'|R_i-R_se|':>11}")
    rows = []
    for N in Ns:
        l1s, Ris, Rss = [], [], []
        for sd in range(SEEDS):
            global rng
            rng = np.random.default_rng(1000 + sd)
            emp_i, R_i = run_N(N, policy="index")
            rng = np.random.default_rng(1000 + sd)  # same noise for paired set-exp
            emp_s, R_s = run_N(N, policy="setexp")
            l1s.append(np.abs(emp_i - mu_star).sum()); Ris.append(R_i); Rss.append(R_s)
        l1 = float(np.mean(l1s)); R_i = float(np.mean(Ris)); R_s = float(np.mean(Rss))
        gap_i = R_rel - R_i; gap_s = R_rel - R_s
        bridge = abs(R_i - R_s)   # substantive index<->set-expansion reward gap
        rows.append((N, l1, R_i, R_s, gap_i, gap_s, bridge))
        print(f"{N:>5} {l1:>14.4f} {R_i:>10.4f} {R_s:>10.4f} "
              f"{gap_i:>10.4f} {gap_s:>10.4f} {bridge:>11.4f}")

    rows = np.array(rows)
    # (d) regress |gap| on 1/sqrt(N)
    invsq = 1 / np.sqrt(rows[:, 0])
    for col, name in [(4, "index"), (5, "setexp")]:
        A = np.vstack([invsq, np.ones_like(invsq)]).T
        slope, intercept = np.linalg.lstsq(A, np.abs(rows[:, col]), rcond=None)[0]
        pred = A @ [slope, intercept]
        ss_res = np.sum((np.abs(rows[:, col]) - pred) ** 2)
        ss_tot = np.sum((np.abs(rows[:, col]) - np.abs(rows[:, col]).mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        print(f"[|gap|~1/sqrt(N)] {name}: slope={slope:.4f} intercept={intercept:.4f} R^2={r2:.3f}")
    print(f"[bridge] max |R_index - R_setexp| over N>=100: "
          f"{np.max(rows[rows[:,0]>=100][:,6]):.4f}")

    np.savez(ROOT / "docs" / "theory_meanfield.npz", rows=rows, mu_star=mu_star, R_rel=R_rel)
    print("[save] docs/theory_meanfield.npz")

def agreement(N, T=200):
    """Fraction of slots where index and set-expansion pick the same served set."""
    B = max(1, int(np.floor(alpha_budget * N)))
    b = rng.integers(0, Gb, size=N)
    bad = rng.random(N) < 0.5
    same = 0
    for t in range(T):
        idx = np.array([state_index(b[i], bad[i]) for i in range(N)])
        pr = priority[idx]
        order = np.argsort(-pr)
        s_idx = set(order[:B].tolist())
        # set-expansion variant
        s_se = set(order[:B].tolist())
        if B < N:
            cutoff = pr[order[B - 1]]
            tie = np.where(np.isclose(pr, cutoff))[0]
            if len(tie) > 1:
                good = [i for i in tie if not bad[i]]
                keep = [i for i in order[:B] if not np.isclose(pr[i], cutoff)]
                fill = (good + [i for i in tie if i not in good])[: B - len(keep)]
                s_se = set(keep + list(fill))
        same += int(s_idx == s_se)
        served = np.zeros(N, bool); served[list(s_idx)] = True
        nb = np.empty(N, int); nbad = np.empty(N, bool)
        for i in range(N):
            nb[i], nbad[i] = step_arm(b[i], bad[i], served[i])
        b, bad = nb, nbad
    return same / T

if __name__ == "__main__":
    main()
