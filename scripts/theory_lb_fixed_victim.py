#!/usr/bin/env python3
"""theory_lb_fixed_victim.py

Verify the *constructive* lower bound used in the proof of the matching
lower-bound theorem. The construction fixes a SINGLE victim v for all time:

    u_v(t)   = 0        for all t      (victim, lowest admissible urgency)
    u_j(t)   = V_max    for all t      (every other node, highest urgency)

starting from all debts = 0. Under the accumulating-debt top-B rule this is a
fully deterministic instance. We measure how many consecutive slots v waits
before its first service and compare it to

    W_lb = ceil(N/B) + ceil(V_max/w) - 1     (claimed lower bound)
    U    = ceil(N/B) + ceil(V_max/w)         (Theorem-1 upper bound)

The claim of the lower-bound theorem is  wait(v) >= W_lb  for every (N,B,w),
i.e. the fixed-victim adversary already realises the worst case up to the
universal additive constant 1.
"""
from __future__ import annotations
import math
import numpy as np


def fixed_victim_wait(N: int, B: int, w: float, V_max: float = 1.0,
                      horizon: int = 100000) -> int:
    """Return the number of consecutive slots the fixed victim v=0 waits
    before its first service, under u_v=0, u_j=V_max, all debts 0 at t=0."""
    debt = np.zeros(N, dtype=float)
    u = np.full(N, V_max, dtype=float)
    u[0] = 0.0  # victim is node 0, urgency 0 forever
    for t in range(horizon):
        score = u + w * debt
        # top-B; ties broken toward larger debt then smaller index (matches
        # theory_validation.py: lexsort keys (index, -debt, -score))
        order = np.lexsort((np.arange(N), -debt, -score))
        selected = order[:B]
        if 0 in selected:
            return t  # victim served at slot t => it waited t slots (0-indexed)
        # debt update
        mask = np.zeros(N, dtype=bool)
        mask[selected] = True
        debt = np.where(mask, 0.0, debt + 1.0)
    return horizon  # never served within horizon


def main():
    Ns = [10, 20, 30, 50, 80, 120]
    Bs = [1, 2, 4, 6, 8]
    ws = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    V_max = 1.0

    print(f"{'N':>4} {'B':>3} {'w':>6} | {'wait':>5} {'W_lb':>5} {'U':>4} "
          f"{'rr':>4} {'slack':>5} | {'wait-W_lb':>9}")
    deficits = []
    below_lb = 0
    above_U = 0
    for N in Ns:
        for B in Bs:
            if B >= N:
                continue
            for w in ws:
                wait = fixed_victim_wait(N, B, w, V_max)
                rr = math.ceil(N / B)
                slack = math.ceil(V_max / w)
                U = rr + slack
                W_lb = U - 1
                d = wait - W_lb
                deficits.append(d)
                if wait < W_lb:
                    below_lb += 1
                if wait > U:
                    above_U += 1
                print(f"{N:>4} {B:>3} {w:>6.2f} | {wait:>5} {W_lb:>5} {U:>4} "
                      f"{rr:>4} {slack:>5} | {d:>9}")

    deficits = np.array(deficits)
    print("\n===== FIXED-VICTIM LOWER-BOUND CERTIFICATE =====")
    print(f"configs tested            : {len(deficits)}")
    print(f"wait <  W_lb (must be 0)   : {below_lb}")
    print(f"wait >  U    (must be 0)   : {above_U}")
    print(f"wait - W_lb : min={deficits.min()} max={deficits.max()} "
          f"mean={deficits.mean():.3f}")
    vals, cnts = np.unique(deficits, return_counts=True)
    print(" wait-W_lb : count")
    for v, c in zip(vals, cnts):
        print(f"   {int(v):>3}     : {int(c)}")
    if below_lb == 0:
        print("=> fixed-victim wait >= ceil(N/B)+ceil(Vmax/w)-1 for ALL configs")
        print("   constructive lower bound CONFIRMED.")
    print("================================================")


if __name__ == "__main__":
    main()
