# SOTA Comparison Report (30 windows, paired)

**Date:** 2026-06-12
**Workdir:** `2026_ProbeTransmit_Greenhouse_IoTJ`
**Script:** `code/scripts/sota_comparison_30windows.py`
**Per-window data:** `code/docs/sota_comparison_30windows.csv`

## Setup (paired, identical to Whittle-exact 30 windows)

- Dataset: Intel Berkeley, 30 motes, real temperature panel (`intel_panel_30motes.npy`).
- 30 windows, each 240 scheduling rounds; start indices from `select_starts(data_len, 240, 30)` (seed 2026).
- Per-window seed `50260 + window_id*17`; channel `severe_burst`; N=30, B_probe=B_payload=4.
- AR(1) forecaster + empirical residual safety model fit on the first 2000 rows.
- Every policy runs on the **same** 30 windows → fully paired (Wilcoxon signed-rank).
- Objective `loss = track_mse/RANGE^2 + 6.0 * missed_violation` (see `safety.loss_terms`). Lower is better for loss, rmse, missed_vio, runtime.

Consistency check: CAW-VoU window-0 loss = 0.001145 here matches `whittle_exact_comparison_30windows.csv` (0.0011453) exactly — same pipeline, no drift.

### Baseline realizability note (honest)

- **AoII-greedy / AoII+debt are GENIE-AIDED oracles.** AoII needs the true-error indicator `|xh-x|<delta`, which a pull-based gateway cannot observe before probing. We update the AoII counter from ground truth inside the policy `step()`, giving AoII an information advantage CAW-VoU does not have. A belief-only AoII proxy (`policies.AoIIProbe`) exists separately. So any CAW-VoU result that matches/beats genie-AoII is a **lower bound** on CAW-VoU's merit.
- MaxWeight (Lyapunov virtual queues), VoI (posterior-variance reduction) use belief state only — directly realizable, fair comparison.
- AoII accuracy band delta = per-node AR(1) one-step residual std.

## Summary table (mean ± 95% CI, n=30)

| Policy | Loss | RMSE | Miss% | Runtime(ms) | p vs CAW-VoU (loss) |
|--------|------|------|-------|-------------|---------------------|
| **CAW-VoU** | **0.00238 ± 0.00105** | 0.1202 ± 0.0393 | **0.0367 ± 0.0172** | 111.2 ± 4.3 | — |
| Whittle-heuristic | 0.09320 ± 0.04972 | 0.5283 ± 0.1729 | 1.4463 ± 0.8381 | 1.18 ± 0.21 | 0.0000 |
| AoII-greedy (genie) | 0.01148 ± 0.00559 | 0.1204 ± 0.0581 | 0.1874 ± 0.0920 | 1.36 ± 0.40 | 0.0028 |
| AoII+debt (genie) | 0.01023 ± 0.00486 | **0.1164 ± 0.0527** | 0.1669 ± 0.0802 | 1.14 ± 0.26 | 0.0035 |
| AoII+debt+threshold (genie) | 0.00777 ± 0.00419 | 0.2487 ± 0.0999 | 0.1125 ± 0.0718 | 1.49 ± 0.42 | 0.0000 |
| MaxWeight-AoI | 0.01481 ± 0.00732 | **0.1042 ± 0.0367** | 0.2445 ± 0.1220 | 1.17 ± 0.33 | 0.0071 |
| MaxWeight-VoU | 0.00649 ± 0.00467 | 0.2077 ± 0.0705 | 0.0921 ± 0.0743 | 2.44 ± 0.62 | 0.0043 |
| VoI-greedy | 0.08796 ± 0.05043 | 0.5573 ± 0.2314 | 1.3282 ± 0.8415 | 1.38 ± 0.33 | 0.0000 |
| VoI+debt | 0.08903 ± 0.05070 | 0.5233 ± 0.2021 | 1.3715 ± 0.8395 | 1.37 ± 0.25 | 0.0000 |

Bold = best column (loss, miss% → CAW-VoU; rmse → MaxWeight-AoI / AoII+debt).

## Paired statistics vs CAW-VoU (Wilcoxon, win/loss/tie, rank-biserial)

| Baseline | Loss (p, W/L/T) | RMSE (p, W/L/T) | Miss% (p, W/L/T) |
|----------|-----------------|------------------|-------------------|
| Whittle-heuristic | 0.0000, 30/0/0, −1.00 | 0.0000, 30/0/0, −1.00 | 0.0005, 15/2/13 |
| AoII-greedy (genie) | 0.0028, 18/12/0, −0.61 | 0.0427, 8/22/0, **+0.42** | 0.0005, 15/1/14 |
| AoII+debt (genie) | 0.0035, 17/13/0, −0.60 | 0.0405, 7/23/0, **+0.43** | 0.0005, 15/1/14 |
| AoII+debt+threshold | 0.0000, 30/0/0, −1.00 | 0.0024, 23/7/0, −0.62 | 0.0131, 11/3/16 |
| MaxWeight-AoI | 0.0071, 17/13/0, −0.55 | 0.0000, 1/29/0, **+0.91** | 0.0005, 15/2/13 |
| MaxWeight-VoU | 0.0043, 23/7/0, −0.58 | 0.0000, 26/4/0, −0.82 | 0.0570, 9/6/15 |
| VoI-greedy | 0.0000, 30/0/0, −1.00 | 0.0000, 30/0/0, −1.00 | 0.0008, 14/1/15 |
| VoI+debt | 0.0000, 30/0/0, −1.00 | 0.0000, 30/0/0, −1.00 | 0.0008, 14/1/15 |

W/L/T = CAW-VoU win/loss/tie (lower-is-better). Negative rank-biserial favors CAW-VoU; positive favors the baseline.

## Key findings

1. **CAW-VoU dominates on the safety-weighted objective (loss) against every baseline, significantly (all p ≤ 0.0071).** Mean-ratio CAW/baseline ranges 0.03 (vs VoI/Whittle) to 0.37 (vs MaxWeight-VoU). On loss it never loses a majority: best case 30/0/0 (vs Whittle, VoI, AoII+threshold), worst case 17/13/0 (vs AoII+debt, MaxWeight-AoI) — still a significant win.

2. **CAW-VoU has the lowest missed-violation rate of all policies (0.037%)**, significantly better than every baseline except MaxWeight-VoU which is borderline (p=0.057, 9/6/15 with many ties because most windows have zero violations). This is the headline safety result: CAW-VoU's empirically-priced safety term plus correlation confirmation suppresses threshold misses.

3. **HONEST THREAT — CAW-VoU does NOT win on pure RMSE.** On raw tracking RMSE it **loses** to MaxWeight-AoI (1/29, p<0.0001, rbc +0.91), genie-AoII+debt (7/23, p=0.041), and genie-AoII-greedy (8/22, p=0.043). Pure age/genie-correctness scheduling tracks the field marginally better in L2 terms. This is expected: CAW-VoU deliberately trades a little RMSE to slash safety-violation cost, which is what the `loss = track + 6*missed` objective rewards. **The paper must defend the objective formulation, not claim universal RMSE superiority.**

4. **VoI ≈ Whittle-heuristic, as predicted, and both are weak here.** VoI-greedy/VoI+debt sit at loss ≈ 0.088–0.089, RMSE ≈ 0.52–0.56, virtually identical to Whittle-heuristic (0.093 / 0.53). Pure max-uncertainty / max-variance probing ignores the safety threshold and is dominated on every metric (30/0/0 on loss and RMSE). Adding debt to VoI changes almost nothing. Low novelty threat, as anticipated.

5. **Runtime is CAW-VoU's real cost.** CAW-VoU is 45×–98× slower than the O(N log N) heuristics (111 ms/step vs ~1–2 ms/step) due to correlation submodular greedy + empirical-residual sampling. At N=30, B=4 this is still ~0.1 s/round, acceptable for greenhouse control cadence, but it is a genuine tradeoff to disclose.

6. **AoII+debt+threshold underperforms AoII+debt** (loss 0.0078 vs 0.0102 — wait, threshold loss is lower but RMSE much worse 0.249 vs 0.116). The near-threshold gating concentrates probing on edge nodes, lowering violation-weighted loss but letting interior tracking drift, inflating RMSE. CAW-VoU still beats it 30/0/0 on loss.

## Implications for the paper

- **Strong, defensible novelty on the stated objective and on safety.** CAW-VoU is the unique policy that is simultaneously best on safety-weighted loss and on missed violations, beating realizable baselines (MaxWeight, VoI) AND a genie-aided AoII oracle on loss. Frame the contribution as *safety-constrained estimation under a pull-based probe budget*, not generic AoI/RMSE minimization.
- **Pre-empt the RMSE objection.** A reviewer will note MaxWeight-AoI / AoII have lower RMSE. The defense is the objective: greenhouse control cares about staying inside `[18, 32]°C`, encoded by the `6×` miss penalty. Report RMSE honestly alongside the loss/miss win; do not hide it. Consider adding a Pareto plot (RMSE vs missed-vio) showing CAW-VoU on the safety-favorable frontier.
- **Disclose the genie advantage of AoII.** State clearly that AoII baselines are oracle-aided; CAW-VoU beating them on loss despite that is a lower bound on its advantage.
- **Disclose runtime honestly** as the cost of correlation-aware submodular selection, and note it is still real-time at deployment scale.
- **Drop or down-weight VoI as a "competitor"**; present it as a sanity-check baseline equivalent to Whittle-heuristic.

## Verification

- Build/run: `python3 scripts/sota_comparison_30windows.py --workers 6`, wall time 148.8 s, exit 0.
- All 8(+1) policies ran on all 30 windows (270 rows in `sota_comparison_30windows.csv`).
- Runtime measured with `perf_counter` per window (ms/step). No estimated numbers.
- Pipeline consistency confirmed against `whittle_exact_comparison_30windows.csv` (CAW-VoU window-0 loss identical).
