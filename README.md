# ProbeTransmit: Threshold-Aware Reliable Scheduling for Probe-Then-Transmit IoT Networks

This repository contains the public reproducibility code for the manuscript:

> Threshold-Aware Reliable Scheduling with Provable Fairness for Probe-Then-Transmit IoT Networks

The implementation evaluates CAW-VoU, a threshold-aware two-stage probe-then-transmit scheduler for bandwidth-constrained IoT monitoring. The public release ships the core simulator, scheduler implementations, baseline policies, and the scripts used to produce every analysis in the manuscript. Result tables and figures are **not shipped**: readers regenerate them by running the scripts below, then check the numbers against the manuscript.

## Repository layout

```text
src/probe_transmit/      Core simulator, channel, forecasting, safety, and scheduler components
policies/                Baseline policy implementations (AoII, VoI, Whittle, MaxWeight, DT+AoI, recent SOTA)
scripts/                 Reproducibility scripts for evaluation and theory
scripts/verify/          Lightweight verification entry point
data/raw/                Small processed sensor panels needed to run the scripts
```

Running the scripts creates `docs/`, `figures/`, and `reports/` locally; these are intentionally untracked so the repository carries only code and source data.

## Data policy

Raw third-party datasets are **not redistributed** in this repository. Please download them from their official/public sources:

- Intel Berkeley Research Lab sensor traces
- Beijing Multi-Site Air Quality / PRSA dataset
- KETI Smart-Building / AISTATS cross-predictability dataset

The repository includes small processed sensor panels under `data/raw/` so the scripts run end-to-end and readers can regenerate every manuscript number locally.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PWD"
```

## Smoke test

```bash
python scripts/verify/probe_transmit_verify.py --quick
```

If raw datasets are available under `data/raw/`, the full evaluation scripts can be run from the repository root. Heavy experiments may take substantially longer than the smoke test.

## Reproducing manuscript artifacts

Examples:

```bash
python scripts/theory_validation.py
python scripts/plot_sensitivity.py
python scripts/tail_metrics.py
```

These scripts write their tables and figures into `docs/` and `figures/` in your local checkout (both untracked). Compare the regenerated numbers against the manuscript to verify reproducibility.

The danger-term ablation table (classic vs. ev-cost vs. severity VoU) is reproduced by:

```bash
python scripts/ablation_danger_evcost.py      # classic + ev-cost rows
python scripts/ablation_danger_severity.py    # classic + severity rows
```

Both default to the 30 matched Intel windows used in the manuscript.

Full real-data benchmark scripts expect the downloaded datasets to be placed under the paths documented in the corresponding script headers.

## Notes

- Obsolete exploratory experiment artifacts are intentionally excluded from this public release.
- The default deployed scheduler uses bounded deficit; the accumulating-debt variant is available for the theorem-level hard deadline guarantee.
- Please cite the associated manuscript if you use this code.
