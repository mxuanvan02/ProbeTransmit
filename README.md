# ProbeTransmit: Threshold-Aware Reliable Scheduling for Probe-Then-Transmit IoT Networks

This repository contains the public reproducibility code for the manuscript:

> Threshold-Aware Reliable Scheduling with Provable Fairness for Probe-Then-Transmit IoT Networks

The implementation evaluates CAW-VoU, a threshold-aware two-stage probe-then-transmit scheduler for bandwidth-constrained IoT monitoring. The public release includes the core simulator, scheduler implementations, baseline policies, manuscript result tables, and scripts used to regenerate the main analyses.

## Repository layout

```text
src/probe_transmit/      Core simulator, channel, forecasting, safety, and scheduler components
policies/                Baseline policy implementations (AoII, VoI, Whittle, MaxWeight)
scripts/                 Reproducibility scripts for evaluation/theory figures
scripts/verify/          Lightweight verification entry point
docs/                    Processed result tables used in the manuscript
figures/                 Publication figures generated from result tables
```

## Data policy

Raw third-party datasets are **not redistributed** in this repository. Please download them from their official/public sources:

- Intel Berkeley Research Lab sensor traces
- Beijing Multi-Site Air Quality / PRSA dataset
- KETI Smart-Building / AISTATS cross-predictability dataset

The repository includes processed result tables used in the manuscript so that key plots and summary checks can be inspected without redistributing raw data.

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
python scripts/theory_validation.py --output figures/theory_validation.png
python scripts/plot_sensitivity.py
python scripts/tail_metrics.py
```

Full real-data benchmark scripts expect the downloaded datasets to be placed under the paths documented in the corresponding script headers.

## Notes

- Obsolete exploratory experiment artifacts are intentionally excluded from this public release.
- The default deployed scheduler uses bounded deficit; the accumulating-debt variant is available for the theorem-level hard deadline guarantee.
- Please cite the associated manuscript if you use this code.
