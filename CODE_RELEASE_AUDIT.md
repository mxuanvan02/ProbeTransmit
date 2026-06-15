# Public Code Release Audit — ProbeTransmit

Date: 2026-06-15 20:15 Asia/Saigon

## Scope

This public release was prepared from the internal `code/` workspace without deleting or modifying the original research workspace. Raw third-party datasets are excluded; processed result tables and manuscript figures are included.

## Included

- Core package: `src/probe_transmit/`
- Baselines: `policies/`
- Reproducibility scripts: selected real-data, theory, sensitivity, and plotting scripts under `scripts/`
- Processed result tables: `docs/*.csv`, `docs/*.json`
- Publication figures: `figures/*.png`
- Metadata: `README.md`, `requirements.txt`, `LICENSE`, `CITATION.cff`

## Excluded

- Raw third-party datasets
- `.DS_Store`, `__pycache__`, logs, local reports/build/cache
- Quarantined or obsolete exploratory artifacts
- Internal manuscript notes and private workspace paths
- Synthetic/fusion exploratory outputs not claimed in the current manuscript

## Audit checks

- Python compile: pass (`python3 -m compileall -q src policies scripts`)
- Public smoke test: pass (`PYTHONPATH="$PWD/src:$PWD" python3 scripts/verify/probe_transmit_verify.py --quick`)
- Sensitive/stale grep: pass for API keys, secrets, internal OpenClaw paths, quarantine markers, B2CRoI/RABS, COMPAG/old-paper markers, and synthetic/fusion stale terms
- Raw data redistribution: no raw dataset files included

## Known limitations

- Full benchmark reproduction requires users to download the third-party datasets and place them under local `data/raw/` paths documented by scripts.
- Processed result tables are included to support inspection of manuscript claims without redistributing raw data.
- The repository is ready for GitHub publication, but pushing requires an authenticated GitHub CLI or a remote URL/token configured by the owner.
