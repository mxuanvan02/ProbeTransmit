# Datasets

This release bundles the **small processed sensor panels** required to reproduce
the experiments in the paper. Each panel is a NumPy array (`.npy`) of shape
`[T, N]` (time steps × sensors), derived from the public source datasets below.
Large raw archives are **not** redistributed here; only the processed panels the
code actually loads are included.

## 1. Intel Berkeley Research Lab (primary)

- Files:
  - `raw/intel_berkeley/intel_panel_30motes.npy` — 30-sensor active subset (main study)
  - `raw/intel_berkeley/intel_panel_12motes.npy` — 12-sensor subset (smoke/sensitivity)
  - `raw/intel_berkeley/mote_locs.txt`, `connectivity.txt` — sensor layout metadata
- Source: Intel Berkeley Research Lab sensor data (temperature/humidity/light/voltage),
  54 Mica2Dot motes, 2004. Public dataset.
  Original: http://db.csail.mit.edu/labdata/labdata.html
- Used by: main evaluation (Section IV), theory validation (Section V), ablations.

## 2. Beijing Multi-Site Air Quality

- File: `raw/_candidates/beijing_prsa/beijing_temp_panel.npy`
- Source: Beijing Multi-Site Air-Quality Data Set (PRSA), 12 monitoring sites,
  2013–2017. UCI Machine Learning Repository.
  Reference: Zhang et al., 2017.
- Role: near-uniform / low-effective-rank spatial-correlation regime
  (robustness / cross-dataset check, Section IV).

## 3. KETI Smart-Building

- File: `raw/_candidates/keti_smartbuilding/keti_clean_panel.npy`
- Source: KETI smart-building sensor dataset, 40+ office rooms
  (temperature/humidity/CO2/light/PIR).
  Reference: Hong et al., 2017.
- Role: block-clustered / high-effective-rank spatial-correlation regime
  (robustness / cross-dataset check, Section IV).

## Notes

- The `.npy` panels are preprocessed (aligned, cleaned, subset-selected) versions
  of the public sources; see `scripts/` for the extraction/preprocessing logic.
- Please cite the original dataset authors when using these panels.
- Raw multi-gigabyte logs and intermediate candidate files are intentionally
  excluded from this public release.
