"""Dataset loading utilities for public ProbeTransmit experiments.

Raw third-party datasets are not redistributed with this repository. Set
``PROBE_TRANSMIT_DATA_ROOT`` to a local directory containing the downloaded and
prepared data files required by the experiment script you want to run.
"""
from __future__ import annotations
from pathlib import Path
import os

import numpy as np
import pandas as pd

DEFAULT_DATA_ROOT = Path(os.environ.get("PROBE_TRANSMIT_DATA_ROOT", "data/raw"))


def load_numeric_panel(csv_path: Path | str | None = None, *, label: str = "probe transmit") -> np.ndarray:
    csv_path = Path(csv_path) if csv_path else DEFAULT_DATA_ROOT / "Full Data Set.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. "
            "Set PROBE_TRANSMIT_DATA_ROOT to the directory containing 'Full Data Set.csv'."
        )
    raw = pd.read_csv(csv_path)
    # Keep numeric temperature streams for threshold-monitoring experiments.
    candidate_cols = [c for c in raw.columns if "Temperature" in c]
    if not candidate_cols:
        raise ValueError(f"No temperature columns found in {csv_path}: {list(raw.columns)}")
    sub = raw[candidate_cols].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    dropped = len(raw) - len(sub)
    print(
        f"[{label}] Loaded {len(sub)} rows, {len(candidate_cols)} columns from {csv_path} "
        f"(dropped {dropped} non-numeric rows)."
    )
    return sub.to_numpy(dtype=float)


def expand_virtual_loops(data: np.ndarray, target_n: int, seed: int = 1337) -> np.ndarray:
    """Construct virtual loops for hard scheduling stress.

    These are not new real sensors. They are deterministic transformations of the
    real columns to create an N-loop scheduling stress test. We document this in
    every paper claim that uses N > data.shape[1].
    """
    if target_n <= data.shape[1]:
        return data[:, :target_n]
    rng = np.random.default_rng(seed)
    cols = []
    t = np.arange(data.shape[0])
    for j in range(target_n):
        base = data[:, j % data.shape[1]]
        shift = (j * 97) % max(len(base), 1)
        x = np.roll(base, shift).astype(float)
        x = x + 0.35 * np.sin(2 * np.pi * (t + 17 * j) / (12 * 24))
        x = x + rng.normal(0.0, 0.08 + 0.01 * (j % 5), size=len(base))
        cols.append(x)
    return np.column_stack(cols)


def select_starts(total_steps: int, window: int, n_windows: int) -> list[int]:
    if total_steps < window + n_windows:
        raise ValueError("Not enough data for the requested windows.")
    rng = np.random.default_rng(2026)
    starts = sorted(int(s) for s in rng.choice(total_steps - window - 1, size=n_windows, replace=False))
    return starts
