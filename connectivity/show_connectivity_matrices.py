"""
Connectivity matrix heatmaps from timeseries CSVs.

Reads every <atlas>_ts.csv from the timeseries directory and saves a PNG
heatmap per file.

Atlases supported (matches DEFAULT_ATLASES in extraction/atlases.py):
  _schaefer400_ts.csv
  _tian_s2_ts.csv

Usage
-----
    python show_connectivity_matrices.py              # reads dir from env / config default
    python show_connectivity_matrices.py --data DIR   # custom directory
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE, ".."))
from utils import pearson_matrix, plot_connectivity_matrix  # noqa: E402
from config import TS_OUTPUT_DIR, OUTPUTS_DIR               # noqa: E402

# Pipeline passes TS_DIR; fall back to config default
TS_DIR = os.environ.get("TS_DIR", TS_OUTPUT_DIR)

# Atlases produced by the current DEFAULT_ATLASES
ATLAS_SUFFIXES = [
    "_schaefer400_ts.csv",
    "_tian_s2_ts.csv",
]


def process_directory(data_dir: str, label_prefix: str, out_dir: str) -> None:
    """Plot every timeseries CSV for supported atlases in *data_dir*."""
    files = sorted(
        f for f in glob.glob(os.path.join(data_dir, "*.csv"))
        if any(f.endswith(suf) for suf in ATLAS_SUFFIXES)
    )
    if not files:
        print(f"  No timeseries CSVs found in {data_dir}")
        return

    for i, path in enumerate(files):
        bname        = os.path.basename(path)
        atlas_suffix = next(s for s in ATLAS_SUFFIXES if bname.endswith(s))
        name         = bname.replace(atlas_suffix, "")
        atlas        = atlas_suffix.replace("_ts.csv", "").lstrip("_")

        df   = pd.read_csv(path)
        corr = pearson_matrix(df).values
        np.fill_diagonal(corr, 0)
        np.nan_to_num(corr, copy=False, nan=0.0)

        title    = f"{label_prefix} | {name} | {atlas}"
        out_path = os.path.join(out_dir, f"conn_{label_prefix.lower()}_{i+1}_{name}_{atlas}.png")
        plot_connectivity_matrix(corr, title, out_path)
        print(f"  Saved: {os.path.basename(out_path)}")


def main(data_dir: str = None) -> None:
    target = data_dir or TS_DIR
    out_dir = os.path.join(target, "connectivity_matrix_plots")
    os.makedirs(out_dir, exist_ok=True)

    process_directory(target, "conn", out_dir)
    print(f"\nDone. Matrices saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="Optional custom data directory")
    args = parser.parse_args()
    main(data_dir=args.data)
