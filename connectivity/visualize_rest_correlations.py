"""
Correlation matrices for rest-task timeseries.

Reads all *_ho_cortical_ts.csv (or --atlas suffix) files from a given
directory and saves a heatmap PNG for each.

Usage
-----
    python visualize_rest_correlations.py                         # uses config.TS_OUTPUT_DIR
    python visualize_rest_correlations.py --data PATH/TO/TS_DIR   # custom directory
    python visualize_rest_correlations.py --atlas schaefer100      # different atlas
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nilearn.plotting import plot_matrix

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE, ".."))
from utils import load_timeseries, pearson_matrix  # noqa: E402
from config import TS_OUTPUT_DIR, OUTPUTS_DIR       # noqa: E402

DEFAULT_OUT = os.path.join(OUTPUTS_DIR, "correlation_matrices_plots")


def visualize_directory(data_dir: str, atlas_suffix: str, out_dir: str) -> None:
    """Compute and plot correlation matrices for all matching files."""
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(f"_{atlas_suffix}_ts.csv")
    )

    if not files:
        print(f"  No *_{atlas_suffix}_ts.csv files found in {data_dir}")
        return

    print(f"  Found {len(files)} files (atlas: {atlas_suffix})")
    for i, path in enumerate(files, 1):
        bname  = os.path.basename(path)
        df     = load_timeseries(path)
        if df is None:
            continue

        corr = pearson_matrix(df)
        vals = corr.values[np.triu_indices_from(corr.values, k=1)]
        print(f"  [{i}/{len(files)}] {bname}: shape={corr.shape}  "
              f"mean_r={vals.mean():.3f}  max_r={vals.max():.3f}")

        fig, ax = plt.subplots(figsize=(14, 10))
        plot_matrix(corr, vmax=0.6, vmin=-0.4, colorbar=True, axes=ax)
        ax.set_title(bname, fontsize=11, pad=16)
        plt.tight_layout()

        out_name = bname.replace(".csv", "_correlation_matrix.png")
        out_path = os.path.join(out_dir, out_name)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {os.path.basename(out_path)}")

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",  default=TS_OUTPUT_DIR,
                        help="Directory containing timeseries CSVs")
    parser.add_argument("--atlas", default="ho_cortical",
                        help="Atlas suffix to search for (default: ho_cortical)")
    parser.add_argument("--out",   default=DEFAULT_OUT,
                        help="Output directory for PNG files")
    args = parser.parse_args()
    visualize_directory(args.data, args.atlas, args.out)
