"""
Quick diagnostic: correlate GCOR with FD across all runs.

Reads the scrubbing report and Schaefer timeseries from the active output directory.

Usage
-----
    python gcor_fd_check.py                      # uses config.TS_OUTPUT_DIR
    python gcor_fd_check.py --dir PATH/TO/DATA   # custom timeseries directory
    python gcor_fd_check.py --subject sub-002    # restrict to one subject
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE, ".."))
from config import TS_OUTPUT_DIR  # noqa: E402


def compute_gcor(ts_path: str) -> float:
    """Mean off-diagonal absolute correlation (simple GCOR proxy)."""
    df  = pd.read_csv(ts_path)
    c   = np.corrcoef(df.values.T)
    np.fill_diagonal(c, 0)
    return float(np.nanmean(np.abs(c[~np.eye(c.shape[0], dtype=bool)])))


def main(ts_dir: str, subject: str = None) -> None:
    # ── 1. Compute GCOR per run from Schaefer timeseries ─────────────────────
    pattern = os.path.join(ts_dir, "*_schaefer100_ts.csv")
    files   = sorted(glob.glob(pattern))
    if subject:
        files = [f for f in files if subject in os.path.basename(f)]

    if not files:
        print(f"No Schaefer timeseries found in {ts_dir}")
        return

    gcor_rows = []
    for f in files:
        name = os.path.basename(f).replace("_schaefer100_ts.csv", "")
        gcor_rows.append({"name": name, "gcor": compute_gcor(f)})
    gcor_df = pd.DataFrame(gcor_rows)

    # ── 2. Load scrubbing report ──────────────────────────────────────────────
    report_path = os.path.join(ts_dir, "scrubbing_report_filtered.csv")
    if not os.path.isfile(report_path):
        print(f"Scrubbing report not found: {report_path}")
        print("Showing GCOR-only table:")
        print(gcor_df.to_string(index=False))
        return

    fd = pd.read_csv(report_path)
    if subject:
        fd = fd[fd.get("subject", pd.Series(dtype=str)) == subject]
    fd["name"] = fd.apply(
        lambda r: "{}_{}_{}_{}".format(
            r.get("subject", ""),
            r["session"],
            r["task"],
            r.get("run", "run-1"),
        ),
        axis=1,
    ).str.strip("_")

    # ── 3. Merge & print ─────────────────────────────────────────────────────
    fd_cols = [c for c in ["fd_mean_filtered", "fd_mean_raw", "raw_spikes",
                           "filtered_spikes", "scrub_percent", "session", "task"]
               if c in fd.columns]
    merged = gcor_df.merge(fd[["name"] + fd_cols], on="name", how="left")

    print("\n=== GCOR vs Motion per run ===\n")
    print(merged.to_string(index=False))

    print("\n=== Correlations (n={}) ===".format(len(merged)))
    for col in [c for c in ["fd_mean_raw", "fd_mean_filtered", "raw_spikes",
                             "filtered_spikes", "scrub_percent"] if c in merged.columns]:
        r = merged["gcor"].corr(merged[col])
        print(f"  GCOR vs {col:<25}: r = {r:+.3f}")

    if "task" in merged.columns and "session" in merged.columns:
        print("\n=== Mean GCOR by session x condition ===")
        merged["is_music"] = merged["task"].str.startswith("music")
        print(merged.groupby(["session", "is_music"])[["gcor"]].mean().round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir",     default=TS_OUTPUT_DIR, help="Timeseries directory")
    parser.add_argument("--subject", default=None,          help="Filter to one subject (e.g. sub-002)")
    args = parser.parse_args()
    main(args.dir, args.subject)
