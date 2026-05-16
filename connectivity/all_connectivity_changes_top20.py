"""
Top-20 connectivity changes between sessions.

For each task, compute the per-session correlation matrix (via utils.pearson_matrix)
and report the 20 edges with the largest absolute change across sessions.

Usage
-----
    python all_connectivity_changes_top20.py                # uses default TS dir
    python all_connectivity_changes_top20.py --data DIR     # custom timeseries dir
    python all_connectivity_changes_top20.py --subject sub-001 --atlas schaefer100
"""
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Allow imports from project root (config) and sibling utils
_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE, ".."))
from utils import load_timeseries, pearson_matrix  # noqa: E402
from config import TS_OUTPUT_DIR, OUTPUTS_DIR       # noqa: E402

DEFAULT_TS_DIR = os.environ.get("TS_GSR_OFF_DIR", TS_OUTPUT_DIR)
DEFAULT_OUT    = os.path.join(OUTPUTS_DIR, "connectivity_changes")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_connectivity_changes(
    ses1_df: pd.DataFrame,
    ses2_df: pd.DataFrame,
    ses3_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Return a DataFrame of pairwise connectivity changes across sessions.

    Connectivity is computed via Pearson r (utils.pearson_matrix) — no
    duplicate implementation here.
    """
    corr1 = pearson_matrix(ses1_df).values if ses1_df is not None else None
    corr2 = pearson_matrix(ses2_df).values if ses2_df is not None else None
    corr3 = pearson_matrix(ses3_df).values if ses3_df is not None else None

    if corr1 is None or corr2 is None:
        return pd.DataFrame()

    regions = list(ses1_df.columns)
    n = len(regions)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            r1 = corr1[i, j]
            r2 = corr2[i, j]
            r3 = corr3[i, j] if corr3 is not None else np.nan

            rows.append({
                "Region_1":      regions[i],
                "Region_2":      regions[j],
                "Ses1_r":        r1,
                "Ses2_r":        r2,
                "Ses3_r":        r3,
                "Change_1v2":    r2 - r1,
                "Change_1v3":    r3 - r1 if not np.isnan(r3) else np.nan,
                "Change_2v3":    r3 - r2 if not np.isnan(r3) else np.nan,
                "Abs_Change_1v2": abs(r2 - r1),
                "Abs_Change_1v3": abs(r3 - r1) if not np.isnan(r3) else 0.0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_top20(df_top20: pd.DataFrame, task_name: str, comparison: str, out_dir: str) -> None:
    """Horizontal bar chart of the 20 largest connectivity changes."""
    change_col = f"Change_{comparison}"
    labels  = [f"{row['Region_1']} <-> {row['Region_2']}" for _, row in df_top20.iterrows()]
    changes = df_top20[change_col].values
    colors  = ["#d62728" if c < 0 else "#2ca02c" for c in changes]
    y_pos   = np.arange(len(labels))

    session_text = {"1v2": "Ses 1 -> Ses 2", "1v3": "Ses 1 -> Ses 3", "2v3": "Ses 2 -> Ses 3"}

    fig, ax = plt.subplots(figsize=(16, 12))
    bars = ax.barh(y_pos, changes, color=colors, alpha=0.8, edgecolor="black", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_xlabel("Connectivity Change (delta r)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Top 20 Connectivity Changes: {task_name.upper()}\n{session_text.get(comparison, comparison)}",
        fontsize=16, fontweight="bold", pad=20,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=2)
    ax.grid(axis="x", alpha=0.4, linestyle="--")

    for bar, change in zip(bars, changes):
        w = bar.get_width()
        ax.text(
            w + (0.005 if w >= 0 else -0.005), bar.get_y() + bar.get_height() / 2,
            f"{change:+.3f}", va="center", ha="left" if w >= 0 else "right",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="black"),
        )

    legend_elems = [
        Patch(facecolor="#2ca02c", label="Increased", alpha=0.8),
        Patch(facecolor="#d62728", label="Decreased", alpha=0.8),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=11, framealpha=0.9)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task_name}_{comparison}_TOP20.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(out_path)}")


# ---------------------------------------------------------------------------
# Task-level analysis
# ---------------------------------------------------------------------------

def analyze_task(
    task_name: str,
    ses1_file: str,
    ses2_file: str,
    ses3_file: str,
    out_dir: str,
) -> None:
    print(f"\n{'='*70}")
    print(f"  {task_name.upper()}")
    print(f"{'='*70}")

    ses1_df = load_timeseries(ses1_file)
    ses2_df = load_timeseries(ses2_file)
    ses3_df = load_timeseries(ses3_file) if ses3_file and os.path.isfile(ses3_file) else None

    if ses1_df is None or ses2_df is None:
        print(f"  [SKIP] Missing session 1 or 2 data for {task_name}")
        return

    print(f"  Ses1 shape: {ses1_df.shape}  Ses2 shape: {ses2_df.shape}", end="")
    print(f"  Ses3 shape: {ses3_df.shape}" if ses3_df is not None else "  Ses3: N/A")

    df_changes = compute_connectivity_changes(ses1_df, ses2_df, ses3_df)
    os.makedirs(out_dir, exist_ok=True)

    full_csv = os.path.join(out_dir, f"{task_name}_all_changes.csv")
    df_changes.to_csv(full_csv, index=False)
    print(f"  All {len(df_changes)} connections -> {os.path.basename(full_csv)}")

    for comp, abs_col in [("1v2", "Abs_Change_1v2"), ("1v3", "Abs_Change_1v3")]:
        if comp == "1v3" and ses3_df is None:
            continue
        top20 = df_changes.nlargest(20, abs_col)
        csv_path = os.path.join(out_dir, f"{task_name}_{comp}_TOP20.csv")
        top20.to_csv(csv_path, index=False)
        print(f"\n  Top-10 changes ({comp}):")
        for _, row in top20.head(10).iterrows():
            print(f"    {row['Region_1']} <-> {row['Region_2']}: {row[f'Change_{comp}']:+.3f}")
        plot_top20(top20, task_name, comp, out_dir)


# ---------------------------------------------------------------------------
# Auto-discovery helper
# ---------------------------------------------------------------------------

def _find_sessions(ts_dir: str, subject: str, task: str, atlas: str):
    """Return (ses1_file, ses2_file, ses3_file) paths (may be None if not found)."""
    def _find(ses: str):
        pattern = os.path.join(ts_dir, f"{subject}_{ses}_task-{task}*_{atlas}_ts.csv")
        hits = sorted(glob.glob(pattern))
        return hits[0] if hits else None

    return _find("ses-1"), _find("ses-2"), _find("ses-3")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",    default=DEFAULT_TS_DIR, help="Timeseries directory")
    parser.add_argument("--out",     default=DEFAULT_OUT,    help="Output directory")
    parser.add_argument("--subject", default="sub-001",      help="Subject ID")
    parser.add_argument("--atlas",   default="schaefer100",  help="Atlas suffix (e.g. schaefer100, ho_cortical)")
    args = parser.parse_args()

    for task in ("rest", "music"):
        s1, s2, s3 = _find_sessions(args.data, args.subject, task, args.atlas)
        if s1 is None and s2 is None:
            print(f"  [SKIP] No files found for {args.subject} task-{task} in {args.data}")
            continue
        analyze_task(
            task_name = f"{args.subject}_{task}_{args.atlas}",
            ses1_file = s1 or "",
            ses2_file = s2 or "",
            ses3_file = s3,
            out_dir   = args.out,
        )

    print(f"\nDone. Results in {args.out}")


if __name__ == "__main__":
    main()
