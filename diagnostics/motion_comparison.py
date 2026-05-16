"""
Motion quality check across all subjects and sessions.

Verifies that enhanced connectivity under psilocybin is not driven by
higher head motion. Reads the unified scrubbing report produced by the
extraction pipeline.

Usage
-----
    python motion_comparison.py                    # uses config.TS_OUTPUT_DIR
    python motion_comparison.py --dir PATH/TO/CSV  # custom directory
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TS_OUTPUT_DIR, OUTPUTS_DIR  # noqa: E402

SESSION_ORDER  = ["ses-1", "ses-2", "ses-3"]
SESSION_LABELS = {"ses-1": "Baseline", "ses-2": "Psilocybin", "ses-3": "Follow-up"}
SESSION_COLORS = {"ses-1": "#2196F3", "ses-2": "#F44336",     "ses-3": "#4CAF50"}
LABEL_ORDER    = ["Baseline", "Psilocybin", "Follow-up"]

OUT_DIR = os.path.join(OUTPUTS_DIR, "motion_qc")
os.makedirs(OUT_DIR, exist_ok=True)


def load_report(ts_dir: str) -> pd.DataFrame:
    """Load the unified scrubbing report from *ts_dir*."""
    path = os.path.join(ts_dir, "scrubbing_report_filtered.csv")
    if not os.path.isfile(path):
        print(f"  [ERROR] Scrubbing report not found: {path}")
        print("  Run the extraction pipeline first (python pipeline.py).")
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["session_label"] = df["session"].map(SESSION_LABELS)
    df["is_rest"]       = df["task"] == "rest"
    return df


def print_tables(df: pd.DataFrame) -> None:
    fd_col = next((c for c in ["fd_mean_filtered", "fd_mean_raw", "fd_mean"] if c in df.columns), None)
    if fd_col is None:
        print("  [WARN] No FD column found in report.")
        return

    print("=" * 75)
    print("PER-RUN MOTION SUMMARY")
    print("=" * 75)
    cols = [c for c in ["subject", "session_label", "task", "run",
                        fd_col, "fd_max_filtered", "scrub_percent", "total_vols"]
            if c in df.columns]
    tmp = df.copy()
    tmp["_ses"] = tmp["session"].map({"ses-1": 1, "ses-2": 2, "ses-3": 3}).fillna(9)
    print(tmp.sort_values(["subject", "_ses", "task"])[cols].to_string(index=False))

    print("\n" + "=" * 75)
    print(f"MEAN {fd_col.upper()} BY SUBJECT x SESSION")
    print("=" * 75)
    pivot = df.pivot_table(values=fd_col, index="subject",
                           columns="session_label", aggfunc="mean").reindex(columns=LABEL_ORDER, fill_value=float("nan"))
    print(pivot.round(3).to_string())
    print()
    for sub in pivot.index:
        base  = pivot.loc[sub, "Baseline"]  if "Baseline"   in pivot.columns else float("nan")
        psilo = pivot.loc[sub, "Psilocybin"] if "Psilocybin" in pivot.columns else float("nan")
        if not (pd.isna(base) or pd.isna(psilo)):
            diff = psilo - base
            flag = " *** HIGHER MOTION IN PSILO ***" if diff > 0.02 else " OK"
            print(f"  {sub}: Psilo - Baseline = {diff:+.3f} mm{flag}")

    print("\n" + "=" * 75)
    print("MOTION CONFOUND VERDICT")
    print("=" * 75)
    any_confound = False
    for sub in pivot.index:
        base  = pivot.loc[sub, "Baseline"]   if "Baseline"   in pivot.columns else float("nan")
        psilo = pivot.loc[sub, "Psilocybin"] if "Psilocybin" in pivot.columns else float("nan")
        if not (pd.isna(base) or pd.isna(psilo)) and psilo > base + 0.02:
            print(f"  {sub}: POTENTIAL CONFOUND — psilocybin {psilo:.3f} > baseline {base:.3f}")
            any_confound = True
    if not any_confound:
        print("  No subject shows meaningfully higher motion during psilocybin session.")
        print("  Enhanced connectivity is unlikely to be motion-driven.")


def plot_motion_figure(df: pd.DataFrame) -> None:
    fd_col  = next((c for c in ["fd_mean_filtered", "fd_mean_raw", "fd_mean"] if c in df.columns), None)
    fmax    = next((c for c in ["fd_max_filtered",  "fd_max_raw",  "fd_max"]  if c in df.columns), None)
    if fd_col is None:
        return

    subjects = sorted(df["subject"].unique())
    n_subs   = len(subjects)
    fig, axes = plt.subplots(2, n_subs, figsize=(5 * n_subs, 9), constrained_layout=True)
    fig.suptitle("Motion QC — All Subjects & Sessions", fontsize=14, fontweight="bold")

    # Make axes always 2-D
    if n_subs == 1:
        axes = axes.reshape(2, 1)

    for col_i, sub in enumerate(subjects):
        sdf = df[df["subject"] == sub].copy()
        sdf["_ord"] = sdf["session"].map({"ses-1": 0, "ses-2": 1, "ses-3": 2}).fillna(3)
        sdf = sdf.sort_values("_ord")

        # Panel A — FD scatter
        ax = axes[0, col_i]
        for _, row in sdf.iterrows():
            color = SESSION_COLORS.get(row["session"], "grey")
            y_val = row[fmax] if fmax else row[fd_col]
            ax.scatter(row[fd_col], y_val, color=color, s=80, zorder=3, alpha=0.85,
                       edgecolors="k", linewidths=0.5)
        ax.axvline(0.2, color="orange", lw=1.2, ls="--", label="FD mean = 0.2")
        ax.set_xlabel(f"{fd_col} (mm)", fontsize=9)
        ax.set_ylabel(f"{fmax or fd_col} (mm)", fontsize=9)
        ax.set_title(f"{sub}\nMean vs Max FD per run", fontsize=10)
        patches = [mpatches.Patch(color=SESSION_COLORS[s], label=SESSION_LABELS[s])
                   for s in SESSION_ORDER]
        ax.legend(handles=patches, fontsize=7, loc="upper left")

        # Panel B — mean FD per session bar
        ax2 = axes[1, col_i]
        sess_fd = sdf.groupby("session_label")[fd_col].mean().reindex(LABEL_ORDER)
        colors  = [SESSION_COLORS[s] for s in SESSION_ORDER]
        bars    = ax2.bar(LABEL_ORDER, sess_fd.values, color=colors, width=0.5,
                          edgecolor="k", linewidth=0.7)
        for bar, val in zip(bars, sess_fd.values):
            if not pd.isna(val):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax2.axhline(0.2, color="orange", lw=1.2, ls="--")
        ax2.set_ylabel("Mean FD (mm)", fontsize=9)
        ax2.set_title(f"{sub}\nMean FD by Session", fontsize=10)
        ax2.set_ylim(0, max(sess_fd.dropna().max() * 1.25, 0.25))

    out_path = os.path.join(OUT_DIR, "motion_qc_all_subjects.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {out_path}")


def plot_fd_distribution(df: pd.DataFrame) -> None:
    fd_col = next((c for c in ["fd_mean_filtered", "fd_mean_raw", "fd_mean"] if c in df.columns), None)
    if fd_col is None:
        return

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    data, labels_x, colors_x = [], [], []
    for ses in SESSION_ORDER:
        vals = df[df["session"] == ses][fd_col].dropna().values
        data.append(vals)
        labels_x.append(SESSION_LABELS[ses])
        colors_x.append(SESSION_COLORS[ses])

    parts = ax.violinplot(data, positions=range(3), widths=0.5, showmedians=True)
    for pc, color in zip(parts["bodies"], colors_x):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    for i, (vals, color) in enumerate(zip(data, colors_x)):
        ax.scatter(np.random.normal(i, 0.06, size=len(vals)), vals,
                   color=color, s=60, zorder=3, edgecolors="k", linewidths=0.5, alpha=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels_x, fontsize=11)
    ax.axhline(0.2, color="orange", lw=1.5, ls="--", label="High-motion threshold (0.2 mm)")
    ax.set_ylabel(f"{fd_col} (mm)", fontsize=11)
    ax.set_title("FD Distribution by Session (all subjects pooled)", fontsize=12)
    ax.legend(fontsize=9)

    out_path = os.path.join(OUT_DIR, "fd_distribution_by_session.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=TS_OUTPUT_DIR,
                        help="Directory containing scrubbing_report_filtered.csv")
    args = parser.parse_args()

    df = load_report(args.dir)
    print_tables(df)
    plot_motion_figure(df)
    plot_fd_distribution(df)
    print("\nAll done.")
