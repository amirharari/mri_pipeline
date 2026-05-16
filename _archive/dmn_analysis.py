"""
DMN Connectivity Analysis
=========================

Two analyses in one file:
  analyze_dmn_full_matrix()  – full pairwise DMN correlation matrix + session comparison
  analyze_dmn_within_network() – within-DMN mean connectivity (Fisher-z) + statistics

Merged from: dmn_connectivity_analysis.py + dmn_within_network_analysis.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

from utils import load_timeseries, identify_network_columns, within_network_connectivity

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "dmn_analysis_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _short_label(col: str) -> str:
    return col.replace("7Networks_", "").replace("_", " ")


def _load_pair(ses1_file: str, ses2_file: str):
    ses1 = load_timeseries(ses1_file)
    ses2 = load_timeseries(ses2_file)
    if ses1 is None or ses2 is None:
        raise FileNotFoundError(f"Missing: {ses1_file} or {ses2_file}")
    return ses1, ses2


# ---------------------------------------------------------------------------
# Full matrix analysis
# ---------------------------------------------------------------------------

def analyze_dmn_full_matrix(ses1_file: str, ses2_file: str, task_name: str) -> pd.DataFrame:
    """
    Full pairwise DMN connectivity matrix comparison (Session 2 vs Session 1).
    Saves heatmaps (1×3: ses1 | ses2 | diff) and a per-connection CSV.
    """
    print(f"\n{'='*60}\nFull-matrix DMN: {task_name}\n{'='*60}")

    ses1_df, ses2_df = _load_pair(ses1_file, ses2_file)
    dmn_cols = identify_network_columns(ses1_df, "Default")

    if not dmn_cols:
        print("  No DMN columns found – skipping.")
        return pd.DataFrame()

    print(f"  {len(dmn_cols)} DMN regions")

    ses1_conn = ses1_df[dmn_cols].corr()
    ses2_conn = ses2_df[dmn_cols].corr()
    diff_matrix = ses2_conn - ses1_conn

    n = len(dmn_cols)
    idx = np.triu_indices(n, k=1)
    ses1_vals = ses1_conn.values[idx]
    ses2_vals = ses2_conn.values[idx]
    diff_vals = diff_matrix.values[idx]

    t_stat, p_value = ttest_rel(ses2_vals, ses1_vals)
    print(f"  Mean Δ: {diff_vals.mean():+.4f}  |  t={t_stat:.4f}  p={p_value:.4f}")
    if p_value < 0.05:
        print("  *** Significant change (p < 0.05) ***")

    # --- Plot ---
    short = [_short_label(c) for c in dmn_cols]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    for ax, mat, title in zip(
        axes,
        [ses1_conn, ses2_conn, diff_matrix],
        [f"Session 1 – {task_name}", f"Session 2 – {task_name}", f"Δ (Ses2−Ses1) – {task_name}"],
    ):
        cmap = "coolwarm" if "Δ" not in title else "RdBu_r"
        vmax_v = 1 if "Δ" not in title else max(abs(mat.min().min()), abs(mat.max().max()))
        vmin_v = -1 if "Δ" not in title else -vmax_v
        sns.heatmap(mat, ax=ax, cmap=cmap, center=0, vmin=vmin_v, vmax=vmax_v,
                    square=True, cbar_kws={"label": "r" if "Δ" not in title else "Δr"})
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short, rotation=0, fontsize=8)

    plt.tight_layout()
    png = os.path.join(OUTPUT_DIR, f"dmn_full_matrix_{task_name}.png")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {png}")

    # --- CSV ---
    results = pd.DataFrame({
        "Region_1": [dmn_cols[i] for i, j in zip(*idx)],
        "Region_2": [dmn_cols[j] for i, j in zip(*idx)],
        "Ses1_Connectivity": ses1_vals,
        "Ses2_Connectivity": ses2_vals,
        "Change": diff_vals,
    }).sort_values("Change", key=abs, ascending=False)

    csv = os.path.join(OUTPUT_DIR, f"dmn_full_matrix_changes_{task_name}.csv")
    results.to_csv(csv, index=False)
    print(f"  Saved: {csv}")
    return results


# ---------------------------------------------------------------------------
# Within-network analysis
# ---------------------------------------------------------------------------

def analyze_dmn_within_network(ses1_file: str, ses2_file: str, task_name: str) -> pd.DataFrame:
    """
    Within-DMN mean connectivity using Fisher-z transformation.
    Saves a 4-panel figure and a summary CSV.
    """
    print(f"\n{'='*70}\nWithin-DMN: {task_name}\n{'='*70}")

    ses1_df, ses2_df = _load_pair(ses1_file, ses2_file)
    dmn_cols = identify_network_columns(ses1_df, "Default")

    if len(dmn_cols) < 2:
        print("  Fewer than 2 DMN columns – skipping.")
        return pd.DataFrame()

    print(f"  {len(dmn_cols)} regions  |  {len(dmn_cols)*(len(dmn_cols)-1)//2} connections")

    ses1_stats = within_network_connectivity(ses1_df, dmn_cols)
    ses2_stats = within_network_connectivity(ses2_df, dmn_cols)

    t_stat, p_value = ttest_rel(ses2_stats["fisher_z_values"], ses1_stats["fisher_z_values"])
    diff = ses2_stats["fisher_z_values"] - ses1_stats["fisher_z_values"]
    cohens_d = float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else 0.0
    change = ses2_stats["mean"] - ses1_stats["mean"]

    # Print summary
    for ses_name, st in [("Session 1", ses1_stats), ("Session 2", ses2_stats)]:
        print(f"  {ses_name}: mean={st['mean']:.4f}  median={st['median']:.4f}  std={st['std']:.4f}")
    print(f"  Change: {change:+.4f}  ({(change/ses1_stats['mean']*100) if ses1_stats['mean'] else 0:+.1f}%)")
    print(f"  Cohen's d: {cohens_d:.3f}  |  t={t_stat:.4f}  p={p_value:.6f}")

    # --- 4-panel figure ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Bar chart
    ax = axes[0, 0]
    bars = ax.bar(["Session 1", "Session 2"],
                  [ses1_stats["mean"], ses2_stats["mean"]],
                  yerr=[ses1_stats["std"], ses2_stats["std"]],
                  capsize=10, alpha=0.7, color=["steelblue", "coral"])
    ax.set_ylabel("Mean Within-DMN Connectivity")
    ax.set_title(f"{task_name}: Within-DMN Connectivity")
    for bar, m, s in zip(bars, [ses1_stats["mean"], ses2_stats["mean"]],
                         [ses1_stats["std"], ses2_stats["std"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontweight="bold")

    # Histogram
    ax = axes[0, 1]
    ax.hist(ses1_stats["values"], bins=30, alpha=0.6, label="Session 1", color="steelblue", density=True)
    ax.hist(ses2_stats["values"], bins=30, alpha=0.6, label="Session 2", color="coral", density=True)
    ax.axvline(ses1_stats["mean"], color="steelblue", linestyle="--", linewidth=2)
    ax.axvline(ses2_stats["mean"], color="coral", linestyle="--", linewidth=2)
    ax.set_xlabel("Correlation Strength")
    ax.set_title("Distribution of Within-DMN Connections")
    ax.legend()
    ax.grid(alpha=0.3)

    # Box plot
    ax = axes[1, 0]
    bp = ax.boxplot([ses1_stats["values"], ses2_stats["values"]],
                    labels=["Session 1", "Session 2"], patch_artist=True, widths=0.6)
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][1].set_facecolor("coral")
    ax.set_ylabel("Correlation Strength")
    ax.set_title("Within-DMN Connectivity Distribution")
    ax.grid(axis="y", alpha=0.3)
    ax.plot([1], [ses1_stats["mean"]], "D", color="darkblue", markersize=10, zorder=3)
    ax.plot([2], [ses2_stats["mean"]], "D", color="darkred", markersize=10, zorder=3)

    # Difference histogram
    ax = axes[1, 1]
    raw_diffs = ses2_stats["values"] - ses1_stats["values"]
    ax.hist(raw_diffs, bins=30, alpha=0.7, color="purple", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="No Change")
    ax.axvline(np.mean(raw_diffs), color="green", linestyle="-", linewidth=2,
               label=f"Mean Δ: {np.mean(raw_diffs):+.3f}")
    ax.set_xlabel("Δ Connectivity (Ses2 – Ses1)")
    ax.set_title("Distribution of Connectivity Changes")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    png = os.path.join(OUTPUT_DIR, f"within_dmn_{task_name}.png")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {png}")

    summary = pd.DataFrame([{
        "Task": task_name,
        "Ses1_Mean": ses1_stats["mean"],
        "Ses2_Mean": ses2_stats["mean"],
        "Change": change,
        "Cohens_d": cohens_d,
        "t_statistic": t_stat,
        "p_value": p_value,
        "Significant": p_value < 0.05,
    }])
    csv = os.path.join(OUTPUT_DIR, f"within_dmn_summary_{task_name}.csv")
    summary.to_csv(csv, index=False)
    print(f"  Saved: {csv}")
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tasks = [
        ("rest",
         "sub-001_ses-1_task-rest_schaefer100_ts.csv",
         "sub-001_ses-2_task-rest_schaefer100_ts.csv"),
        ("music_run-2",
         "sub-001_ses-1_task-music_run-2_schaefer100_ts.csv",
         "sub-001_ses-2_task-music_run-2_schaefer100_ts.csv"),
        ("music_run-3",
         "sub-001_ses-1_task-music_run-3_schaefer100_ts.csv",
         "sub-001_ses-2_task-music_run-3_schaefer100_ts.csv"),
    ]

    summaries_full = []
    summaries_within = []

    for task_name, f1, f2 in tasks:
        if not (os.path.exists(f1) and os.path.exists(f2)):
            print(f"\nSkipping {task_name} – files not found")
            continue
        summaries_full.append(analyze_dmn_full_matrix(f1, f2, task_name))
        summaries_within.append(analyze_dmn_within_network(f1, f2, task_name))

    if summaries_within:
        combined = pd.concat(summaries_within, ignore_index=True)
        out = os.path.join(OUTPUT_DIR, "all_tasks_within_dmn_summary.csv")
        combined.to_csv(out, index=False)
        print(f"\nCombined within-network summary: {out}")
        print(combined.to_string(index=False))

    print(f"\nDone. Results in {OUTPUT_DIR}/")
