"""
Within-Network Connectivity Analysis
=====================================

Two modes:
  analyze_2sessions()  – compare Session 1 vs Session 2 (4-panel figure + CSV)
  analyze_3sessions()  – compare Sessions 1, 2 and 3 (grouped bar chart)

Run with: python network_within_analysis.py

Merged from: all_networks_within_analysis.py + network_activation_3sessions.py
"""
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import YEO_NETWORKS, load_and_concatenate, identify_network_columns, within_network_connectivity

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "network_within_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 2-session analysis
# ---------------------------------------------------------------------------

def _compute_network_stats(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Return per-network stats comparing two session DataFrames."""
    rows = []
    for net_name, keyword in YEO_NETWORKS.items():
        cols = identify_network_columns(df1, keyword)
        if len(cols) < 2:
            continue
        s1 = within_network_connectivity(df1, cols)
        s2 = within_network_connectivity(df2, cols)
        if s1 is None or s2 is None:
            continue
        change = s2["mean"] - s1["mean"]
        pct = (change / s1["mean"] * 100) if s1["mean"] != 0 else 0.0
        rows.append({
            "Network":         net_name,
            "Ses1_Mean":       s1["mean"],
            "Ses1_Std":        s1["std"],
            "Ses2_Mean":       s2["mean"],
            "Ses2_Std":        s2["std"],
            "Change":          change,
            "Percent_Change":  pct,
            "N_Regions":       s1["n_regions"],
            "N_Connections":   s1["n_connections"],
        })
    return pd.DataFrame(rows)


def _plot_2session(results_df: pd.DataFrame, condition_name: str, out_dir: str) -> None:
    """4-panel figure for 2-session comparison."""
    networks       = results_df["Network"].values
    ses1_means     = results_df["Ses1_Mean"].values
    ses2_means     = results_df["Ses2_Mean"].values
    changes        = results_df["Change"].values
    percent_changes = results_df["Percent_Change"].values

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Side-by-side bars
    ax = axes[0, 0]
    x, w = np.arange(len(networks)), 0.35
    ax.bar(x - w / 2, ses1_means, w, label="Session 1", alpha=0.8, color="steelblue")
    ax.bar(x + w / 2, ses2_means, w, label="Session 2", alpha=0.8, color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(networks, rotation=45, ha="right")
    ax.set_ylabel("Mean Within-Network Connectivity")
    ax.set_title(f"{condition_name}: Within-Network Connectivity")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Absolute change
    ax = axes[0, 1]
    colors = ["green" if c >= 0 else "red" for c in changes]
    bars = ax.bar(networks, changes, alpha=0.7, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticklabels(networks, rotation=45, ha="right")
    ax.set_ylabel("Δ Connectivity (Ses2 – Ses1)")
    ax.set_title("Connectivity Changes by Network")
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                f"{h:+.3f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=9, fontweight="bold")

    # Percent change
    ax = axes[1, 0]
    colors = ["green" if c >= 0 else "red" for c in percent_changes]
    bars = ax.bar(networks, percent_changes, alpha=0.7, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticklabels(networks, rotation=45, ha="right")
    ax.set_ylabel("% Change")
    ax.set_title("Percent Change in Connectivity")
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                f"{h:+.1f}%", ha="center", va="bottom" if h >= 0 else "top", fontsize=9, fontweight="bold")

    # Summary table
    ax = axes[1, 1]
    ax.axis("off")
    table_data = [
        [row["Network"], f"{row['Ses1_Mean']:.3f}", f"{row['Ses2_Mean']:.3f}",
         f"{row['Change']:+.3f}", f"{row['Percent_Change']:+.1f}%"]
        for _, row in results_df.iterrows()
    ]
    tbl = ax.table(cellText=table_data,
                   colLabels=["Network", "Ses1", "Ses2", "Δ", "%Δ"],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2)
    for i in range(5):
        tbl[(0, i)].set_facecolor("#4472C4")
        tbl[(0, i)].set_text_props(weight="bold", color="white")
    for i, row in enumerate(table_data):
        bg = "#ccffcc" if float(row[3]) >= 0 else "#ffcccc"
        tbl[(i + 1, 3)].set_facecolor(bg)
        tbl[(i + 1, 4)].set_facecolor(bg)

    plt.tight_layout()
    png = os.path.join(out_dir, f"within_network_{condition_name}.png")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {png}")


def analyze_2sessions(
    ses1_files: List[str],
    ses2_files: List[str],
    condition_name: str,
    out_dir: str = OUTPUT_DIR,
) -> Optional[pd.DataFrame]:
    """
    Compare all Yeo-7 networks between two sessions (or concatenated file lists).
    Returns a results DataFrame.
    """
    print(f"\n{'='*80}\n2-session analysis: {condition_name}\n{'='*80}")

    ses1_df = load_and_concatenate(ses1_files)
    ses2_df = load_and_concatenate(ses2_files)
    if ses1_df is None or ses2_df is None:
        print("  Insufficient data – skipping.")
        return None

    print(f"  Ses1: {ses1_df.shape[0]} tp  |  Ses2: {ses2_df.shape[0]} tp")

    results = _compute_network_stats(ses1_df, ses2_df)
    _plot_2session(results, condition_name, out_dir)

    csv = os.path.join(out_dir, f"within_network_{condition_name}.csv")
    results.to_csv(csv, index=False)
    print(f"  Saved: {csv}")
    return results


# ---------------------------------------------------------------------------
# 3-session analysis
# ---------------------------------------------------------------------------

def _plot_3session(results_df: pd.DataFrame, condition_name: str, out_dir: str) -> None:
    """Grouped bar chart for 3 sessions."""
    networks = results_df["Network"].values
    v1 = results_df["Session_1"].values
    v2 = results_df["Session_2"].values
    v3 = results_df["Session_3"].values
    has1 = results_df["Has_Ses1"].values
    has2 = results_df["Has_Ses2"].values
    has3 = results_df["Has_Ses3"].values

    fig, ax = plt.subplots(figsize=(14, 8))
    x, w = np.arange(len(networks)), 0.25
    ax.bar(x - w,  v1, w, label="Session 1", alpha=0.9, color="steelblue",     edgecolor="black")
    ax.bar(x,      v2, w, label="Session 2", alpha=0.9, color="coral",          edgecolor="black")
    ax.bar(x + w,  v3, w, label="Session 3", alpha=0.9, color="mediumseagreen", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(networks, rotation=45, ha="right", fontsize=11, fontweight="bold")
    ax.set_xlabel("Network", fontsize=14, fontweight="bold")
    ax.set_ylabel("Within-Network Connectivity", fontsize=14, fontweight="bold")
    ax.set_title(f"{condition_name.upper()}\nWithin-Network Connectivity Across Sessions",
                 fontsize=16, fontweight="bold", pad=20)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    max_v = max(v1.max(), v2.max(), v3.max())
    ax.set_ylim([0, max(0.1, max_v) * 1.15])

    for vals, has, bars_x in [(v1, has1, x - w), (v2, has2, x), (v3, has3, x + w)]:
        for i, (h, ok) in enumerate(zip(vals, has)):
            if ok and h > 0:
                ax.text(bars_x[i] + w / 2, h + 0.01, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    for i, net in enumerate(networks):
        if has1[i] and has3[i]:
            trend = v3[i] - v1[i]
        elif has1[i] and has2[i]:
            trend = v2[i] - v1[i]
        elif has2[i] and has3[i]:
            trend = v3[i] - v2[i]
        else:
            continue
        y_base = max(v1[i], v2[i], v3[i]) * 1.08
        arrow = "↑" if trend > 0.05 else ("↓" if trend < -0.05 else "→")
        color = "green" if trend > 0.05 else ("red" if trend < -0.05 else "gray")
        ax.text(i, y_base, arrow, ha="center", va="bottom", fontsize=18, color=color, fontweight="bold")

    plt.tight_layout()
    png = os.path.join(out_dir, f"{condition_name}_3sessions.png")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {png}")


def analyze_3sessions(
    ses1_files: List[str],
    ses2_files: List[str],
    ses3_files: List[str],
    condition_name: str,
    out_dir: str = OUTPUT_DIR,
) -> Optional[pd.DataFrame]:
    """
    Compare all Yeo-7 networks across three sessions (lists of files per session).
    Returns a results DataFrame.
    """
    print(f"\n{'='*80}\n3-session analysis: {condition_name}\n{'='*80}")

    ses1_df = load_and_concatenate(ses1_files)
    ses2_df = load_and_concatenate(ses2_files)
    ses3_df = load_and_concatenate(ses3_files)

    available = sum(df is not None for df in [ses1_df, ses2_df, ses3_df])
    if available < 2:
        print("  Need at least 2 sessions – skipping.")
        return None

    ref = next(df for df in [ses1_df, ses2_df, ses3_df] if df is not None)
    rows = []
    for net_name, keyword in YEO_NETWORKS.items():
        cols = identify_network_columns(ref, keyword)
        if len(cols) < 2:
            continue
        r1 = within_network_connectivity(ses1_df, cols)["mean"] if ses1_df is not None else None
        r2 = within_network_connectivity(ses2_df, cols)["mean"] if ses2_df is not None else None
        r3 = within_network_connectivity(ses3_df, cols)["mean"] if ses3_df is not None else None
        if sum(v is not None for v in [r1, r2, r3]) < 2:
            continue
        rows.append({
            "Network":  net_name,
            "Session_1": r1 if r1 is not None else 0.0,
            "Session_2": r2 if r2 is not None else 0.0,
            "Session_3": r3 if r3 is not None else 0.0,
            "Has_Ses1":  r1 is not None,
            "Has_Ses2":  r2 is not None,
            "Has_Ses3":  r3 is not None,
        })

    results = pd.DataFrame(rows)
    _plot_3session(results, condition_name, out_dir)

    csv = os.path.join(out_dir, f"{condition_name}_3sessions.csv")
    results.to_csv(csv, index=False)
    print(f"  Saved: {csv}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("WITHIN-NETWORK CONNECTIVITY ANALYSIS — 2 and 3 sessions")
    print("=" * 80)

    all_2ses = []

    # --- 2-session: Rest ---
    r = analyze_2sessions(
        ["sub-001_ses-1_task-rest_schaefer100_ts.csv"],
        ["sub-001_ses-2_task-rest_schaefer100_ts.csv"],
        "rest",
    )
    if r is not None:
        r["Condition"] = "Rest"
        all_2ses.append(r)

    # --- 2-session: Combined ---
    r = analyze_2sessions(
        [
            "sub-001_ses-1_task-rest_schaefer100_ts.csv",
            "sub-001_ses-1_task-glass_schaefer100_ts.csv",
            "sub-001_ses-1_task-music_run-2_schaefer100_ts.csv",
            "sub-001_ses-1_task-music_run-3_schaefer100_ts.csv",
            "sub-001_ses-1_task-music_run-4_schaefer100_ts.csv",
            "sub-001_ses-1_task-music_run-5_schaefer100_ts.csv",
        ],
        [
            "sub-001_ses-2_task-rest_schaefer100_ts.csv",
            "sub-001_ses-2_task-glass_schaefer100_ts.csv",
            "sub-001_ses-2_task-music_run-1_schaefer100_ts.csv",
            "sub-001_ses-2_task-music_run-2_schaefer100_ts.csv",
            "sub-001_ses-2_task-music_run-3_schaefer100_ts.csv",
        ],
        "combined_all",
    )
    if r is not None:
        r["Condition"] = "Combined"
        all_2ses.append(r)

    if all_2ses:
        master = pd.concat(all_2ses, ignore_index=True)
        out = os.path.join(OUTPUT_DIR, "master_summary_all_networks.csv")
        master.to_csv(out, index=False)
        print(f"\nMaster 2-session summary: {out}")

    # --- 3-session: Rest ---
    analyze_3sessions(
        ["sub-001_ses-1_task-rest_schaefer100_ts.csv"],
        ["sub-001_ses-2_task-rest_schaefer100_ts.csv"],
        ["sub-001_ses-3_task-rest_schaefer100_ts.csv"],
        "REST",
    )

    # --- 3-session: Combined ---
    analyze_3sessions(
        [
            "sub-001_ses-1_task-rest_schaefer100_ts.csv",
            "sub-001_ses-1_task-music_run-2_schaefer100_ts.csv",
            "sub-001_ses-1_task-music_run-3_schaefer100_ts.csv",
            "sub-001_ses-1_task-music_run-4_schaefer100_ts.csv",
            "sub-001_ses-1_task-music_run-5_schaefer100_ts.csv",
        ],
        [
            "sub-001_ses-2_task-rest_schaefer100_ts.csv",
            "sub-001_ses-2_task-music_run-1_schaefer100_ts.csv",
            "sub-001_ses-2_task-music_run-2_schaefer100_ts.csv",
            "sub-001_ses-2_task-music_run-3_schaefer100_ts.csv",
        ],
        [
            "sub-001_ses-3_task-rest_schaefer100_ts.csv",
            "sub-001_ses-3_task-music_run-1_schaefer100_ts.csv",
            "sub-001_ses-3_task-music_run-3_schaefer100_ts.csv",
            "sub-001_ses-3_task-music_run-4_schaefer100_ts.csv",
            "sub-001_ses-3_task-music_run-5_schaefer100_ts.csv",
            "sub-001_ses-3_task-music_run-6_schaefer100_ts.csv",
            "sub-001_ses-3_task-music_run-7_schaefer100_ts.csv",
        ],
        "COMBINED",
    )

    print(f"\nDone. Results in {OUTPUT_DIR}/")
