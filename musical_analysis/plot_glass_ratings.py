"""
Plot average continuous valence and arousal ratings for the Glass excerpt.

Data source: Stable_meanRatings-Glass.xlsx
  GH_VMeanThr 0Alls  — per-sample mean valence + SE (423 timepoints)
  GH_AMeanThr 0Alls  — per-sample mean arousal  + SE (423 timepoints)

Output: outputs/musical_analysis/glass_valence_arousal.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402

XLSX   = r"C:\Users\amirh\Downloads\Stable_meanRatings-Glass.xlsx"
OUT_DIR = os.path.join(config.OUTPUTS_DIR, "musical_analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def load_sheet(path: str, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    df.columns = ["subj_included", "subj_excluded", "mean", "se"]
    # First row is header string — drop it
    df = df[df["mean"].apply(lambda x: isinstance(x, (int, float)))].copy()
    df = df.reset_index(drop=True)
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["se"]   = pd.to_numeric(df["se"],   errors="coerce")
    return df


def main() -> None:
    df_v = load_sheet(XLSX, "GH_VMeanThr 0Alls")
    df_a = load_sheet(XLSX, "GH_AMeanThr 0Alls")

    n = min(len(df_v), len(df_a))
    t = np.arange(n)   # sample index (= time in samples)

    print(f"Loaded {n} timepoints")
    print(f"Valence  mean range: [{df_v['mean'].min():.3f}, {df_v['mean'].max():.3f}]")
    print(f"Arousal  mean range: [{df_a['mean'].min():.3f}, {df_a['mean'].max():.3f}]")

    # ── figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 11})

    panel_cfg = [
        (axes[0], df_v, "Valence",  "#E94E77", "A"),
        (axes[1], df_a, "Arousal",  "#4A90D9", "B"),
    ]

    for ax, df, label, color, panel in panel_cfg:
        mean = df["mean"].values[:n]
        se   = df["se"].values[:n]

        ax.fill_between(t, mean - se, mean + se,
                        alpha=0.25, color=color, linewidth=0)
        ax.plot(t, mean, color=color, linewidth=1.6, label=f"Mean {label}")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)

        ax.set_ylabel(f"{label} (a.u.)", fontsize=12)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.spines[["top", "right"]].set_visible(False)

        # Panel label
        ax.text(-0.04, 1.05, panel, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")

        # Annotation: global mean
        gmean = float(np.nanmean(mean))
        ax.text(0.99, 0.93,
                f"Global mean = {gmean:+.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec=color))

    axes[1].set_xlabel("Sample (time)", fontsize=12)

    fig.suptitle(
        "Continuous Valence & Arousal Ratings — Glass Excerpt\n"
        "Mean ± SE across included subjects",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "glass_valence_arousal.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
