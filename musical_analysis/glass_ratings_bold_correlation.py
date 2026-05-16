"""
Correlate normative Glass valence/arousal (group ratings Excel) with
group-averaged BOLD in limbic vs auditory-relevant (SomMot) Schaefer parcels.

Method (exploratory)
---------------------
1. Load mean valence & arousal time courses from Stable_meanRatings-Glass.xlsx
   (423 samples; external normative trajectory).
2. Find all *glass* Schaefer-400 timeseries CSVs under timeseries_anatomical.
3. Per run: mean signal across Limbic columns and across SomMot columns
   (SomMot includes A1/STG in the 7-network Schaefer labelling).
4. Resample each run's two traces to 423 points (linear interpolation on [0,1]).
5. Average resampled traces across all runs → group-mean limbic(t), auditory(t).
6. Pearson r between ratings and each group-mean BOLD trace; scatter plots.

This does not claim independent subjects for the ratings side — interpret as
alignment between normative affect dynamics and mean fMRI dynamics in your
cohort during Glass.

Output: outputs/musical_analysis/glass_ratings_bold_correlation.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import config  # noqa: E402

# Default paths
DEFAULT_XLSX = os.path.join(os.path.expanduser("~"), "Downloads", "Stable_meanRatings-Glass.xlsx")
OUT_DIR = os.path.join(config.OUTPUTS_DIR, "musical_analysis")


def load_ratings(xlsx_path: str) -> tuple[np.ndarray, np.ndarray, int]:
    def _sheet(sheet: str) -> pd.DataFrame:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
        df.columns = ["a", "b", "mean", "se"]
        df = df[pd.to_numeric(df["mean"], errors="coerce").notna()].copy()
        return df

    df_v = _sheet("GH_VMeanThr 0Alls")
    df_a = _sheet("GH_AMeanThr 0Alls")
    n = min(len(df_v), len(df_a))
    v = df_v["mean"].values[:n].astype(float)
    a = df_a["mean"].values[:n].astype(float)
    return v, a, n


def find_glass_csvs(ts_dir: str) -> list[str]:
    """All Schaefer-400 CSVs whose filename contains 'glass' (case-insensitive)."""
    if not os.path.isdir(ts_dir):
        return []
    files = sorted(
        os.path.join(ts_dir, f)
        for f in os.listdir(ts_dir)
        if "glass" in f.lower() and f.endswith("_schaefer400_ts.csv")
    )
    return [f for f in files if os.path.isfile(f)]


def network_mean_timeseries(df: pd.DataFrame, substr: str) -> np.ndarray:
    cols = [c for c in df.columns if substr in str(c)]
    if not cols:
        return np.array([])
    return df[cols].mean(axis=1).values.astype(float)


def resample_to_n(y: np.ndarray, n_out: int) -> np.ndarray:
    """Linearly interpolate y (length T) to length n_out."""
    T = len(y)
    if T < 2:
        return np.full(n_out, np.nan)
    x_old = np.linspace(0.0, 1.0, T)
    x_new = np.linspace(0.0, 1.0, n_out)
    return np.interp(x_new, x_old, y)


def collect_group_bold(ts_dir: str, n_target: int) -> tuple[np.ndarray, np.ndarray, int]:
    paths = find_glass_csvs(ts_dir)
    if not paths:
        raise FileNotFoundError(f"No *glass*schaefer400_ts.csv under {ts_dir}")

    limbic_stack = []
    aud_stack = []
    for p in paths:
        df = pd.read_csv(p)
        lim = network_mean_timeseries(df, "Limbic")
        aud = network_mean_timeseries(df, "SomMot")
        if lim.size == 0 or aud.size == 0:
            continue
        limbic_stack.append(resample_to_n(lim, n_target))
        aud_stack.append(resample_to_n(aud, n_target))

    if not limbic_stack:
        raise RuntimeError("No valid Glass runs with Limbic/SomMot columns")

    L = np.nanmean(np.stack(limbic_stack, axis=0), axis=0)
    A = np.nanmean(np.stack(aud_stack, axis=0), axis=0)
    return L, A, len(limbic_stack)


def _scatter_panel(ax, x: np.ndarray, y: np.ndarray, xlab: str, ylab: str, color: str) -> float:
    r, p = stats.pearsonr(x, y)
    ax.scatter(x, y, s=8, alpha=0.35, color=color, edgecolors="none", rasterized=True)
    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, m * xx + b, color="black", linewidth=1.2)
    ax.set_xlabel(xlab, fontsize=11)
    ax.set_ylabel(ylab, fontsize=11)
    ax.text(
        0.03, 0.97,
        f"r = {r:+.3f}\np = {p:.2e}" if p < 0.001 else f"r = {r:+.3f}\np = {p:.3f}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85, ec="grey"),
    )
    ax.spines[["top", "right"]].set_visible(False)
    return float(r)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", default=DEFAULT_XLSX, help="Path to Stable_meanRatings-Glass.xlsx")
    parser.add_argument(
        "--ts-dir",
        default=config.TS_OUTPUT_DIR_ANATOMICAL,
        help="Directory with Glass Schaefer-400 CSVs",
    )
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.isfile(args.xlsx):
        raise FileNotFoundError(f"Excel not found: {args.xlsx}")

    valence, arousal, n = load_ratings(args.xlsx)
    limbic_bold, aud_bold, n_runs = collect_group_bold(args.ts_dir, n)

    print(f"Ratings length: {n}")
    print(f"Glass fMRI runs averaged: {n_runs}")

    # ── figure: 2x2 scatter + bar of r ─────────────────────────────────────
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.45], hspace=0.35, wspace=0.28)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax_bar = fig.add_subplot(gs[2, :])

    r_v_lim = _scatter_panel(
        ax00, valence, limbic_bold,
        "Valence (normative mean)", "BOLD — Limbic mean (z, group avg)", "#8E44AD",
    )
    r_v_aud = _scatter_panel(
        ax01, valence, aud_bold,
        "Valence (normative mean)", "BOLD — SomMot mean (z, group avg)", "#2980B9",
    )
    r_a_lim = _scatter_panel(
        ax10, arousal, limbic_bold,
        "Arousal (normative mean)", "BOLD — Limbic mean (z, group avg)", "#8E44AD",
    )
    r_a_aud = _scatter_panel(
        ax11, arousal, aud_bold,
        "Arousal (normative mean)", "BOLD — SomMot mean (z, group avg)", "#2980B9",
    )

    ax00.set_title("A", loc="left", fontweight="bold", fontsize=12)
    ax01.set_title("B", loc="left", fontweight="bold", fontsize=12)
    ax10.set_title("C", loc="left", fontweight="bold", fontsize=12)
    ax11.set_title("D", loc="left", fontweight="bold", fontsize=12)

    labels = ["Valence × Limbic", "Valence × SomMot", "Arousal × Limbic", "Arousal × SomMot"]
    rs = [r_v_lim, r_v_aud, r_a_lim, r_a_aud]
    colors = ["#8E44AD", "#2980B9", "#8E44AD", "#2980B9"]
    xpos = np.arange(len(labels))
    ax_bar.bar(xpos, rs, color=colors, edgecolor="black", linewidth=0.6)
    ax_bar.axhline(0, color="grey", linewidth=0.8)
    ax_bar.set_xticks(xpos)
    ax_bar.set_xticklabels(labels, rotation=15, ha="right")
    ax_bar.set_ylabel("Pearson r")
    ax_bar.set_title("Summary: correlation of normative ratings with group-mean network BOLD (n timepoints = %d)" % n)
    ax_bar.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Glass excerpt: normative valence/arousal vs group-averaged fMRI (Limbic vs SomMot)\n"
        f"All Glass Schaefer-400 runs resampled to rating grid and averaged (n_runs = {n_runs})",
        fontsize=12, fontweight="bold", y=0.98,
    )

    out_path = os.path.join(OUT_DIR, "glass_ratings_bold_correlation.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
