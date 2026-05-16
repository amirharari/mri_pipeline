"""
Group-average BOLD during the Glass segment (after Salience) for three ROIs:

  Limbic            — mean of Schaefer-400 parcels with 'Limbic' in the label
  Auditory (proxy)  — mean of Schaefer-400 'SomMot' parcels (includes A1/STG)
  Ventral striatum  — mean of Tian S2 NAc shell + core (LH/RH)

Design assumption
-----------------
First ``salience_sec`` seconds of each run are Salience (not Glass); only
TRs after that enter the average.

Per Glass run we compute the temporal mean of each ROI aggregate (already
z-scored per run in the extractor).  We then average across runs and report
mean ± SEM (treating each run as one pseudo-observation; with few subjects
this is descriptive only).

Output: outputs/musical_analysis/glass_group_roi_bar.png (+ printed table)
"""
from __future__ import annotations

import argparse
from typing import Optional
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import config  # noqa: E402

OUT_DIR = os.path.join(config.OUTPUTS_DIR, "musical_analysis")

# Ventral striatum proxy — Tian S2 accumbens
VS_COLS = [
    "NAc-shell-lh", "NAc-shell-rh", "NAc-core-lh", "NAc-core-rh",
]


def find_glass_schaefer(ts_dir: str) -> list[str]:
    return sorted(
        os.path.join(ts_dir, f)
        for f in os.listdir(ts_dir)
        if "glass" in f.lower() and f.endswith("_schaefer400_ts.csv")
    )


def matching_tian(schaefer_path: str) -> Optional[str]:
    tian = schaefer_path.replace("_schaefer400_ts.csv", "_tian_s2_ts.csv")
    return tian if os.path.isfile(tian) else None


def network_mean(df: pd.DataFrame, substr: str) -> np.ndarray:
    cols = [c for c in df.columns if substr in str(c)]
    if not cols:
        return np.array([])
    return df[cols].mean(axis=1).values.astype(float)


def vs_mean(df: pd.DataFrame) -> np.ndarray:
    have = [c for c in VS_COLS if c in df.columns]
    if not have:
        return np.array([])
    return df[have].mean(axis=1).values.astype(float)


def glass_segment(y: np.ndarray, tr: float, salience_sec: float) -> np.ndarray:
    n_skip = int(np.ceil(salience_sec / tr))
    n_skip = min(n_skip, max(0, len(y) - 1))
    return y[n_skip:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ts-dir", default=config.TS_OUTPUT_DIR_ANATOMICAL)
    ap.add_argument("--tr", type=float, default=0.8, help="TR in seconds (set to your sequence)")
    ap.add_argument("--salience-sec", type=float, default=30.0, help="Skip first N seconds (Salience)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    sch_paths = find_glass_schaefer(args.ts_dir)
    if not sch_paths:
        raise SystemExit(f"No Glass Schaefer CSVs in {args.ts_dir}")

    per_run = []  # rows: limbic_mean, aud_mean, vs_mean (scalar per run)

    for sp in sch_paths:
        tian_p = matching_tian(sp)
        if tian_p is None:
            continue
        df_s = pd.read_csv(sp)
        df_t = pd.read_csv(tian_p)

        lim = network_mean(df_s, "Limbic")
        aud = network_mean(df_s, "SomMot")
        vsm = vs_mean(df_t)
        if lim.size == 0 or aud.size == 0 or vsm.size == 0:
            continue

        n = min(len(lim), len(aud), len(vsm))
        lim, aud, vsm = lim[:n], aud[:n], vsm[:n]

        lim_g = glass_segment(lim, args.tr, args.salience_sec)
        aud_g = glass_segment(aud, args.tr, args.salience_sec)
        vsm_g = glass_segment(vsm, args.tr, args.salience_sec)

        if len(lim_g) < 5:
            continue

        per_run.append({
            "run": os.path.basename(sp),
            "limbic":   float(np.nanmean(lim_g)),
            "auditory": float(np.nanmean(aud_g)),
            "vs":       float(np.nanmean(vsm_g)),
        })

    if not per_run:
        raise SystemExit("No valid Glass runs with paired Schaefer + Tian after skip.")

    dfp = pd.DataFrame(per_run)
    n_runs = len(dfp)

    labels = ["Limbic\n(Schaefer)", "Auditory\n(SomMot)", "Ventral striatum\n(NAc shell+core)"]
    means  = [dfp["limbic"].mean(), dfp["auditory"].mean(), dfp["vs"].mean()]
    sems   = [dfp["limbic"].sem(ddof=1), dfp["auditory"].sem(ddof=1), dfp["vs"].sem(ddof=1)]

    print(f"Glass runs (paired Schaefer+Tian): {n_runs}")
    print(f"Skipped first {args.salience_sec}s (~{int(np.ceil(args.salience_sec/args.tr))} TRs), TR={args.tr}s")
    print()
    print(dfp.to_string(index=False))
    print()
    print("Group mean ± SEM across runs (descriptive):")
    for lab, m, s in zip(labels, means, sems):
        print(f"  {lab.replace(chr(10), ' ')}: {m:+.4f} ± {s:.4f}")

    # ── bar figure ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(3)
    colors = ["#8E44AD", "#2980B9", "#27AE60"]
    ax.bar(x, means, yerr=sems, capsize=6, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean BOLD (z-scored units)\nGlass segment only")
    ax.set_title(
        f"Group average ROI signal during Glass\n"
        f"Mean ± SEM across {n_runs} runs | Salience excluded: first {args.salience_sec:.0f}s",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    outp = os.path.join(OUT_DIR, "glass_group_roi_bar.png")
    plt.savefig(outp, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {outp}")


if __name__ == "__main__":
    main()
