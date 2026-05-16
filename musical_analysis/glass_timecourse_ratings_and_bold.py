"""
Glass run: normative valence/arousal + session-mean BOLD (Limbic, SomMot, NAc).

**Default (``--drop-salience`` not set):** BOLD is the **full run from scan
start (t=0)** so the first ``salience_sec`` (e.g. 30 s) is a **Salience /
baseline** context, then Glass continues to end of scan.
  - Shaded band = Salience window; dashed vertical line = Glass onset.
  - **No BOLD resampling:** native TR spacing; shorter scans NaN-padded to the
    longest run; session curves = ``nanmean`` across runs.
  - Normative valence/arousal (423 samples) are plotted only over the **Glass**
    window: x from ``salience_sec`` to ``salience_sec + L_glass×TR`` (qualitative;
    external rating study).

**``--drop-salience``:** BOLD and x-axis start **after** Salience (Glass-only),
as in the previous version.

**``--resample-to-ratings``:** extra figure with BOLD **interpolated** to 423 bins
(Glass segment only).

Using baseline in analysis
--------------------------
Yes, it can make sense: (1) **context** — same scanner, same subject, pre-Glass
state; (2) **change** — Glass vs Salience (paired difference or percent change);
(3) **not** a substitute for a separate resting-state run if you need a
task-free baseline. For strict FC/activation inference, prefer GLM contrasts or
pre-Glass baseline regression; for exploration, overlaying full run is fine.

Outputs
-------
  glass_timecourse_ratings_bold.png
  glass_timecourse_ratings_bold_resampled.png   — if --resample-to-ratings
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import config  # noqa: E402

OUT_DIR = os.path.join(config.OUTPUTS_DIR, "musical_analysis")
DEFAULT_XLSX = os.path.join(os.path.expanduser("~"), "Downloads", "Stable_meanRatings-Glass.xlsx")

VS_COLS = [
    "NAc-shell-lh", "NAc-shell-rh", "NAc-core-lh", "NAc-core-rh",
]

SESSION_ORDER = ["ses-1", "ses-2", "ses-3"]


def load_ratings(xlsx_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    def _sheet(sheet: str) -> pd.DataFrame:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
        df.columns = ["a", "b", "mean", "se"]
        df = df[pd.to_numeric(df["mean"], errors="coerce").notna()].copy()
        return df

    df_v = _sheet("GH_VMeanThr 0Alls")
    df_a = _sheet("GH_AMeanThr 0Alls")
    n = min(len(df_v), len(df_a))
    return (
        df_v["mean"].values[:n].astype(float),
        df_v["se"].values[:n].astype(float),
        df_a["mean"].values[:n].astype(float),
        df_a["se"].values[:n].astype(float),
        n,
    )


def parse_session(fname: str) -> Optional[str]:
    m = re.search(r"(ses-\d+)", fname)
    return m.group(1) if m else None


def find_glass_schaefer(ts_dir: str) -> List[str]:
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


def resample_to_n(y: np.ndarray, n_out: int) -> np.ndarray:
    T = len(y)
    if T < 2:
        return np.full(n_out, np.nan)
    x_old = np.linspace(0.0, 1.0, T)
    x_new = np.linspace(0.0, 1.0, n_out)
    return np.interp(x_new, x_old, y)


def n_skip_trs(tr: float, salience_sec: float) -> int:
    return int(np.ceil(salience_sec / tr))


def iter_glass_runs_full(ts_dir: str, tr: float, salience_sec: float) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Full scan from first TR: (session, limbic, aud, vs). Requires enough TRs after Salience."""
    out = []
    n_skip = n_skip_trs(tr, salience_sec)
    for sp in find_glass_schaefer(ts_dir):
        ses = parse_session(os.path.basename(sp))
        if ses not in SESSION_ORDER:
            continue
        tp = matching_tian(sp)
        if tp is None:
            continue

        df_s = pd.read_csv(sp)
        df_t = pd.read_csv(tp)
        lim = network_mean(df_s, "Limbic")
        aud = network_mean(df_s, "SomMot")
        vsm = vs_mean(df_t)
        if lim.size == 0 or aud.size == 0 or vsm.size == 0:
            continue

        n = min(len(lim), len(aud), len(vsm))
        lim, aud, vsm = lim[:n], aud[:n], vsm[:n]
        if len(lim) < n_skip + 5:
            continue

        out.append((ses, lim, aud, vsm))
    return out


def to_glass_only(
    runs_full: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    tr: float,
    salience_sec: float,
) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    out = []
    for ses, lim, aud, vsm in runs_full:
        lim_g = glass_segment(lim, tr, salience_sec)
        aud_g = glass_segment(aud, tr, salience_sec)
        vsm_g = glass_segment(vsm, tr, salience_sec)
        if len(lim_g) < 5:
            continue
        out.append((ses, lim_g, aud_g, vsm_g))
    return out


def _pad_nan(y: np.ndarray, L_max: int) -> np.ndarray:
    """Pad 1d array to length L_max with NaN (tail after scan end)."""
    out = np.full(L_max, np.nan, dtype=float)
    n = min(len(y), L_max)
    out[:n] = y[:n]
    return out


def collect_by_session_glass_tr(
    runs: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]], L_max: int
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """Per session: list of length-L_max arrays (NaN pad after shorter scans end)."""
    by_ses: Dict[str, Dict[str, List[np.ndarray]]] = {
        s: {"limbic": [], "auditory": [], "vs": []} for s in SESSION_ORDER
    }
    for ses, lim_g, aud_g, vsm_g in runs:
        by_ses[ses]["limbic"].append(_pad_nan(lim_g, L_max))
        by_ses[ses]["auditory"].append(_pad_nan(aud_g, L_max))
        by_ses[ses]["vs"].append(_pad_nan(vsm_g, L_max))
    return by_ses


def collect_by_session_resample(
    runs: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]], n_target: int
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    by_ses: Dict[str, Dict[str, List[np.ndarray]]] = {
        s: {"limbic": [], "auditory": [], "vs": []} for s in SESSION_ORDER
    }
    for ses, lim_g, aud_g, vsm_g in runs:
        by_ses[ses]["limbic"].append(resample_to_n(lim_g, n_target))
        by_ses[ses]["auditory"].append(resample_to_n(aud_g, n_target))
        by_ses[ses]["vs"].append(resample_to_n(vsm_g, n_target))
    return by_ses


def plot_four_panels(
    x_top: np.ndarray,
    v_m: np.ndarray,
    v_se: np.ndarray,
    a_m: np.ndarray,
    a_se: np.ndarray,
    x_bold: np.ndarray,
    by_ses: Dict[str, Dict[str, List[np.ndarray]]],
    xlabel: str,
    suptitle_extra: str,
    out_path: str,
    glass_onset_sec: Optional[float] = None,
) -> None:
    roi_names = [
        ("limbic", "Limbic (Schaefer mean)"),
        ("auditory", "Auditory / SomMot (Schaefer mean)"),
        ("vs", "Ventral striatum (NAc shell + core, Tian)"),
    ]
    panel_letters = ["A", "B", "C", "D"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    plt.rcParams.update({"font.size": 10})

    def _mark_salience(ax) -> None:
        if glass_onset_sec is not None and glass_onset_sec > 0:
            ax.axvspan(0.0, glass_onset_sec, facecolor="0.5", alpha=0.12, zorder=0)
            ax.axvline(
                glass_onset_sec, color="0.25", linestyle="--", linewidth=1.1, zorder=2
            )

    ax = axes[0]
    _mark_salience(ax)
    ax.fill_between(x_top, v_m - v_se, v_m + v_se, alpha=0.2, color="#E94E77", linewidth=0)
    ax.fill_between(x_top, a_m - a_se, a_m + a_se, alpha=0.2, color="#4A90D9", linewidth=0)
    ax.plot(x_top, v_m, color="#C0392B", linewidth=1.8, label="Valence (normative)")
    ax.plot(x_top, a_m, color="#1F618D", linewidth=1.8, label="Arousal (normative)")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_ylabel("Valence / Arousal (a.u.)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.02, 1.05, "A", transform=ax.transAxes, fontsize=14, fontweight="bold")

    for ax, (key, title), pl in zip(axes[1:], roi_names, panel_letters[1:]):
        _mark_salience(ax)
        for ses in SESSION_ORDER:
            lst = by_ses[ses][key]
            if not lst:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                y = np.nanmean(np.stack(lst, axis=0), axis=0)
            label = f"{ses} ({config.SESSION_LABELS.get(ses, ses)})  n={len(lst)} runs"
            ax.plot(
                x_bold, y,
                color=config.SESSION_COLORS[ses],
                linewidth=1.7,
                label=label,
            )
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_ylabel("BOLD (z)")
        ax.set_title(title, fontsize=11, loc="left")
        ax.legend(loc="upper right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.02, 1.05, pl, transform=ax.transAxes, fontsize=14, fontweight="bold")

    ax0 = axes[0]
    h1, l1 = ax0.get_legend_handles_labels()

    leg_handles: List[object] = []
    leg_labels: List[str] = []
    if glass_onset_sec is not None and glass_onset_sec > 0:
        leg_handles.append(
            Patch(facecolor="0.5", alpha=0.25, edgecolor="none", label="Salience")
        )
        leg_labels.append("Salience (baseline)")
        leg_handles.append(
            Line2D([0], [0], color="0.25", linestyle="--", linewidth=1.1, label="Glass onset")
        )
        leg_labels.append("Glass onset")
    leg_handles.extend(h1)
    leg_labels.extend(l1)
    ax0.legend(leg_handles, leg_labels, loc="upper right", fontsize=8)

    axes[-1].set_xlabel(xlabel)

    fig.suptitle(
        "Glass run: normative affect + session-mean ROI BOLD\n" + suptitle_extra,
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xlsx", default=DEFAULT_XLSX)
    ap.add_argument("--ts-dir", default=config.TS_OUTPUT_DIR_ANATOMICAL)
    ap.add_argument("--tr", type=float, default=0.8)
    ap.add_argument("--salience-sec", type=float, default=30.0)
    ap.add_argument(
        "--drop-salience",
        action="store_true",
        help="BOLD x-axis starts at Glass onset (no Salience segment on plot)",
    )
    ap.add_argument(
        "--resample-to-ratings",
        action="store_true",
        help="Also write legacy resampled figure (423 bins)",
    )
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.isfile(args.xlsx):
        raise SystemExit(f"Excel not found: {args.xlsx}")

    v_m, v_se, a_m, a_se, n_rat = load_ratings(args.xlsx)
    runs_full = iter_glass_runs_full(args.ts_dir, args.tr, args.salience_sec)
    if not runs_full:
        raise SystemExit("No valid Glass runs (Schaefer + Tian).")

    n_skip = n_skip_trs(args.tr, args.salience_sec)

    if args.drop_salience:
        runs_plot = to_glass_only(runs_full, args.tr, args.salience_sec)
        if not runs_plot:
            raise SystemExit("No runs left after Salience skip.")
        L_max = max(len(lim) for _, lim, _, _ in runs_plot)
        T_wall = L_max * args.tr
        x_bold = np.arange(L_max, dtype=float) * args.tr
        x_ratings = np.linspace(0.0, T_wall, n_rat)
        glass_onset_sec = None
        xlabel = (
            f"Time after Salience offset (s)  |  BOLD: native TR, TR={args.tr}s"
        )
        suptitle_extra = (
            f"BOLD = Glass segment only (after {args.salience_sec:.0f}s Salience). "
            f"L_max = {L_max} TRs ({T_wall:.1f}s). Shorter runs NaN-padded; "
            "session mean = nanmean across runs. Ratings scaled to Glass duration for display."
        )
    else:
        runs_plot = runs_full
        L_max = max(len(lim) for _, lim, _, _ in runs_plot)
        L_glass_max = max(len(lim) - n_skip for _, lim, _, _ in runs_plot)
        T_glass = L_glass_max * args.tr
        x_bold = np.arange(L_max, dtype=float) * args.tr
        x_ratings = np.linspace(
            args.salience_sec, args.salience_sec + T_glass, n_rat
        )
        glass_onset_sec = args.salience_sec
        xlabel = (
            f"Time from scan start (s)  |  BOLD: full run, TR={args.tr}s"
        )
        suptitle_extra = (
            f"BOLD = full run from t=0; Salience ~{args.salience_sec:.0f}s (shaded), then Glass. "
            f"Longest run L_full = {L_max} TRs; Glass window up to {L_glass_max} TRs ({T_glass:.1f}s). "
            "Ratings (423 pts) span Glass window only. Shorter runs NaN-padded."
        )

    by_ses_tr = collect_by_session_glass_tr(runs_plot, L_max)

    plot_four_panels(
        x_top=x_ratings,
        v_m=v_m,
        v_se=v_se,
        a_m=a_m,
        a_se=a_se,
        x_bold=x_bold,
        by_ses=by_ses_tr,
        xlabel=xlabel,
        suptitle_extra=suptitle_extra,
        out_path=os.path.join(OUT_DIR, "glass_timecourse_ratings_bold.png"),
        glass_onset_sec=glass_onset_sec,
    )

    print(
        f"Main figure: L_max = {L_max} TRs ({L_max * args.tr:.1f} s on x-axis); "
        f"drop_salience={args.drop_salience}"
    )
    for ses in SESSION_ORDER:
        for k in ("limbic", "auditory", "vs"):
            print(f"  {ses} {k}: {len(by_ses_tr[ses][k])} runs")

    if args.resample_to_ratings:
        runs_glass = to_glass_only(runs_full, args.tr, args.salience_sec)
        if not runs_glass:
            raise SystemExit("No runs for resampled figure after Salience skip.")
        by_ses_rs = collect_by_session_resample(runs_glass, n_rat)
        t_idx = np.arange(n_rat, dtype=float)
        plot_four_panels(
            x_top=t_idx,
            v_m=v_m,
            v_se=v_se,
            a_m=a_m,
            a_se=a_se,
            x_bold=t_idx,
            by_ses=by_ses_rs,
            xlabel=f"Resampled bin index (n = {n_rat})",
            suptitle_extra=(
                f"LEGACY: BOLD Glass-only resampled to {n_rat} bins per run. "
                f"Salience skip {args.salience_sec:.0f}s, TR = {args.tr}s."
            ),
            out_path=os.path.join(OUT_DIR, "glass_timecourse_ratings_bold_resampled.png"),
            glass_onset_sec=None,
        )


if __name__ == "__main__":
    main()
