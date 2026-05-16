"""Leave-one-out (LOO) psychedelic signature similarity analysis
================================================================
For each subject × music piece, computes how similar that piece's
functional connectivity is to the psilocybin "brain state signature"
— *without* using that piece to define the signature (LOO design).

Pipeline
--------
1.  LOO psilocybin signature (Schaefer-400 FC, Fisher-z averaged):

        sig = mean_FC(ses-2, all runs EXCEPT target piece)
            − mean_FC(ses-1, all runs EXCEPT target piece)

    Both means include ALL run types (rest + other music).
    Fisher-z averaging is used to avoid bias from bounded correlations.

2.  Per-session similarity  [Pearson r of upper-triangle vectors]

        r_ses1  = r( piece FC ses-1,  sig )   baseline control  → should be ≈ 0
        r_ses2  = r( piece FC ses-2,  sig )   psilo validation  → should be highest
        r_ses3  = r( piece FC ses-3,  sig )   follow-up         → KEY question
        r_rest3 = r( rest  FC ses-3,  sig )   non-specific ctrl

3.  Summary metrics

        Δr           = r_ses3 − r_ses1   (piece became more psilo-like?)
        music_spec_r = r_ses3 − r_rest3  (music-specific vs general follow-up shift?)

Outputs  (saved to <ts_dir>/../music_psychedelic_similarity/)
--------------------------------------------------------------
    <subject>_loo_similarity.csv   per-piece metrics
    <subject>_loo_similarity.png   grouped bar chart
    group_loo_summary.csv          all subjects × pieces
    group_loo_summary.png          Δr heatmap + bar chart

Usage
-----
    cd connectivity
    python music_psychedelic_similarity.py
    python music_psychedelic_similarity.py --data PATH/TO/TS_DIR
    python music_psychedelic_similarity.py --data PATH --subjects sub-001 sub-002
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import config  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SCHAEFER_SUFFIX = "_schaefer400_ts.csv"

SESSION_COLORS = {
    "ses-1":  "#4A90D9",   # blue   — baseline
    "ses-2":  "#E94E77",   # pink   — psilocybin
    "ses-3":  "#5CB85C",   # green  — follow-up
    "rest-3": "#A8D5A2",   # light-green — follow-up rest (control)
}

# Non-canonical → canonical task name (shared with subject_overview.py)
TASK_ALIASES: dict[str, str] = {
    "glass":             "music_acq-glass",
    "music_acq-baliero": "music_acq-bailero",
}

TASK_LABELS: dict[str, str] = {
    "rest":               "Rest",
    "restchecktr2":       "Rest (checkTR)",
    "music_acq-glass":    "Glass",
    "music_acq-bailero":  "Bailero",
    "music_acq-tundra":   "Tundra",
    "music_acq-personal": "Personal",
    "music_acq-personal1":"Personal 1",
    "music_acq-personal2":"Personal 2",
    "music_acq-agami":    "Agami",
    "music_acq-salome":   "Salomé",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_task(task: str) -> str:
    return TASK_ALIASES.get(task, task)


def _subject_from(fname: str) -> str | None:
    m = re.match(r"(sub-\d+)_", fname)
    return m.group(1) if m else None


def _session_from(fname: str) -> str | None:
    m = re.search(r"(ses-\d+)", fname)
    return m.group(1) if m else None


def _task_from(fname: str) -> str | None:
    m = re.search(r"task-(.+?)_run-", fname)
    return _normalize_task(m.group(1)) if m else None


def _fisher_z(r: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(r, -0.9999, 0.9999))


def _inv_fisher_z(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def _mean_fc_fisher(matrices: list) -> np.ndarray | None:
    """Fisher-z average of a list of FC (correlation) matrices."""
    if not matrices:
        return None
    zs = [_fisher_z(m) for m in matrices]
    return _inv_fisher_z(np.mean(zs, axis=0))


def _upper_tri(m: np.ndarray) -> np.ndarray:
    """Return flattened upper triangle (no diagonal)."""
    idx = np.triu_indices(m.shape[0], k=1)
    return m[idx]


def _fc_sim(fc1: np.ndarray, fc2: np.ndarray) -> float:
    """Pearson r between upper triangles of two FC matrices.
    Returns NaN if either has no variance or < 10 finite values."""
    v1 = _upper_tri(fc1)
    v2 = _upper_tri(fc2)
    mask = np.isfinite(v1) & np.isfinite(v2)
    if mask.sum() < 10:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = np.corrcoef(v1[mask], v2[mask])[0, 1]
    return float(r)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_fc(csv_path: str) -> np.ndarray | None:
    """Load a timeseries CSV → FC matrix.  Returns None on failure."""
    try:
        df = pd.read_csv(csv_path)
        ts = df.values.T   # shape: (parcels, timepoints)
        fc = np.corrcoef(ts)
        np.fill_diagonal(fc, np.nan)
        return fc
    except Exception as exc:
        print(f"    [WARNING] could not load {os.path.basename(csv_path)}: {exc}")
        return None


def load_all_fc(ts_dir: str, subjects: list | None = None) -> dict:
    """Scan ts_dir and return nested dict {subject: {session: {task: FC_matrix}}}.

    Only Schaefer-400 CSVs are used (SCHAEFER_SUFFIX).
    """
    data: dict = {}
    for fname in sorted(os.listdir(ts_dir)):
        if not fname.endswith(SCHAEFER_SUFFIX):
            continue
        sub = _subject_from(fname)
        ses = _session_from(fname)
        tsk = _task_from(fname)
        if sub is None or ses is None or tsk is None:
            continue
        if subjects and sub not in subjects:
            continue

        fc = _load_fc(os.path.join(ts_dir, fname))
        if fc is None:
            continue

        data.setdefault(sub, {}).setdefault(ses, {})[tsk] = fc

    return data


# ─────────────────────────────────────────────────────────────────────────────
# LOO analysis — per subject
# ─────────────────────────────────────────────────────────────────────────────

def _loo_signature(ses2_fcs: dict, ses1_fcs: dict, exclude: str) -> np.ndarray | None:
    """Compute the LOO psilocybin signature (difference FC matrix).

    Parameters
    ----------
    ses2_fcs : {task: FC_matrix}  all ses-2 FCs
    ses1_fcs : {task: FC_matrix}  all ses-1 FCs
    exclude  : task key to leave out

    Returns
    -------
    sig = mean_FC(ses2 \ exclude) − mean_FC(ses1 \ exclude)
    or None if too few runs.
    """
    ses2_list = [fc for t, fc in ses2_fcs.items() if t != exclude]
    ses1_list = [fc for t, fc in ses1_fcs.items() if t != exclude]

    if not ses2_list or not ses1_list:
        return None

    mean2 = _mean_fc_fisher(ses2_list)
    mean1 = _mean_fc_fisher(ses1_list)
    return mean2 - mean1


def analyze_subject(subject: str, subj_data: dict) -> pd.DataFrame:
    """Run LOO analysis for one subject.

    Returns a DataFrame with one row per music piece that exists in ses-3.
    """
    ses1_fcs = subj_data.get("ses-1", {})
    ses2_fcs = subj_data.get("ses-2", {})
    ses3_fcs = subj_data.get("ses-3", {})

    # Identify music pieces present in ses-3 (follow-up) — these are the target pieces
    music_tasks_ses3 = [t for t in ses3_fcs if t.startswith("music_acq-")]

    rest_ses3_fc = ses3_fcs.get("rest")  # follow-up rest (non-specific control)

    rows = []
    for piece in sorted(music_tasks_ses3):
        piece_ses1 = ses1_fcs.get(piece)
        piece_ses2 = ses2_fcs.get(piece)   # optional — used only for validation
        piece_ses3 = ses3_fcs.get(piece)

        if piece_ses3 is None:
            continue

        # LOO signature: exclude the target piece from both session averages
        sig = _loo_signature(ses2_fcs, ses1_fcs, exclude=piece)
        if sig is None:
            print(f"  [{subject}] {piece}: insufficient runs for LOO — skipping")
            continue

        n_ses2_loo = sum(1 for t in ses2_fcs if t != piece)
        n_ses1_loo = sum(1 for t in ses1_fcs if t != piece)

        r_ses1  = _fc_sim(piece_ses1, sig) if piece_ses1 is not None else float("nan")
        r_ses2  = _fc_sim(piece_ses2, sig) if piece_ses2 is not None else float("nan")
        r_ses3  = _fc_sim(piece_ses3, sig)
        r_rest3 = _fc_sim(rest_ses3_fc, sig) if rest_ses3_fc is not None else float("nan")

        # Also compute direct FC similarity (raw, not vs signature):
        # piece_ses3 vs piece_ses1  — how much did connectivity change?
        r_ses3_vs_ses1_raw = _fc_sim(piece_ses3, piece_ses1) if piece_ses1 is not None else float("nan")
        # piece_ses3 vs rest_ses3   — is music distinct from rest in follow-up?
        r_ses3_vs_rest3_raw = _fc_sim(piece_ses3, rest_ses3_fc) if rest_ses3_fc is not None else float("nan")

        delta_r      = r_ses3 - r_ses1 if np.isfinite(r_ses1) and np.isfinite(r_ses3) else float("nan")
        music_spec_r = r_ses3 - r_rest3 if np.isfinite(r_rest3) and np.isfinite(r_ses3) else float("nan")

        rows.append({
            "subject":           subject,
            "piece":             piece,
            "piece_label":       TASK_LABELS.get(piece, piece),
            "n_ses2_loo":        n_ses2_loo,
            "n_ses1_loo":        n_ses1_loo,
            # --- similarity to LOO psilocybin signature ---
            "r_ses1_vs_sig":     r_ses1,    # baseline control — expect ≈ 0
            "r_ses2_vs_sig":     r_ses2,    # psilo validation — expect highest
            "r_ses3_vs_sig":     r_ses3,    # KEY: follow-up similarity to psilo sig
            "r_rest3_vs_sig":    r_rest3,   # non-specific follow-up control
            # --- derived metrics ---
            "delta_r":           delta_r,       # r_ses3 − r_ses1
            "music_specific_r":  music_spec_r,  # r_ses3 − r_rest3
            # --- raw FC similarity (supplementary) ---
            "r_ses3_vs_ses1_raw":   r_ses3_vs_ses1_raw,
            "r_ses3_vs_rest3_raw":  r_ses3_vs_rest3_raw,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — per subject
# ─────────────────────────────────────────────────────────────────────────────

def plot_subject(subject: str, df: pd.DataFrame, out_dir: str) -> None:
    """Grouped bar chart: 4 similarity bars per piece, with Δr annotated."""
    if df.empty:
        return

    n_pieces = len(df)
    fig_w = max(8, n_pieces * 2.0 + 2)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 5.5),
                             gridspec_kw={"width_ratios": [3, 1]})

    ax = axes[0]
    x = np.arange(n_pieces)
    width = 0.19
    offsets = [-1.5, -0.5, 0.5, 1.5]

    bar_specs = [
        ("r_ses1_vs_sig",  "Baseline (ses-1)",          SESSION_COLORS["ses-1"],  "//"),
        ("r_ses2_vs_sig",  "Psilo (ses-2, validation)",  SESSION_COLORS["ses-2"],  ""),
        ("r_ses3_vs_sig",  "Follow-up (ses-3)",          SESSION_COLORS["ses-3"],  ""),
        ("r_rest3_vs_sig", "Rest follow-up (control)",   SESSION_COLORS["rest-3"], "\\\\"),
    ]

    for (col, label, color, hatch), offset in zip(bar_specs, offsets):
        vals = df[col].values
        bars = ax.bar(x + offset * width, vals, width,
                      label=label, color=color, hatch=hatch,
                      edgecolor="white", linewidth=0.6, alpha=0.88)
        # Small value label on top of each bar
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=6.5, color="#333333")

    # Δr annotation: bracket from ses-1 to ses-3 bar
    for i, row in df.iterrows():
        xi = list(df.index).index(i)
        delta = row["delta_r"]
        if np.isfinite(delta):
            col = "#1A7A3C" if delta > 0 else "#AA2222"
            ax.text(xi, max(df[["r_ses1_vs_sig","r_ses2_vs_sig","r_ses3_vs_sig","r_rest3_vs_sig"]].iloc[list(df.index).index(i)].fillna(0)) + 0.055,
                    f"Δr={delta:+.2f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color=col)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["piece_label"].tolist(), fontsize=9)
    ax.set_ylabel("Pearson r  (similarity to LOO psilo signature)", fontsize=9)
    ax.set_title(f"{subject}  —  LOO Psilocybin Signature Similarity\n"
                 f"Δr = r(follow-up) − r(baseline)  |  positive = more psilo-like at follow-up",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(min(-0.15, df[["r_ses1_vs_sig","r_ses2_vs_sig","r_ses3_vs_sig","r_rest3_vs_sig"]].min().min() - 0.1),
                max(0.35, df[["r_ses1_vs_sig","r_ses2_vs_sig","r_ses3_vs_sig","r_rest3_vs_sig"]].max().max() + 0.12))

    # Right panel: music-specific Δr bar chart (r_ses3 - r_rest3)
    ax2 = axes[1]
    colors = ["#1A7A3C" if v > 0 else "#AA2222" for v in df["music_specific_r"].fillna(0)]
    ax2.barh(x[::-1], df["music_specific_r"].values[::-1],
             color=colors[::-1], edgecolor="white", alpha=0.85)
    ax2.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax2.set_yticks(x[::-1])
    ax2.set_yticklabels(df["piece_label"].tolist()[::-1], fontsize=9)
    ax2.set_xlabel("r_ses3 − r_rest3\n(music-specific psilo imprint)", fontsize=8)
    ax2.set_title("Music-specific\nEffect", fontsize=9, fontweight="bold")

    fig.tight_layout(pad=2.0)
    out_path = os.path.join(out_dir, f"{subject}_loo_similarity.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(out_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — group summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_group(all_df: pd.DataFrame, out_dir: str) -> None:
    """Two-panel group figure: Δr heatmap + mean Δr bar chart per piece."""
    if all_df.empty:
        return

    pieces   = sorted(all_df["piece"].unique())
    subjects = sorted(all_df["subject"].unique())

    # Build Δr matrix (subjects × pieces)
    delta_mat = np.full((len(subjects), len(pieces)), np.nan)
    for ri, sub in enumerate(subjects):
        for ci, piece in enumerate(pieces):
            row = all_df[(all_df["subject"] == sub) & (all_df["piece"] == piece)]
            if not row.empty:
                delta_mat[ri, ci] = row["delta_r"].values[0]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(pieces) * 1.8 + 4), max(5, len(subjects) * 0.7 + 3)),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Heatmap
    ax = axes[0]
    vabs = np.nanmax(np.abs(delta_mat)) if not np.all(np.isnan(delta_mat)) else 0.3
    vabs = max(vabs, 0.1)
    im = ax.imshow(delta_mat, cmap="RdYlGn", vmin=-vabs, vmax=vabs,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(pieces)))
    ax.set_xticklabels([TASK_LABELS.get(p, p) for p in pieces], rotation=35,
                       ha="right", fontsize=9)
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=9)
    ax.set_title("Δr per subject × piece\n(r_ses3 − r_ses1 vs LOO psilo signature)",
                 fontsize=10, fontweight="bold")
    # Annotate cells
    for ri in range(len(subjects)):
        for ci in range(len(pieces)):
            v = delta_mat[ri, ci]
            if np.isfinite(v):
                ax.text(ci, ri, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7.5, color="black" if abs(v) < vabs * 0.7 else "white")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cb.set_label("Δr", fontsize=9)

    # Mean Δr per piece
    ax2 = axes[1]
    mean_delta = [np.nanmean(delta_mat[:, ci]) for ci in range(len(pieces))]
    sem_delta  = [np.nanstd(delta_mat[:, ci]) / max(1, np.sum(np.isfinite(delta_mat[:, ci])) ** 0.5)
                  for ci in range(len(pieces))]
    colors = ["#1A7A3C" if v > 0 else "#AA2222" for v in mean_delta]
    y = np.arange(len(pieces))
    ax2.barh(y[::-1], [v for v in mean_delta[::-1]],
             xerr=[v for v in sem_delta[::-1]],
             color=colors[::-1], edgecolor="white", alpha=0.85, capsize=4)
    ax2.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax2.set_yticks(y[::-1])
    ax2.set_yticklabels([TASK_LABELS.get(p, p) for p in pieces[::-1]], fontsize=9)
    ax2.set_xlabel("Mean Δr ± SEM\n(across subjects)", fontsize=8)
    ax2.set_title("Group\nAverage", fontsize=9, fontweight="bold")

    fig.suptitle("LOO Psilocybin Signature Similarity — Group Summary\n"
                 "Positive Δr = music more psilo-like at follow-up than baseline",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout(pad=2.0)
    out_path = os.path.join(out_dir, "group_loo_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(out_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(ts_dir: str, subjects: list | None = None, out_dir: str | None = None) -> None:
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(ts_dir), "music_psychedelic_similarity")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nLOO Psilocybin Signature Similarity")
    print(f"  Input  : {ts_dir}")
    print(f"  Output : {out_dir}\n")

    print("Loading FC matrices …")
    all_data = load_all_fc(ts_dir, subjects=subjects)
    if not all_data:
        print("[ERROR] No Schaefer-400 timeseries found.")
        return
    print(f"  Loaded data for {len(all_data)} subjects\n")

    all_rows = []
    for subject in sorted(all_data):
        subj_data = all_data[subject]
        print(f"[{subject}]")

        ses_keys  = list(subj_data.keys())
        n_ses2    = len(subj_data.get("ses-2", {}))
        n_ses1    = len(subj_data.get("ses-1", {}))
        n_ses3    = len(subj_data.get("ses-3", {}))
        music_ses3 = [t for t in subj_data.get("ses-3", {}) if t.startswith("music_acq-")]
        print(f"  Sessions: {ses_keys}  |  ses-1:{n_ses1} runs  ses-2:{n_ses2}  ses-3:{n_ses3}")
        print(f"  Music in ses-3: {music_ses3}")

        if not music_ses3:
            print("  (no music tasks in ses-3 — skipping)\n")
            continue
        if n_ses2 < 2:
            print("  (ses-2 has < 2 runs — LOO would leave empty set — skipping)\n")
            continue

        df = analyze_subject(subject, subj_data)
        if df.empty:
            print("  (no valid pieces for LOO analysis)\n")
            continue

        # Print per-piece summary
        print(f"  {'Piece':<20} {'r_ses1':>7} {'r_ses2':>7} {'r_ses3':>7} {'r_rest3':>8} {'dR':>7} {'MusicSpec':>10}")
        for _, row in df.iterrows():
            def _fmt(v):
                return f"{v:+.3f}" if np.isfinite(v) else "  n/a "
            flag = "  *** IMPRINT ***" if row["delta_r"] > 0.05 else ""
            print(f"  {row['piece_label']:<20} {_fmt(row['r_ses1_vs_sig'])} "
                  f"{_fmt(row['r_ses2_vs_sig'])} {_fmt(row['r_ses3_vs_sig'])} "
                  f"{_fmt(row['r_rest3_vs_sig'])} {_fmt(row['delta_r'])} "
                  f"{_fmt(row['music_specific_r'])}{flag}")

        # Save CSV
        csv_path = os.path.join(out_dir, f"{subject}_loo_similarity.csv")
        df.to_csv(csv_path, index=False)

        # Save figure
        plot_subject(subject, df, out_dir)
        all_rows.append(df)
        print()

    if not all_rows:
        print("No valid analyses produced.")
        return

    group_df = pd.concat(all_rows, ignore_index=True)
    group_csv = os.path.join(out_dir, "group_loo_summary.csv")
    group_df.to_csv(group_csv, index=False)
    print(f"Group CSV: {group_csv}")

    plot_group(group_df, out_dir)
    print(f"\nDone.  Results in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data",
        default=config.TS_OUTPUT_DIR_ANATOMICAL,
        help="Timeseries directory (default: config.TS_OUTPUT_DIR_ANATOMICAL)",
    )
    parser.add_argument(
        "--subjects", nargs="*", default=None,
        help="Restrict to subjects e.g. sub-001 sub-002",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Override output directory (default: <data>/../music_psychedelic_similarity)",
    )
    args = parser.parse_args()
    main(args.data, args.subjects, out_dir=args.out_dir)
