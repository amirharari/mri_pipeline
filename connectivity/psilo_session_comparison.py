"""psilo_session_comparison.py

Publication-ready connectivity analysis for the psilocybin study.

Study design (assumed standard psilocybin paradigm):
  ses-1 = baseline  |  ses-2 = psilocybin  |  ses-3 = follow-up

Figures produced
----------------
  fig1_rest_connectivity_matrices.png        — Schaefer-400 + Tian S2 rest matrices × 3 sessions
  fig2a_music_connectivity_schaefer400.png   — music conditions × 3 sessions (Schaefer-400)
  fig2b_music_connectivity_tian_s2.png       — music conditions × 3 sessions (Tian S2 subcortical)
  fig3_global_integration.png             — GCOR + mean connectivity bar charts
  fig4_network_heatmaps.png               — Schaefer-400 7-network integration × 3 sessions
  fig5_ventral_striatum.png               — Accumbens seed connectivity × session/condition
  fig6_connectivity_differences.png       — psilocybin – baseline difference maps

Usage
-----
  cd connectivity
  python psilo_session_comparison.py                    # default: sub-002
  python psilo_session_comparison.py --subject sub-004
"""

from __future__ import annotations

import json
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import within_network_connectivity, fisher_z_mean

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (defaults = sub-002; overridden by _apply_config / --subject)
# ─────────────────────────────────────────────────────────────────────────────
_REPO   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS_DIR  = os.path.join(_REPO, "outputs", "psilo_sub002_GSR_off_v2")
OUT_DIR = os.path.join(_REPO, "outputs", "psilo_sub002_analysis")

SUBJECT_LABEL  = "sub-002"
SESSION_LABELS = {"ses-1": "Baseline", "ses-2": "Psilocybin", "ses-3": "Follow-up"}
SESSION_COLORS = {"ses-1": "#2196F3", "ses-2": "#F44336", "ses-3": "#4CAF50"}

CONDITIONS: Dict[str, Dict[str, Optional[str]]] = {
    "rest": {
        "ses-1": "sub-002_ses-1_task-rest_run-1",
        "ses-2": "sub-002_ses-2_task-rest_run-1",
        "ses-3": "sub-002_ses-3_task-rest_run-1",
    },
    "agami": {
        "ses-1": "sub-002_ses-1_task-music_acq-agami_run-3",
        "ses-2": "sub-002_ses-2_task-music_acq-agami_run-3",
        "ses-3": "sub-002_ses-3_task-music_acq-agami_run-3",
    },
    "bailero": {
        "ses-1": "sub-002_ses-1_task-music_acq-baliero_run-2",
        "ses-2": "sub-002_ses-2_task-music_acq-bailero_run-2",
        "ses-3": "sub-002_ses-3_task-music_acq-bailero_run-2",
    },
    "glass": {
        "ses-1": "sub-002_ses-1_task-music_acq-glass_run-1",
        "ses-2": "sub-002_ses-2_task-music_acq-glass_run-1",
        "ses-3": "sub-002_ses-3_task-music_acq-glass_run-1",
    },
}

REST_CONDITIONS   = ["rest"]
MUSIC_CONDITIONS  = ["agami", "bailero", "glass"]


def _apply_config(cfg: dict) -> None:
    """Update all module-level config variables from a subject config dict.

    SUBJECT_LABEL is derived by combining cfg['subject_label'] with the GSR
    status read from extraction_config.json in the timeseries folder — so
    figure titles always reflect the actual extraction parameters rather than
    any hardcoded string.
    """
    global TS_DIR, OUT_DIR, SUBJECT_LABEL
    global SESSION_LABELS, SESSION_COLORS
    global CONDITIONS, REST_CONDITIONS, MUSIC_CONDITIONS
    TS_DIR           = cfg["ts_dir"]
    OUT_DIR          = cfg["out_dir"]
    SESSION_LABELS   = cfg["session_labels"]
    SESSION_COLORS   = cfg["session_colors"]
    CONDITIONS       = cfg["conditions"]
    REST_CONDITIONS  = cfg["rest_conditions"]
    MUSIC_CONDITIONS = cfg["music_conditions"]

    # Derive label from provenance file written by fmri_timeseries_extractor
    base_label  = cfg["subject_label"]
    prov_path   = os.path.join(TS_DIR, "extraction_config.json")
    if os.path.isfile(prov_path):
        with open(prov_path) as fh:
            ext_cfg = json.load(fh)
        gsr_on = ext_cfg.get("gsr", False)
        SUBJECT_LABEL = base_label + (" (GSR on)" if gsr_on else " (GSR off)")
    else:
        print(f"  [WARN] extraction_config.json not found in {TS_DIR} — "
              f"GSR status unknown, using base label.")
        SUBJECT_LABEL = base_label

YEONET_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
YEONET_LABELS = {
    "Vis": "Visual", "SomMot": "Somatomotor", "DorsAttn": "Dorsal Attn",
    "SalVentAttn": "Salience/VA", "Limbic": "Limbic",
    "Cont": "Control", "Default": "Default Mode",
}
YEONET_COLORS = {
    "Vis": "#781286", "SomMot": "#4682B4", "DorsAttn": "#00760E",
    "SalVentAttn": "#C43AFA", "Limbic": "#DCF8A4", "Cont": "#E69422",
    "Default": "#CD3E4E",
}

PLOT_STYLE = {
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
}

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def ts_path(prefix: str, atlas: str) -> str:
    return os.path.join(TS_DIR, f"{prefix}_{atlas}_ts.csv")


def load_ts(prefix: Optional[str], atlas: str) -> Optional[pd.DataFrame]:
    """Load a timeseries CSV. Returns None if prefix is None or file not found."""
    if prefix is None:
        return None
    p = ts_path(prefix, atlas)
    if not os.path.isfile(p):
        return None
    return pd.read_csv(p)


def conn_matrix(df: pd.DataFrame) -> np.ndarray:
    """Pearson correlation matrix, diagonal zeroed, NaN/±inf -> 0.

    np.corrcoef can return values marginally outside [-1, 1] due to floating
    point; clipping prevents downstream np.arctanh from producing inf.
    """
    c = np.corrcoef(df.values.T)
    c = np.clip(c, -1.0, 1.0)
    np.fill_diagonal(c, 0)
    np.nan_to_num(c, copy=False, nan=0.0)
    return c


def fisher_z_matrix(c: np.ndarray) -> np.ndarray:
    """Fisher-z transform a correlation matrix (clip to avoid inf)."""
    return np.arctanh(np.clip(c, -0.999, 0.999))


def mean_connectivity(c: np.ndarray) -> float:
    """Mean off-diagonal correlation (Fisher-z averaged then back-transformed)."""
    n = c.shape[0]
    idx = np.triu_indices(n, k=1)
    return float(np.tanh(np.nanmean(fisher_z_matrix(c)[idx])))


def schaefer_network(col: str) -> str:
    """Extract Yeo network name from Schaefer column like '7Networks_LH_Vis_1'."""
    parts = col.split("_")
    return parts[2] if len(parts) >= 3 else "Unknown"


# ── AAL lobe ordering ─────────────────────────────────────────────────────────
# Maps each AAL region name prefix to a sort-key so the matrix shows anatomical
# blocks (Frontal → Parietal → Temporal → Occipital → Limbic → Sub-cortical).
_AAL_LOBE_ORDER = {
    "Precentral": 0, "Frontal_Sup": 1, "Frontal_Mid": 2, "Frontal_Inf": 3,
    "Rolandic": 4, "Supp_Motor": 5, "Olfactory": 6, "Frontal_Sup_Medial": 7,
    "Frontal_Med_Orb": 8, "Rectus": 9,
    "Postcentral": 10, "Parietal_Sup": 11, "Parietal_Inf": 12,
    "SupraMarginal": 13, "Angular": 14, "Precuneus": 15,
    "Paracentral_Lobule": 16,
    "Heschl": 17, "Temporal_Sup": 18, "Temporal_Pole_Sup": 19,
    "Temporal_Mid": 20, "Temporal_Pole_Mid": 21, "Temporal_Inf": 22,
    "Calcarine": 23, "Cuneus": 24, "Lingual": 25,
    "Occipital_Sup": 26, "Occipital_Mid": 27, "Occipital_Inf": 28,
    "Fusiform": 29,
    "Insula": 30, "Cingulum_Ant": 31, "Cingulum_Mid": 32, "Cingulum_Post": 33,
    "Hippocampus": 34, "ParaHippocampal": 35, "Amygdala": 36,
    "Caudate": 37, "Putamen": 38, "Pallidum": 39, "Thalamus": 40,
    "Cerebelum": 41, "Vermis": 42,
}

def _aal_sort_key(col: str) -> int:
    """Return lobe sort key for an AAL column name."""
    for prefix, key in _AAL_LOBE_ORDER.items():
        if col.startswith(prefix):
            return key
    return 99


def aal_sorted_order(columns: List[str]) -> List[int]:
    """Return index permutation that reorders AAL columns by lobe."""
    indexed = sorted(enumerate(columns), key=lambda x: _aal_sort_key(x[1]))
    return [i for i, _ in indexed]


def schaefer_sorted_order(columns: List[str]) -> List[int]:
    """Return index permutation that groups Schaefer columns by Yeo network."""
    net_order = {n: i for i, n in enumerate(YEONET_ORDER)}
    indexed   = sorted(enumerate(columns),
                       key=lambda x: net_order.get(schaefer_network(x[1]), 99))
    return [i for i, _ in indexed]


def _add_network_borders(ax: plt.Axes, columns: List[str]) -> None:
    """Draw white separator lines between Yeo network blocks on a Schaefer matrix."""
    nets    = [schaefer_network(c) for c in columns]
    borders = [i for i in range(1, len(nets)) if nets[i] != nets[i - 1]]
    for b in borders:
        ax.axhline(b - 0.5, color="white", linewidth=1.0, alpha=0.9)
        ax.axvline(b - 0.5, color="white", linewidth=1.0, alpha=0.9)


def _add_lobe_borders(ax: plt.Axes, ordered_cols: List[str]) -> None:
    """Draw white separator lines between anatomical lobe groups on an AAL matrix."""
    keys    = [_aal_sort_key(c) for c in ordered_cols]
    borders = [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]]
    for b in borders:
        ax.axhline(b - 0.5, color="white", linewidth=0.8, alpha=0.8)
        ax.axvline(b - 0.5, color="white", linewidth=0.8, alpha=0.8)


def build_network_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Compute 7×7 mean Fisher-z connectivity between Yeo networks (Schaefer)."""
    nets = [schaefer_network(c) for c in df.columns]
    mat  = np.zeros((len(YEONET_ORDER), len(YEONET_ORDER)))
    for i, ni in enumerate(YEONET_ORDER):
        for j, nj in enumerate(YEONET_ORDER):
            ci = [df.columns[k] for k, n in enumerate(nets) if n == ni]
            cj = [df.columns[k] for k, n in enumerate(nets) if n == nj]
            if not ci or not cj:
                continue
            sub = df[ci].values
            sub2 = df[cj].values
            r = np.corrcoef(sub.T, sub2.T)
            block = r[:len(ci), len(ci):]
            z_vals = np.arctanh(np.clip(block, -0.999, 0.999))
            if i == j:
                # Within-network: upper triangle only
                idx = np.triu_indices(block.shape[0], k=1) if block.shape[0] > 1 else ([], [])
                vals = z_vals[idx]
            else:
                vals = z_vals.ravel()
            mat[i, j] = float(np.tanh(np.nanmean(vals))) if len(vals) > 0 else 0.0
    return mat, [YEONET_LABELS[n] for n in YEONET_ORDER]


def integration_index(net_mat: np.ndarray) -> float:
    """Between-network / within-network mean connectivity ratio."""
    n = len(YEONET_ORDER)
    within  = np.mean([net_mat[i, i] for i in range(n)])
    off_idx = [(i, j) for i in range(n) for j in range(n) if i != j]
    between = np.mean([net_mat[i, j] for i, j in off_idx])
    return float(between / within) if within != 0 else 0.0


def gcor(df: pd.DataFrame) -> float:
    c = conn_matrix(df)
    n = c.shape[0]
    return float(np.nanmean(c[~np.eye(n, dtype=bool)]))


def normalize_matrix(c: np.ndarray) -> np.ndarray:
    """Z-score normalize a connectivity matrix (mean=0, std=1 across all elements).

    Diagonal stays zero.  Useful for comparing *pattern* across scans that differ
    in overall GCOR level — removes global amplitude differences while preserving
    relative network structure.
    """
    mask = ~np.eye(c.shape[0], dtype=bool)
    vals = c[mask]
    mu, sd = vals.mean(), vals.std()
    if sd < 1e-8:
        return np.zeros_like(c)
    z = np.zeros_like(c)
    z[mask] = (vals - mu) / sd
    return z


def _draw_conn_panel(ax: plt.Axes, c: np.ndarray, title: str, ylabel: str,
                     vmin: float, vmax: float, cbar_label: str,
                     gcor_val: Optional[float] = None,
                     ses_color: str = "black") -> None:
    """Shared helper: render one connectivity matrix panel with colorbar and labels.

    `title` is only rendered when non-empty, preventing accidental overwriting of
    titles that were set on the axes before this function was called.
    """
    im = ax.imshow(c, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    full_title = title
    if gcor_val is not None:
        full_title = (full_title + "\n" if full_title else "") + f"GCOR = {gcor_val:.3f}"
    if full_title:                             # never call set_title with empty string
        ax.set_title(full_title, fontsize=9, fontweight="bold", color=ses_color)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    else:
        ax.set_yticks([])
    ax.set_xticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label,
                 format="%.1f")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Rest connectivity matrices — AAL + Schaefer × 3 sessions
#           Each atlas: raw row (r) + z-normalised row (z-score)
# ─────────────────────────────────────────────────────────────────────────────

def fig1_rest_matrices(condition: str = "rest") -> None:
    """Plot AAL + Schaefer connectivity matrices for a rest-type condition × 3 sessions.

    Called once per entry in REST_CONDITIONS so that each rest scan gets its own figure.
    """
    cond_label = condition.replace("restchecktr2", "rest-checkTR").replace("rest", "Rest")
    print(f"Fig 1 ({condition}): rest connectivity matrices ...")
    if condition not in CONDITIONS:
        print(f"  [SKIP] condition '{condition}' not in CONDITIONS")
        return

    atlases   = [("schaefer400", "Schaefer-400 (7 Net.)"), ("tian_s2", "Tian S2 (32 subcortical)")]
    sessions  = list(SESSION_LABELS.keys())
    nrows, ncols = 4, 3

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 14),
                                 gridspec_kw={"hspace": 0.55, "wspace": 0.15})
        fig.suptitle(f"Resting-state Connectivity ({cond_label})  |  {SUBJECT_LABEL}  (GSR off)\n"
                     "Odd rows = raw Pearson r  |  Even rows = z-score normalised",
                     fontsize=12, fontweight="bold")

        stripe_labels = {0: "AAL\n(raw r)", 1: "AAL\n(z-norm)",
                         2: "Schaefer\n(raw r)", 3: "Schaefer\n(z-norm)"}

        for ai, (atlas, atlas_label) in enumerate(atlases):
            raw_row  = ai * 2
            norm_row = ai * 2 + 1

            _sample_df = next(
                (load_ts(CONDITIONS[condition][s], atlas) for s in sessions
                 if CONDITIONS[condition].get(s) and
                 load_ts(CONDITIONS[condition][s], atlas) is not None), None)
            if _sample_df is None:
                continue
            if atlas == "aal":
                perm = aal_sorted_order(list(_sample_df.columns))
            else:
                perm = schaefer_sorted_order(list(_sample_df.columns))
            ordered_cols = [list(_sample_df.columns)[i] for i in perm]

            all_raw = [conn_matrix(df)[np.ix_(perm, perm)]
                       for ses in sessions
                       if CONDITIONS[condition].get(ses)
                       for df in [load_ts(CONDITIONS[condition].get(ses), atlas)]
                       if df is not None]
            if not all_raw:
                continue
            vmax_raw  = max(0.2,
                            float(np.percentile(np.abs(np.concatenate([c.ravel() for c in all_raw])), 97)))
            vmax_norm = 2.0

            for col, ses in enumerate(sessions):
                prefix = CONDITIONS[condition].get(ses)
                df = load_ts(prefix, atlas) if prefix else None

                ax = axes[raw_row, col]
                if df is None:
                    ax.set_visible(False)
                else:
                    c = conn_matrix(df)[np.ix_(perm, perm)]
                    gcor_val = float(np.nanmean(c[~np.eye(c.shape[0], dtype=bool)]))
                    ylabel = stripe_labels[raw_row] if col == 0 else ""
                    _draw_conn_panel(ax, c, SESSION_LABELS[ses], ylabel,
                                     -vmax_raw, vmax_raw, "r",
                                     gcor_val=gcor_val,
                                     ses_color=SESSION_COLORS[ses])
                    if atlas == "schaefer400":
                        _add_network_borders(ax, ordered_cols)
                    else:
                        _add_lobe_borders(ax, ordered_cols)

                ax2 = axes[norm_row, col]
                if df is None:
                    ax2.set_visible(False)
                else:
                    c_full = conn_matrix(df)[np.ix_(perm, perm)]
                    zc     = normalize_matrix(c_full)
                    ylabel2 = stripe_labels[norm_row] if col == 0 else ""
                    _draw_conn_panel(ax2, zc, "", ylabel2,
                                     -vmax_norm, vmax_norm, "z",
                                     ses_color=SESSION_COLORS[ses])
                    if atlas == "schaefer400":
                        _add_network_borders(ax2, ordered_cols)
                    else:
                        _add_lobe_borders(ax2, ordered_cols)

        for row in (1, 2):
            for col in range(ncols):
                axes[row, col].spines["top"].set_linewidth(1.5)
                axes[row, col].spines["top"].set_color("#aaaaaa")

        safe_cond = condition.replace("-", "_")
        out = os.path.join(OUT_DIR, f"fig1_{safe_cond}_connectivity_matrices.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Music piece × session (Schaefer)
# ─────────────────────────────────────────────────────────────────────────────

def _music_matrix_fig(atlas: str, atlas_title: str, fname: str) -> None:
    """Shared engine for fig2a (AAL) and fig2b (Schaefer) music connectivity figures.

    3 rows (agami / bailero / glass) × 3 columns (ses-1 / ses-2 / ses-3).
    Raw Pearson r only — no z-norm rows.
    """
    sessions    = list(SESSION_LABELS.keys())
    cond_labels = {c: c.capitalize() for c in MUSIC_CONDITIONS}

    # ── column ordering + shared vmax ────────────────────────────────────────
    _sdf = next(
        (load_ts(CONDITIONS[c][s], atlas)
         for c in MUSIC_CONDITIONS for s in sessions
         if load_ts(CONDITIONS[c][s], atlas) is not None), None)
    if _sdf is None:
        print(f"  [WARN] no data for atlas {atlas}, skipping {fname}")
        return

    if atlas == "aal":
        perm = aal_sorted_order(list(_sdf.columns))
    else:
        perm = schaefer_sorted_order(list(_sdf.columns))
    ord_cols = [list(_sdf.columns)[i] for i in perm]

    all_raw = [conn_matrix(df)[np.ix_(perm, perm)].ravel()
               for cond in MUSIC_CONDITIONS
               for ses in sessions
               for df in [load_ts(CONDITIONS[cond].get(ses), atlas)]
               if df is not None]
    vmax = max(0.3, float(np.percentile(np.abs(np.concatenate(all_raw)), 97))) \
           if all_raw else 0.5

    nrows, ncols = len(MUSIC_CONDITIONS), len(sessions)   # 3 × 3

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 10),
                                 gridspec_kw={"hspace": 0.45, "wspace": 0.12})
        fig.suptitle(f"Music-listening Connectivity ({atlas_title})  |  {SUBJECT_LABEL}  (GSR off)",
                     fontsize=12, fontweight="bold")

        # Column headers (session labels)
        for col, ses in enumerate(sessions):
            axes[0, col].set_title(SESSION_LABELS[ses], fontsize=10,
                                   fontweight="bold", color=SESSION_COLORS[ses], pad=6)

        for row, cond in enumerate(MUSIC_CONDITIONS):
            for col, ses in enumerate(sessions):
                ax = axes[row, col]
                df = load_ts(CONDITIONS[cond].get(ses), atlas)
                if df is None:
                    ax.set_visible(False)
                    continue

                c = conn_matrix(df)[np.ix_(perm, perm)]
                g = float(np.nanmean(c[~np.eye(c.shape[0], dtype=bool)]))
                ylabel = cond_labels[cond] if col == 0 else ""
                _draw_conn_panel(ax, c, "", ylabel, -vmax, vmax, "r",
                                 ses_color=SESSION_COLORS[ses])
                if atlas == "schaefer400":
                    _add_network_borders(ax, ord_cols)
                else:
                    _add_lobe_borders(ax, ord_cols)

                # GCOR badge
                fc = "firebrick" if g > 0.4 else "steelblue"
                ax.text(0.02, 0.97, f"GCOR={g:.2f}", transform=ax.transAxes,
                        fontsize=6.5, va="top", ha="left", color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc=fc, alpha=0.85, ec="none"))

        out = os.path.join(OUT_DIR, fname)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  -> {out}")


def fig2_music_matrices() -> None:
    print("Fig 2a: music connectivity matrices (AAL) ...")
    _music_matrix_fig("schaefer400",  "Schaefer-400 7-Net.", "fig2a_music_connectivity_schaefer400.png")
    print("Fig 2b: music connectivity matrices (Schaefer) ...")
    _music_matrix_fig("tian_s2",     "Tian S2 subcortical", "fig2b_music_connectivity_tian_s2.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Global integration metrics
# ─────────────────────────────────────────────────────────────────────────────

def fig3_global_metrics() -> None:
    print("Fig 3: global integration metrics ...")
    sessions  = list(SESSION_LABELS.keys())
    conds_all = REST_CONDITIONS + MUSIC_CONDITIONS
    cond_labels = {c: c.replace("restchecktr2", "Rest-checkTR").capitalize() for c in conds_all}

    records = []
    for cond in conds_all:
        for ses in sessions:
            for atlas in ["aal", "schaefer400"]:
                df = load_ts(CONDITIONS[cond].get(ses), atlas)
                if df is None:
                    continue
                c = conn_matrix(df)
                g = gcor(df)
                mc = mean_connectivity(c)
                records.append({
                    "condition": cond_labels[cond],
                    "session": SESSION_LABELS[ses],
                    "ses_key": ses,
                    "atlas": atlas,
                    "gcor": g,
                    "mean_conn": mc,
                })
    df_m = pd.DataFrame(records)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"Global Integration Metrics  |  {SUBJECT_LABEL}  (GSR off)",
                     fontsize=13, fontweight="bold")

        x = np.arange(len(conds_all))
        w = 0.25

        for panel, (metric, ylabel, title) in enumerate([
            ("gcor",      "GCOR (mean off-diagonal r)", "Global Correlation (GCOR)"),
            ("mean_conn", "Mean connectivity (Fisher-z avg r)", "Mean Pairwise Connectivity"),
        ]):
            ax = axes[panel]
            for si, ses in enumerate(sessions):
                sub = df_m[(df_m["ses_key"] == ses) & (df_m["atlas"] == "schaefer400")]
                vals = [sub[sub["condition"] == cond_labels[c]][metric].values
                        for c in conds_all]
                vals = [v[0] if len(v) > 0 else np.nan for v in vals]
                bars = ax.bar(x + (si - 1) * w, vals, w,
                              label=SESSION_LABELS[ses],
                              color=SESSION_COLORS[ses], alpha=0.85,
                              edgecolor="white", linewidth=0.5)
                for bar, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.005, f"{v:.2f}",
                                ha="center", va="bottom", fontsize=6.5)

            if panel == 0:   # GCOR warning line only relevant on GCOR panel
                ax.axhline(0.4, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
                           label="GCOR warning (0.4)")
            ax.set_xticks(x)
            ax.set_xticklabels([cond_labels[c] for c in conds_all], fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight="bold")
            ax.legend(fontsize=7, frameon=False)
            ax.set_ylim(0, max(0.7, ax.get_ylim()[1] * 1.15))

        fig.text(0.5, -0.02,
                       f"Schaefer-400 atlas  |   Single subject ({SUBJECT_LABEL})  |  "
                 "ses-2 agami note: elevated GCOR due to high motion",
                 ha="center", fontsize=7, color="grey", style="italic")
        plt.tight_layout()
        out = os.path.join(OUT_DIR, "fig3_global_integration.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Schaefer 7-network integration heatmaps
# ─────────────────────────────────────────────────────────────────────────────

def fig4_network_heatmaps() -> None:
    print("Fig 4: network integration heatmaps ...")
    sessions = list(SESSION_LABELS.keys())
    short_labels = [YEONET_LABELS[n] for n in YEONET_ORDER]

    # Pre-compute all network matrices to derive a shared, data-driven color scale
    all_net_vals: List[float] = []
    net_mats: Dict[str, Dict] = {}   # net_mats[condition][ses] = matrix
    for cond_key in REST_CONDITIONS + MUSIC_CONDITIONS:
        net_mats[cond_key] = {}
        for ses in sessions:
            df = load_ts(CONDITIONS[cond_key].get(ses), "schaefer400")
            if df is not None:
                m, _ = build_network_matrix(df)
                net_mats[cond_key][ses] = m
                all_net_vals.extend(m.ravel().tolist())
    vmax_net = max(0.3, float(np.percentile(np.abs(all_net_vals), 97))) if all_net_vals else 0.5

    with plt.rc_context(PLOT_STYLE):
        # Two rows: rest (top), music average (bottom)
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle(f"Schaefer-400 7-Network Integration Matrix  |  {SUBJECT_LABEL}",
                     fontsize=13, fontweight="bold")

        for col, ses in enumerate(sessions):
            # Row 0: rest
            ax = axes[0, col]
            if ses in net_mats["rest"]:
                mat  = net_mats["rest"][ses]
                integ = integration_index(mat)
                sns.heatmap(mat, ax=ax, cmap="RdBu_r", center=0,
                            vmin=-vmax_net, vmax=vmax_net,
                            xticklabels=short_labels, yticklabels=short_labels,
                            annot=True, fmt=".2f", annot_kws={"size": 6.5},
                            cbar_kws={"label": "r", "shrink": 0.8}, linewidths=0.3)
                ax.set_title(f"Rest — {SESSION_LABELS[ses]}\n(Integ. index = {integ:.2f})",
                             fontsize=9, fontweight="bold", color=SESSION_COLORS[ses])
                ax.tick_params(axis="x", rotation=40)
                ax.tick_params(axis="y", rotation=0)
            else:
                ax.set_visible(False)

            # Row 1: average across music conditions (Fisher-z before averaging)
            ax2 = axes[1, col]
            z_mats = [fisher_z_matrix(net_mats[c][ses])
                      for c in MUSIC_CONDITIONS if ses in net_mats[c]]
            if z_mats:
                avg_mat = np.tanh(np.mean(z_mats, axis=0))
                integ = integration_index(avg_mat)
                sns.heatmap(avg_mat, ax=ax2, cmap="RdBu_r", center=0,
                            vmin=-vmax_net, vmax=vmax_net,
                            xticklabels=short_labels, yticklabels=short_labels,
                            annot=True, fmt=".2f", annot_kws={"size": 6.5},
                            cbar_kws={"label": "r", "shrink": 0.8}, linewidths=0.3)
                ax2.set_title(f"Music avg — {SESSION_LABELS[ses]}\n(Integ. index = {integ:.2f})",
                              fontsize=9, fontweight="bold", color=SESSION_COLORS[ses])
                ax2.tick_params(axis="x", rotation=40)
                ax2.tick_params(axis="y", rotation=0)
            else:
                ax2.set_visible(False)

        # Row labels
        axes[0, 0].set_ylabel("Rest", fontsize=10, fontweight="bold", labelpad=20)
        axes[1, 0].set_ylabel("Music (avg)", fontsize=10, fontweight="bold", labelpad=20)

        plt.tight_layout()
        out = os.path.join(OUT_DIR, "fig4_network_heatmaps.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Ventral striatum seed connectivity
# ─────────────────────────────────────────────────────────────────────────────

def fig5_ventral_striatum() -> None:
    print("Fig 5: ventral striatum analysis ...")
    sessions   = list(SESSION_LABELS.keys())
    vs_cols    = ["Left Accumbens", "Right Accumbens"]
    conds_all  = REST_CONDITIONS + MUSIC_CONDITIONS
    cond_labels = {c: c.replace("restchecktr2", "Rest-checkTR").capitalize() for c in conds_all}

    with plt.rc_context(PLOT_STYLE):
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f"Ventral Striatum (Nucleus Accumbens) Analysis  |  {SUBJECT_LABEL}",
                     fontsize=13, fontweight="bold")
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        # ── Panel A: VS–VS bilateral connectivity across conditions × sessions
        ax_a = fig.add_subplot(gs[0, 0])
        records = []
        for cond in conds_all:
            for ses in sessions:
                df_sc = load_ts(CONDITIONS[cond].get(ses), "tian_s2")
                if df_sc is None:
                    continue
                avail = [c for c in vs_cols if c in df_sc.columns]
                if len(avail) < 2:
                    continue
                r = float(df_sc[avail[0]].corr(df_sc[avail[1]]))
                records.append({
                    "condition": cond_labels[cond], "session": SESSION_LABELS[ses],
                    "ses_key": ses, "LR_corr": r,
                })
        df_vs = pd.DataFrame(records)
        x     = np.arange(len(conds_all))
        w     = 0.25
        for si, ses in enumerate(sessions):
            sub  = df_vs[df_vs["ses_key"] == ses]
            vals = [sub[sub["condition"] == cond_labels[c]]["LR_corr"].values for c in conds_all]
            vals = [v[0] if len(v) > 0 else np.nan for v in vals]
            ax_a.bar(x + (si - 1) * w, vals, w,
                     label=SESSION_LABELS[ses], color=SESSION_COLORS[ses],
                     alpha=0.85, edgecolor="white", linewidth=0.5)
        ax_a.set_xticks(x)
        ax_a.set_xticklabels([cond_labels[c] for c in conds_all], fontsize=8)
        ax_a.set_ylabel("L–R Accumbens correlation (r)")
        ax_a.set_title("A  Bilateral Accumbens Connectivity\n(Left–Right correlation)",
                       fontweight="bold")
        ax_a.legend(fontsize=7, frameon=False)

        # ── Panel B: VS seed connectivity with all HO cortical regions (rest only)
        ax_b = fig.add_subplot(gs[0, 1])
        # Build {region: {ses: r}} — used for both sorting and plotting
        all_ses_r: Dict[str, Dict[str, float]] = {}
        for ses in sessions:
            _rest_key = REST_CONDITIONS[0]
            df_sc  = load_ts(CONDITIONS[_rest_key].get(ses), "tian_s2")
            df_cor = load_ts(CONDITIONS[_rest_key].get(ses), "ho_cortical")
            if df_sc is None or df_cor is None:
                continue
            avail_vs = [c for c in vs_cols if c in df_sc.columns]
            if not avail_vs:
                continue
            vs_ts = df_sc[avail_vs].mean(axis=1).values
            for reg in df_cor.columns:
                all_ses_r.setdefault(reg, {})[ses] = float(pd.Series(vs_ts).corr(df_cor[reg]))

            # Sort by baseline connectivity
            sorted_regs = sorted(all_ses_r.keys(),
                                 key=lambda r: all_ses_r[r].get("ses-1", 0), reverse=True)
            top_n = 15
            top_regs = sorted_regs[:top_n]
            ys = np.arange(len(top_regs))
            for si, ses in enumerate(sessions):
                vals = [all_ses_r[r].get(ses, np.nan) for r in top_regs]
                # No label on scatter — legend built manually below to avoid duplicates
                ax_b.scatter(vals, ys + (si - 1) * 0.2,
                             color=SESSION_COLORS[ses], s=25,
                             zorder=3, alpha=0.9)
            ax_b.axvline(0, color="grey", linewidth=0.6, linestyle="--")
            ax_b.set_yticks(ys)
            ax_b.set_yticklabels([r.replace("Left ", "L. ").replace("Right ", "R. ")
                                   for r in top_regs], fontsize=7)
            ax_b.set_xlabel("Seed correlation (r)")
            ax_b.set_title("B  Accumbens Seed Connectivity\n(Top 15 cortical targets, Rest)",
                           fontweight="bold")
            # Manual legend — one handle per session, no duplication
            handles_b = [plt.Line2D([0], [0], marker="o", color="w",
                                    markerfacecolor=SESSION_COLORS[s], markersize=6)
                         for s in sessions]
            ax_b.legend(handles_b, [SESSION_LABELS[s] for s in sessions],
                        fontsize=7, frameon=False, loc="lower right")

        # ── Panel C: VS mean connectivity × condition × session (HO cortical)
        ax_c = fig.add_subplot(gs[1, 0])
        records_c = []
        for cond in conds_all:
            for ses in sessions:
                df_sc  = load_ts(CONDITIONS[cond].get(ses), "tian_s2")
                df_cor = load_ts(CONDITIONS[cond].get(ses), "ho_cortical")
                if df_sc is None or df_cor is None:
                    continue
                avail_vs = [c for c in vs_cols if c in df_sc.columns]
                if not avail_vs:
                    continue
                vs_ts  = df_sc[avail_vs].mean(axis=1).values
                rs     = np.array([float(pd.Series(vs_ts).corr(df_cor[reg]))
                                   for reg in df_cor.columns])
                # Fisher-z average — correct for bias when r is far from 0
                mean_r = float(np.tanh(np.nanmean(np.arctanh(np.clip(rs, -0.999, 0.999)))))
                records_c.append({
                    "condition": cond_labels[cond], "ses_key": ses,
                    "session": SESSION_LABELS[ses], "mean_cortical_r": mean_r,
                })
        df_vc = pd.DataFrame(records_c)
        for si, ses in enumerate(sessions):
            sub  = df_vc[df_vc["ses_key"] == ses]
            vals = [sub[sub["condition"] == cond_labels[c]]["mean_cortical_r"].values for c in conds_all]
            vals = [v[0] if len(v) > 0 else np.nan for v in vals]
            ax_c.bar(x + (si - 1) * w, vals, w,
                     label=SESSION_LABELS[ses], color=SESSION_COLORS[ses],
                     alpha=0.85, edgecolor="white", linewidth=0.5)
        ax_c.set_xticks(x)
        ax_c.set_xticklabels([cond_labels[c] for c in conds_all], fontsize=8)
        ax_c.set_ylabel("Mean Accumbens–Cortex r")
        ax_c.set_title("C  Mean Accumbens–Cortex Connectivity\nby Condition & Session",
                       fontweight="bold")
        ax_c.legend(fontsize=7, frameon=False)

        # ── Panel D: VS temporal dynamics (rest, 30-TR smoothing, x in minutes)
        ax_d = fig.add_subplot(gs[1, 1])
        for ses in sessions:
            _rest_key = REST_CONDITIONS[0]
            df_sc = load_ts(CONDITIONS[_rest_key].get(ses), "tian_s2")
            if df_sc is None:
                continue
            avail_vs = [c for c in vs_cols if c in df_sc.columns]
            if not avail_vs:
                continue
            ts = df_sc[avail_vs].mean(axis=1).values
            # 30-TR (~30 s) smoothing — reveals slow fluctuations, reduces noise
            smoothed = pd.Series(ts).rolling(30, center=True, min_periods=1).mean().values
            t_min = np.arange(len(smoothed)) / 60.0  # convert to minutes (TR=1 s)
            ax_d.plot(t_min, smoothed, color=SESSION_COLORS[ses], linewidth=1.2,
                      label=SESSION_LABELS[ses], alpha=0.9)
        ax_d.set_xlabel("Time (minutes)")
        ax_d.set_ylabel("Mean z-score (Accumbens L+R)")
        ax_d.set_title("D  Accumbens Temporal Dynamics\n(Rest, 30-s smoothing)",
                       fontweight="bold")
        ax_d.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax_d.legend(fontsize=7, frameon=False)

        out = os.path.join(OUT_DIR, "fig5_ventral_striatum.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Connectivity difference maps (psilocybin – baseline)
# ─────────────────────────────────────────────────────────────────────────────

def fig6_differences() -> None:
    print("Fig 6: connectivity difference maps ...")
    comparisons = [
        ("ses-2", "ses-1", "Psilocybin – Baseline"),
        ("ses-3", "ses-1", "Follow-up – Baseline"),
    ]
    _rc = REST_CONDITIONS[0]
    _rc_label = _rc.replace("restchecktr2", "Rest-checkTR").capitalize()
    _music_show = MUSIC_CONDITIONS[:3]  # show up to 3 music conditions
    conditions_show = [(_rc, _rc_label, "aal")] + [
        (c, c.capitalize(), "schaefer400") for c in _music_show
    ]
    # 4 columns: [Psilo–Base raw] [Psilo–Base z-norm] [FU–Base raw] [FU–Base z-norm]
    nrows = len(conditions_show)
    ncols = 4

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2),
                                 gridspec_kw={"wspace": 0.15, "hspace": 0.5})
        fig.suptitle(f"Connectivity Difference Maps  |  {SUBJECT_LABEL}  (GSR off)\n"
                     "Columns 1 & 3 = raw Δr  |  Columns 2 & 4 = z-normalised Δ (pattern only)",
                     fontsize=12, fontweight="bold")

        # Column headers
        col_titles = [
            "Psilo – Baseline\n(raw Δr)",
            "Psilo – Baseline\n(z-norm Δ)",
            "Follow-up – Baseline\n(raw Δr)",
            "Follow-up – Baseline\n(z-norm Δ)",
        ]
        for col, ct in enumerate(col_titles):
            axes[0, col].set_title(ct, fontsize=9, fontweight="bold", pad=5)

        # Shared vmaxes
        all_raw_diffs, all_norm_diffs = [], []
        for cond, _, atlas in conditions_show:
            for ses_b, ses_a, _ in comparisons:
                df_b = load_ts(CONDITIONS[cond].get(ses_b), atlas)
                df_a = load_ts(CONDITIONS[cond].get(ses_a), atlas)
                if df_b is not None and df_a is not None:
                    all_raw_diffs.append((conn_matrix(df_b) - conn_matrix(df_a)).ravel())
                    all_norm_diffs.append((normalize_matrix(conn_matrix(df_b))
                                           - normalize_matrix(conn_matrix(df_a))).ravel())
        vmax_raw  = max(0.15, float(np.percentile(np.abs(np.concatenate(all_raw_diffs)),  97))) \
                    if all_raw_diffs  else 0.3
        vmax_norm = max(0.5,  float(np.percentile(np.abs(np.concatenate(all_norm_diffs)), 97))) \
                    if all_norm_diffs else 1.0

        # Pre-compute permutation for each atlas (sort by lobe/network)
        _all_sessions = list(SESSION_LABELS.keys())
        _perm_cache: Dict[str, List[int]] = {}
        _ord_cache:  Dict[str, List[str]] = {}
        for _, _, atlas in conditions_show:
            if atlas not in _perm_cache:
                _sdf = next((load_ts(CONDITIONS[c].get(s), atlas)
                             for c in list(CONDITIONS) for s in _all_sessions
                             if load_ts(CONDITIONS[c].get(s), atlas) is not None), None)
                if _sdf is not None:
                    p = aal_sorted_order(list(_sdf.columns)) if atlas == "aal" \
                        else schaefer_sorted_order(list(_sdf.columns))
                    _perm_cache[atlas] = p
                    _ord_cache[atlas]  = [list(_sdf.columns)[i] for i in p]

        for row, (cond, cond_label, atlas) in enumerate(conditions_show):
            atlas_str = "AAL" if atlas == "aal" else "Sch-100"
            row_label = f"{cond_label}\n({atlas_str})"
            perm      = _perm_cache.get(atlas)
            ord_cols  = _ord_cache.get(atlas, [])

            for ci, (ses_b, ses_a, _) in enumerate(comparisons):
                df_b = load_ts(CONDITIONS[cond].get(ses_b), atlas)
                df_a = load_ts(CONDITIONS[cond].get(ses_a), atlas)
                col_raw  = ci * 2
                col_norm = ci * 2 + 1

                ax_r = axes[row, col_raw]
                ax_n = axes[row, col_norm]

                if df_b is None or df_a is None or perm is None:
                    ax_r.set_visible(False)
                    ax_n.set_visible(False)
                    continue

                c_b = conn_matrix(df_b)[np.ix_(perm, perm)]
                c_a = conn_matrix(df_a)[np.ix_(perm, perm)]
                diff_raw  = c_b - c_a
                diff_norm = normalize_matrix(c_b) - normalize_matrix(c_a)

                n_rois   = diff_raw.shape[0]
                off_diag = ~np.eye(n_rois, dtype=bool)
                mean_raw  = float(np.nanmean(diff_raw[off_diag]))
                mean_norm = float(np.nanmean(diff_norm[off_diag]))

                ylabel_r = row_label if col_raw == 0 else ""

                _draw_conn_panel(ax_r, diff_raw,  "", ylabel_r,
                                 -vmax_raw,  vmax_raw,  "Δr")
                if atlas == "schaefer400":
                    _add_network_borders(ax_r, ord_cols)
                    _add_network_borders(ax_n, ord_cols)
                else:
                    _add_lobe_borders(ax_r, ord_cols)
                    _add_lobe_borders(ax_n, ord_cols)
                ax_r.text(0.97, 0.03, f"Δ̄={mean_raw:+.3f}",
                          transform=ax_r.transAxes, fontsize=6.5,
                          ha="right", va="bottom",
                          color="darkred" if mean_raw > 0 else "navy")

                _draw_conn_panel(ax_n, diff_norm, "", "",
                                 -vmax_norm, vmax_norm, "Δz")
                ax_n.text(0.97, 0.03, f"Δ̄={mean_norm:+.3f}",
                          transform=ax_n.transAxes, fontsize=6.5,
                          ha="right", va="bottom",
                          color="darkred" if mean_norm > 0 else "navy")

        fig.text(0.5, -0.01,
                 "Red = increased connectivity  |  Blue = decreased  |"
                 "  z-norm panels remove global GCOR differences — compare pattern only",
                 ha="center", fontsize=7, color="grey", style="italic")
        out = os.path.join(OUT_DIR, "fig6_connectivity_differences.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Within-network connectivity bar chart (Schaefer 7 networks)
# ─────────────────────────────────────────────────────────────────────────────

def _znorm_within_network_vals(run_records: List[dict]) -> List[dict]:
    """Z-score within-network mean-r values *across all 7 networks* for each run.

    Each (session, condition) is one run.  For that run we have up to 7 network
    mean-r values.  We z-score the vector of 7 values so that the global GCOR
    shift is removed and only the *relative* network profile remains.

    Returns a copy of run_records with an extra key ``mean_r_z``.
    """
    from itertools import groupby

    # Group by (ses_key, condition) — each group is one run's 7-network vector
    records_out = []
    key_fn = lambda d: (d["ses_key"], d["condition"])
    for run_key, grp in groupby(sorted(run_records, key=key_fn), key=key_fn):
        grp_list = list(grp)
        vals = np.array([d["mean_r"] for d in grp_list], dtype=float)
        mu, sd = np.nanmean(vals), np.nanstd(vals)
        for d in grp_list:
            z = (d["mean_r"] - mu) / sd if sd > 1e-8 else 0.0
            records_out.append({**d, "mean_r_z": float(z)})
    return records_out


def fig7_within_network() -> None:
    print("Fig 7: within-network connectivity ...")
    sessions   = list(SESSION_LABELS.keys())
    _all_conds = REST_CONDITIONS + MUSIC_CONDITIONS
    conds_show = {c: c.replace("restchecktr2", "Rest-checkTR").capitalize() for c in _all_conds}
    cond_list  = list(conds_show.values())
    n_nets     = len(YEONET_ORDER)

    # ── 1. Collect raw within-network mean-r for every network × run ──────────
    all_records: List[dict] = []
    for ni, net in enumerate(YEONET_ORDER):
        for cond, cond_label in conds_show.items():
            for ses in sessions:
                df = load_ts(CONDITIONS[cond].get(ses), "schaefer400")
                if df is None:
                    continue
                cols = [c for c in df.columns if schaefer_network(c) == net]
                if len(cols) < 2:
                    continue
                res = within_network_connectivity(df, cols)
                if res:
                    all_records.append({
                        "net": net, "net_idx": ni,
                        "condition": cond_label, "ses_key": ses,
                        "session": SESSION_LABELS[ses],
                        "mean_r": res["mean"],
                    })

    # ── 2. Add z-normalised column (z-score across 7 networks per run) ────────
    all_records = _znorm_within_network_vals(all_records)
    df_all = pd.DataFrame(all_records)

    x = np.arange(len(cond_list))
    w = 0.25

    def _draw_row(axes_row: np.ndarray, value_col: str,
                  ylabel: str, row_title_suffix: str,
                  show_delta: bool = False) -> None:
        """Draw one row of within-network bars.

        show_delta: if True, annotate each group with a Δ arrow showing the
        psilocybin–baseline difference so the drug effect pops out visually.
        """
        for ni, net in enumerate(YEONET_ORDER):
            ax        = axes_row[ni]
            net_color = YEONET_COLORS.get(net, "grey")
            df_n      = df_all[df_all["net"] == net]

            if df_n.empty:
                ax.set_visible(False)
                continue

            all_vals_by_ses: Dict[str, List[float]] = {}
            for si, ses in enumerate(sessions):
                sub  = df_n[df_n["ses_key"] == ses]
                vals = [sub[sub["condition"] == c][value_col].values for c in cond_list]
                vals = [float(v[0]) if len(v) > 0 else np.nan for v in vals]
                all_vals_by_ses[ses] = vals
                ax.bar(x + (si - 1) * w, vals, w,
                       color=SESSION_COLORS[ses], alpha=0.85,
                       label=SESSION_LABELS[ses] if ni == 0 else "",
                       edgecolor="white", linewidth=0.5)

            # Psilocybin–Baseline delta annotation
            if show_delta:
                base = all_vals_by_ses.get("ses-1", [])
                psil = all_vals_by_ses.get("ses-2", [])
                for xi, (b, p) in enumerate(zip(base, psil)):
                    if np.isnan(b) or np.isnan(p):
                        continue
                    delta = p - b
                    ypos  = max(p, b) + 0.08
                    arrow = "▲" if delta > 0 else "▼"
                    col   = "darkred" if delta > 0 else "navy"
                    ax.text(x[xi], ypos, f"{arrow}{abs(delta):.1f}",
                            ha="center", va="bottom", fontsize=5.5,
                            color=col, fontweight="bold")

            net_label = f"{YEONET_LABELS[net]}{row_title_suffix}"
            ax.set_title(net_label, fontsize=7.5, fontweight="bold",
                         color=net_color, pad=3)
            ax.set_xticks(x)
            ax.set_xticklabels(cond_list, rotation=45, ha="right", fontsize=5.5)
            if ni == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            else:
                ax.set_yticks([])
            ax.axhline(0, color="grey", linewidth=0.4, linestyle="--")

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, n_nets, figsize=(16, 9),
                                 gridspec_kw={"hspace": 0.6, "wspace": 0.08})
        fig.suptitle(f"Within-Network Connectivity (Schaefer-100)  |  {SUBJECT_LABEL}  (GSR off)\n"
                     "Top = raw Pearson r  |  Bottom = z-score normalised across 7 networks per run",
                     fontsize=12, fontweight="bold")

        _draw_row(axes[0], "mean_r",   "Within-network\nmean r (raw)",
                  "\n(raw r)", show_delta=False)
        _draw_row(axes[1], "mean_r_z", "Within-network\nz-score (vs. other nets)",
                  "\n(z-norm)", show_delta=True)   # delta arrows on z-norm row

        # Shared legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=SESSION_COLORS[s], alpha=0.85)
                   for s in sessions]
        labels  = [SESSION_LABELS[s] for s in sessions]
        fig.legend(handles, labels, loc="lower center", ncol=3,
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))

        # Row-level annotation
        for col_label, row_idx, xpos in [("Raw r", 0, -0.01), ("Z-norm", 1, -0.01)]:
            fig.text(xpos, 0.73 - row_idx * 0.42, col_label,
                     rotation=90, va="center", ha="right",
                     fontsize=9, color="dimgrey", style="italic")

        out = os.path.join(OUT_DIR, "fig7_within_network.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    from subject_configs import CONFIGS

    parser = argparse.ArgumentParser(description="psilocybin session comparison figures")
    parser.add_argument("--subject", default="sub-002",
                        choices=list(CONFIGS.keys()),
                        help="Subject ID to analyse (default: sub-002)")
    args = parser.parse_args()

    _apply_config(CONFIGS[args.subject])

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Subject  -> {SUBJECT_LABEL}")
    print(f"TS dir   -> {TS_DIR}")
    print(f"Output   -> {OUT_DIR}\n")

    for rest_cond in REST_CONDITIONS:
        fig1_rest_matrices(rest_cond)
    fig2_music_matrices()
    fig3_global_metrics()
    fig4_network_heatmaps()
    fig5_ventral_striatum()
    fig6_differences()
    fig7_within_network()

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
