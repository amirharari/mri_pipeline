"""group_analysis.py
Group-level connectivity analysis across all subjects (sub-002, sub-003, sub-004).

Since n=3, we avoid inferential statistics and instead show:
  - Group-average matrices (Fisher-z averaged across subjects)
  - Individual-subject spaghetti lines so readers can see consistency vs outliers
  - Group means as thick markers

Figures produced
----------------
  figG1_group_rest_matrices.png       — Fisher-z averaged AAL + Schaefer × session + diff
  figG2_group_global_metrics.png      — Spaghetti plots: global metrics (REST)
  figG3_group_within_network.png      — Within-Yeo-network connectivity × session (REST)
  figG4_group_condition_comparison.png — REST vs GLASS mean connectivity × session

Reuses directly from psilo_session_comparison.py:
  conn_matrix, fisher_z_matrix, mean_connectivity, normalize_matrix,
  aal_sorted_order, schaefer_sorted_order, _draw_conn_panel,
  _add_network_borders, _add_lobe_borders,
  build_network_matrix, integration_index, gcor, schaefer_network,
  YEONET_ORDER, YEONET_LABELS, YEONET_COLORS, PLOT_STYLE

Usage
-----
  cd connectivity
  python group_analysis.py
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── import shared primitives from psilo_session_comparison ────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from psilo_session_comparison import (
    conn_matrix, fisher_z_matrix, mean_connectivity, normalize_matrix,
    aal_sorted_order, schaefer_sorted_order, _draw_conn_panel,
    _add_network_borders, _add_lobe_borders,
    build_network_matrix, integration_index, gcor, schaefer_network,
    YEONET_ORDER, YEONET_LABELS, YEONET_COLORS, PLOT_STYLE,
)
from utils import fisher_z_mean
from subject_configs import CONFIGS

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
_REPO    = os.path.dirname(_HERE)
OUT_DIR  = os.path.join(_REPO, "outputs", "group_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# Subjects to include (the three subjects for which data exists)
SUBJECTS = ["sub-002", "sub-003", "sub-004"]

SESSION_ORDER  = ["ses-1", "ses-2", "ses-3"]
SESSION_LABELS = {"ses-1": "Baseline", "ses-2": "Psilocybin", "ses-3": "Follow-up"}
SESSION_COLORS = {"ses-1": "#2196F3", "ses-2": "#F44336", "ses-3": "#4CAF50"}

# Subject-level visual style
SUB_MARKERS  = {"sub-002": "o", "sub-003": "s", "sub-004": "^"}
SUB_COLORS   = {"sub-002": "#444444", "sub-003": "#888888", "sub-004": "#BBBBBB"}
SUB_LABELS   = {"sub-002": "sub-002", "sub-003": "sub-003 (Nimrod)", "sub-004": "sub-004"}

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts_path(cfg: dict, condition: str, session: str, atlas: str) -> Optional[str]:
    """Return the timeseries CSV path for a given subject/condition/session/atlas, or None."""
    prefix = cfg["conditions"].get(condition, {}).get(session)
    if prefix is None:
        return None
    return os.path.join(cfg["ts_dir"], f"{prefix}_{atlas}_ts.csv")


def _load_df(cfg: dict, condition: str, session: str, atlas: str) -> Optional[pd.DataFrame]:
    p = _ts_path(cfg, condition, session, atlas)
    if p is None or not os.path.isfile(p):
        return None
    return pd.read_csv(p)


def group_avg_matrix(
    condition: str, session: str, atlas: str
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """
    Fisher-z average correlation matrices across all subjects for one
    condition × session × atlas.  Returns (mean_r_matrix, column_names).
    """
    z_mats, cols_ref = [], None
    for sub in SUBJECTS:
        cfg = CONFIGS[sub]
        df  = _load_df(cfg, condition, session, atlas)
        if df is None:
            continue
        c = conn_matrix(df)
        z_mats.append(fisher_z_matrix(c))
        if cols_ref is None:
            cols_ref = list(df.columns)
    if not z_mats:
        return None, None
    return np.tanh(np.mean(z_mats, axis=0)), cols_ref


def per_subject_metric(
    condition: str, session: str, atlas: str,
    metric_fn,  # callable: (pd.DataFrame) -> float
) -> Dict[str, Optional[float]]:
    """Return {sub_id: metric_value} for every subject (None if data missing)."""
    out = {}
    for sub in SUBJECTS:
        df = _load_df(CONFIGS[sub], condition, session, atlas)
        out[sub] = metric_fn(df) if df is not None else None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure G1: Group-average REST connectivity matrices
#   Rows: AAL (lobe-sorted), Schaefer-100 (network-sorted)
#   Cols: Baseline | Psilocybin | Follow-up | Psilo − Baseline difference
# ─────────────────────────────────────────────────────────────────────────────

def figG1_group_rest_matrices() -> None:
    plt.rcParams.update(PLOT_STYLE)
    condition = "rest"
    atlases   = [("aal", "AAL (116 ROIs, lobe-sorted)"),
                 ("schaefer100", "Schaefer-100 (Yeo 7 networks)")]

    n_rows = len(atlases)
    n_cols = 4  # Baseline, Psilocybin, Follow-up, Diff
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 4 * n_rows),
                             constrained_layout=True)
    fig.suptitle("Group-Average Resting-State Connectivity\n"
                 f"(Fisher-z mean across n={len(SUBJECTS)} subjects)",
                 fontsize=13, fontweight="bold")

    col_titles = ["Baseline", "Psilocybin", "Follow-up", "Psilo − Baseline"]

    for row, (atlas, row_label) in enumerate(atlases):
        matrices = {}
        cols_ref  = None
        for ses in SESSION_ORDER:
            mat, cols = group_avg_matrix(condition, ses, atlas)
            if mat is not None:
                matrices[ses] = mat
                if cols_ref is None:
                    cols_ref = cols

        if cols_ref is None:
            continue

        # Apply lobe / network sorting
        if atlas == "aal":
            order = aal_sorted_order(cols_ref)
        else:
            order = schaefer_sorted_order(cols_ref)
        sorted_cols = [cols_ref[i] for i in order]

        vmax = max(
            np.percentile(np.abs(matrices[s]), 97)
            for s in matrices if matrices[s] is not None
        )

        for col_i, ses in enumerate(SESSION_ORDER):
            ax  = axes[row, col_i]
            mat = matrices.get(ses)
            if mat is None:
                ax.set_visible(False)
                continue

            c = mat[np.ix_(order, order)]
            label = SESSION_LABELS[ses]
            _draw_conn_panel(
                ax, c, title="" if row > 0 else col_titles[col_i],
                ylabel=row_label if col_i == 0 else "",
                vmin=-vmax, vmax=vmax, cbar_label="r",
                gcor_val=float(np.nanmean(c[~np.eye(len(order), dtype=bool)])),
                ses_color=SESSION_COLORS[ses],
            )
            if atlas == "aal":
                _add_lobe_borders(ax, sorted_cols)
            else:
                _add_network_borders(ax, sorted_cols)
            if row == 0:
                ax.set_title(col_titles[col_i], fontsize=10, fontweight="bold",
                             color=SESSION_COLORS[ses])

        # Difference column: Psilocybin − Baseline
        ax_diff = axes[row, 3]
        mat_base  = matrices.get("ses-1")
        mat_psilo = matrices.get("ses-2")
        if mat_base is not None and mat_psilo is not None:
            diff = mat_psilo - mat_base
            d    = diff[np.ix_(order, order)]
            dmax = np.percentile(np.abs(d), 98)
            _draw_conn_panel(
                ax_diff, d,
                title="" if row > 0 else col_titles[3],
                ylabel="" ,
                vmin=-dmax, vmax=dmax, cbar_label="Δr",
            )
            if atlas == "aal":
                _add_lobe_borders(ax_diff, sorted_cols)
            else:
                _add_network_borders(ax_diff, sorted_cols)
            if row == 0:
                ax_diff.set_title(col_titles[3], fontsize=10, fontweight="bold")
        else:
            ax_diff.set_visible(False)

    out_path = os.path.join(OUT_DIR, "figG1_group_rest_matrices.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figG1 -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure G2: Global metrics spaghetti plots (REST, Schaefer atlas)
#   4 panels: Mean Pairwise Connectivity | GCOR | Integration Index | Within-DMN
#   Lines = individual subjects, thick dot = group mean
# ─────────────────────────────────────────────────────────────────────────────

def _within_network_r(df: pd.DataFrame, network: str) -> float:
    """Mean within-network Fisher-z-averaged r for one Yeo network."""
    cols = [c for c in df.columns if schaefer_network(c) == network]
    if len(cols) < 2:
        return float("nan")
    from utils import within_network_connectivity
    res = within_network_connectivity(df, cols)
    return res["mean"] if res else float("nan")


def figG2_group_global_metrics() -> None:
    plt.rcParams.update(PLOT_STYLE)
    condition = "rest"
    atlas     = "schaefer100"

    metrics = {
        "Mean Connectivity":  lambda df: mean_connectivity(conn_matrix(df)),
        "GCOR":               lambda df: gcor(df),
        "Integration Index":  lambda df: integration_index(build_network_matrix(df)[0]),
        "Within-DMN (r)":     lambda df: _within_network_r(df, "Default"),
        "Within-Visual (r)":  lambda df: _within_network_r(df, "Vis"),
        "Within-Control (r)": lambda df: _within_network_r(df, "Cont"),
    }

    n_metrics = len(metrics)
    n_cols    = 3
    n_rows    = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 4 * n_rows),
                             constrained_layout=True)
    axes = axes.ravel()
    fig.suptitle("Group-Level Global Connectivity Metrics — Resting State\n"
                 "(lines = individual subjects; filled circle = group mean)",
                 fontsize=12, fontweight="bold")

    x_ticks   = [0, 1, 2]
    x_labels  = [SESSION_LABELS[s] for s in SESSION_ORDER]

    for ax_i, (metric_name, metric_fn) in enumerate(metrics.items()):
        ax = axes[ax_i]

        # Collect per-subject values across sessions
        sub_vals: Dict[str, List[Optional[float]]] = {}
        for sub in SUBJECTS:
            vals = []
            for ses in SESSION_ORDER:
                df = _load_df(CONFIGS[sub], condition, ses, atlas)
                vals.append(metric_fn(df) if df is not None else None)
            sub_vals[sub] = vals

        # Draw individual subject lines
        for sub in SUBJECTS:
            ys = sub_vals[sub]
            xs_valid = [x_ticks[i] for i, y in enumerate(ys) if y is not None]
            ys_valid = [y for y in ys if y is not None]
            ax.plot(xs_valid, ys_valid,
                    color=SUB_COLORS[sub], linewidth=1.5, alpha=0.7,
                    marker=SUB_MARKERS[sub], markersize=8,
                    label=SUB_LABELS[sub], zorder=2)

        # Group mean line (only sessions where all subjects have data)
        group_means = []
        for i, ses in enumerate(SESSION_ORDER):
            vals_ses = [sub_vals[s][i] for s in SUBJECTS if sub_vals[s][i] is not None]
            group_means.append(np.mean(vals_ses) if vals_ses else None)

        xs_g = [x_ticks[i] for i, v in enumerate(group_means) if v is not None]
        ys_g = [v for v in group_means if v is not None]
        if xs_g:
            ax.plot(xs_g, ys_g, color="black", linewidth=3, linestyle="--",
                    marker="o", markersize=10, zorder=3, label="Group mean")

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel(metric_name, fontsize=9)
        ax.set_title(metric_name, fontsize=10, fontweight="bold")

        # Shade psilocybin session
        ax.axvspan(0.5, 1.5, color="#F44336", alpha=0.07, zorder=0)

        if ax_i == 0:
            ax.legend(fontsize=7, loc="best")

    # Hide any unused axes
    for ax_i in range(n_metrics, len(axes)):
        axes[ax_i].set_visible(False)

    # Add a single shared legend for subjects in the last used panel
    handles = [
        mlines.Line2D([], [], color=SUB_COLORS[s], marker=SUB_MARKERS[s],
                      markersize=7, linewidth=1.5, label=SUB_LABELS[s])
        for s in SUBJECTS
    ] + [
        mlines.Line2D([], [], color="black", linewidth=3, linestyle="--",
                      marker="o", markersize=8, label="Group mean")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))

    out_path = os.path.join(OUT_DIR, "figG2_group_global_metrics.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figG2 -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure G3: Within-network connectivity × session (REST, Yeo 7 networks)
#   Grouped bar chart + individual subject dots
# ─────────────────────────────────────────────────────────────────────────────

def figG3_group_within_network() -> None:
    plt.rcParams.update(PLOT_STYLE)
    condition = "rest"
    atlas     = "schaefer100"

    from utils import within_network_connectivity

    # Collect {sub: {ses: {net: r}}}
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sub in SUBJECTS:
        data[sub] = {}
        for ses in SESSION_ORDER:
            df = _load_df(CONFIGS[sub], condition, ses, atlas)
            data[sub][ses] = {}
            if df is None:
                continue
            for net in YEONET_ORDER:
                cols = [c for c in df.columns if schaefer_network(c) == net]
                res  = within_network_connectivity(df, cols)
                data[sub][ses][net] = res["mean"] if res else float("nan")

    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    fig.suptitle("Group Within-Network Connectivity — Resting State\n"
                 "(bars = group mean; dots = individual subjects)",
                 fontsize=12, fontweight="bold")

    n_nets   = len(YEONET_ORDER)
    n_ses    = len(SESSION_ORDER)
    width    = 0.22
    x_base   = np.arange(n_nets)

    offsets = np.linspace(-width, width, n_ses)

    for si, ses in enumerate(SESSION_ORDER):
        ses_label = SESSION_LABELS[ses]
        ses_color = SESSION_COLORS[ses]
        means  = []
        for ni, net in enumerate(YEONET_ORDER):
            vals = [data[sub][ses].get(net, float("nan")) for sub in SUBJECTS]
            vals = [v for v in vals if not np.isnan(v)]
            means.append(np.mean(vals) if vals else float("nan"))

        bars = ax.bar(x_base + offsets[si], means, width=width * 0.9,
                      color=ses_color, alpha=0.75, label=ses_label,
                      edgecolor="white", linewidth=0.5)

        # Individual subject dots
        for ni, net in enumerate(YEONET_ORDER):
            for sub in SUBJECTS:
                v = data[sub][ses].get(net, float("nan"))
                if not np.isnan(v):
                    ax.scatter(x_base[ni] + offsets[si], v,
                               color="black", s=25, zorder=4, alpha=0.7,
                               marker=SUB_MARKERS[sub])

    ax.set_xticks(x_base)
    ax.set_xticklabels([YEONET_LABELS[n] for n in YEONET_ORDER], fontsize=9)
    ax.set_ylabel("Within-network mean r (Fisher-z averaged)", fontsize=10)
    ax.set_title("")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    # Network background shading
    for ni, net in enumerate(YEONET_ORDER):
        if ni % 2 == 0:
            ax.axvspan(ni - 0.5, ni + 0.5, color=YEONET_COLORS[net], alpha=0.08, zorder=0)

    # Legends: sessions + subject markers
    session_handles = [
        mpatches.Patch(facecolor=SESSION_COLORS[s], label=SESSION_LABELS[s],
                       alpha=0.75, edgecolor="white")
        for s in SESSION_ORDER
    ]
    sub_handles = [
        mlines.Line2D([], [], color="black", marker=SUB_MARKERS[s],
                      markersize=6, linewidth=0, label=SUB_LABELS[s])
        for s in SUBJECTS
    ]
    ax.legend(handles=session_handles + sub_handles, fontsize=7,
              loc="upper right", ncol=3)

    out_path = os.path.join(OUT_DIR, "figG3_group_within_network.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figG3 -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure G4: Condition × Session — REST vs GLASS (group level)
#   2 rows (AAL, Schaefer): mean connectivity spaghetti + group-average matrix diff
# ─────────────────────────────────────────────────────────────────────────────

def figG4_group_condition_comparison() -> None:
    """Compare REST and GLASS (the one music condition available for all 3 subjects
    across all 3 sessions) at the group level.

    Left column:  spaghetti plot of mean pairwise connectivity × session
    Right column: group-average difference matrix (Glass − Rest) per session
                  for AAL and Schaefer
    """
    plt.rcParams.update(PLOT_STYLE)

    conditions_plot = ["rest", "glass"]
    cond_labels      = {"rest": "Rest", "glass": "Glass (Philip Glass)"}
    cond_colors      = {"rest": "#607D8B", "glass": "#FF9800"}
    cond_linestyles  = {"rest": "-", "glass": "--"}

    atlases = [("aal", "AAL"), ("schaefer100", "Schaefer-100")]

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    fig.suptitle("Group-Level Condition Comparison — REST vs GLASS\n"
                 "(n=3 subjects, all sessions)",
                 fontsize=12, fontweight="bold")

    # Layout: 2 rows (AAL, Schaefer) × 3 cols (spaghetti | Diff Baseline | Diff Psilocybin)
    n_rows  = len(atlases)
    gs      = fig.add_gridspec(n_rows, 3, width_ratios=[1.8, 1, 1])

    for row_i, (atlas, atlas_label) in enumerate(atlases):

        # ── Left: spaghetti of mean connectivity ──────────────────────────────
        ax_line = fig.add_subplot(gs[row_i, 0])

        for cond in conditions_plot:
            for sub in SUBJECTS:
                ys = []
                xs_valid = []
                for xi, ses in enumerate(SESSION_ORDER):
                    df = _load_df(CONFIGS[sub], cond, ses, atlas)
                    if df is None:
                        continue
                    c = conn_matrix(df)
                    ys.append(mean_connectivity(c))
                    xs_valid.append(xi)

                ax_line.plot(xs_valid, ys,
                             color=cond_colors[cond],
                             linestyle=cond_linestyles[cond],
                             linewidth=1.2, alpha=0.55,
                             marker=SUB_MARKERS[sub], markersize=7)

            # Group mean
            group_ys, group_xs = [], []
            for xi, ses in enumerate(SESSION_ORDER):
                vals = []
                for sub in SUBJECTS:
                    df = _load_df(CONFIGS[sub], cond, ses, atlas)
                    if df is not None:
                        vals.append(mean_connectivity(conn_matrix(df)))
                if vals:
                    group_ys.append(np.mean(vals))
                    group_xs.append(xi)
            ax_line.plot(group_xs, group_ys,
                         color=cond_colors[cond], linewidth=3,
                         linestyle=cond_linestyles[cond],
                         marker="o", markersize=10, zorder=3,
                         label=cond_labels[cond])

        ax_line.set_xticks([0, 1, 2])
        ax_line.set_xticklabels(["Baseline", "Psilocybin", "Follow-up"], fontsize=9)
        ax_line.set_ylabel("Mean pairwise r", fontsize=9)
        ax_line.set_title(f"{atlas_label} — Mean connectivity", fontsize=10, fontweight="bold")
        ax_line.axvspan(0.5, 1.5, color="#F44336", alpha=0.07, zorder=0)

        # Subject marker legend (first row only)
        if row_i == 0:
            cond_handles = [
                mlines.Line2D([], [], color=cond_colors[c],
                              linestyle=cond_linestyles[c],
                              linewidth=2, label=cond_labels[c])
                for c in conditions_plot
            ]
            sub_handles = [
                mlines.Line2D([], [], color="gray", marker=SUB_MARKERS[s],
                              markersize=6, linewidth=0, label=SUB_LABELS[s])
                for s in SUBJECTS
            ]
            thick = mlines.Line2D([], [], color="gray", linewidth=3,
                                  marker="o", label="Group mean")
            ax_line.legend(handles=cond_handles + sub_handles + [thick],
                           fontsize=7, loc="best")

        # ── Middle: group-average Glass − Rest matrix for Baseline ─────────────
        for diff_col_i, diff_ses in enumerate(["ses-1", "ses-2"]):
            ax_diff = fig.add_subplot(gs[row_i, diff_col_i + 1])
            mat_r, cols_r = group_avg_matrix("rest",  diff_ses, atlas)
            mat_g, cols_g = group_avg_matrix("glass", diff_ses, atlas)

            if mat_r is None or mat_g is None:
                ax_diff.set_visible(False)
                continue

            diff = mat_g - mat_r
            if atlas == "aal":
                order = aal_sorted_order(cols_r)
                sorted_cols = [cols_r[i] for i in order]
            else:
                order = schaefer_sorted_order(cols_r)
                sorted_cols = [cols_r[i] for i in order]

            d = diff[np.ix_(order, order)]
            dmax = np.percentile(np.abs(d), 98)
            ses_lbl = SESSION_LABELS[diff_ses]
            _draw_conn_panel(
                ax_diff, d,
                title=f"Glass − Rest ({ses_lbl})",
                ylabel=atlas_label if diff_col_i == 0 else "",
                vmin=-dmax, vmax=dmax, cbar_label="Δr",
            )
            if atlas == "aal":
                _add_lobe_borders(ax_diff, sorted_cols)
            else:
                _add_network_borders(ax_diff, sorted_cols)

    out_path = os.path.join(OUT_DIR, "figG4_group_condition_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figG4 -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure G5: Psilocybin effect size heatmap (group-average Δ, REST)
#   Shows which specific region-pairs change most consistently across subjects.
#   One panel per atlas.  Sorted by lobe / network.
# ─────────────────────────────────────────────────────────────────────────────

def figG5_group_effect_map() -> None:
    """
    Group-consistency Δ connectivity map: Psilocybin − Baseline.

    For each ROI pair: compute mean Δr across subjects AND how many subjects
    show the same sign (consistency score 1/2/3).  Overlay both in a 2×2 grid
    (AAL | Schaefer) × (mean Δr | consistency).
    """
    plt.rcParams.update(PLOT_STYLE)
    condition = "rest"
    atlases = [("aal", "AAL (lobe-sorted)"), ("schaefer100", "Schaefer (Yeo networks)")]

    n_cols_per_atlas = 2  # [mean Δr, consistency]
    n_cols = n_cols_per_atlas * len(atlases)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5),
                             constrained_layout=True)
    fig.suptitle("Psilocybin Effect Map — Group REST Connectivity (Psilo − Baseline)\n"
                 "Left: mean Δr across subjects   |   Right: sign consistency (out of 3)",
                 fontsize=11, fontweight="bold")

    for ai, (atlas, atlas_label) in enumerate(atlases):
        # Collect per-subject difference matrices
        diff_mats, cols_ref = [], None
        for sub in SUBJECTS:
            cfg  = CONFIGS[sub]
            df_b = _load_df(cfg, condition, "ses-1", atlas)
            df_p = _load_df(cfg, condition, "ses-2", atlas)
            if df_b is None or df_p is None:
                continue
            cb   = conn_matrix(df_b)
            cp   = conn_matrix(df_p)
            diff_mats.append(cp - cb)
            if cols_ref is None:
                cols_ref = list(df_b.columns)

        if not diff_mats or cols_ref is None:
            continue

        if atlas == "aal":
            order       = aal_sorted_order(cols_ref)
            sorted_cols = [cols_ref[i] for i in order]
        else:
            order       = schaefer_sorted_order(cols_ref)
            sorted_cols = [cols_ref[i] for i in order]

        mean_diff = np.mean(diff_mats, axis=0)
        mean_d    = mean_diff[np.ix_(order, order)]
        dmax      = np.percentile(np.abs(mean_d), 98)

        # Consistency: number of subjects showing same sign as group mean
        n_same_sign = np.zeros_like(mean_d)
        for dm in diff_mats:
            d_ord = dm[np.ix_(order, order)]
            n_same_sign += (np.sign(d_ord) == np.sign(mean_d)).astype(float)
        np.fill_diagonal(n_same_sign, 0)

        # Mean Δr panel
        ax_delta = axes[ai * 2]
        _draw_conn_panel(
            ax_delta, mean_d,
            title=f"{atlas_label}\nMean Δr (Psilo − Baseline)",
            ylabel="",
            vmin=-dmax, vmax=dmax, cbar_label="Δr",
        )
        if atlas == "aal":
            _add_lobe_borders(ax_delta, sorted_cols)
        else:
            _add_network_borders(ax_delta, sorted_cols)

        # Consistency panel
        ax_cons = axes[ai * 2 + 1]
        im = ax_cons.imshow(n_same_sign, cmap="YlOrRd", vmin=0, vmax=3,
                            aspect="auto", interpolation="nearest")
        ax_cons.set_title(f"{atlas_label}\nSign consistency (n subjects)", fontsize=9,
                          fontweight="bold")
        ax_cons.set_xticks([])
        ax_cons.set_yticks([])
        plt.colorbar(im, ax=ax_cons, fraction=0.046, pad=0.04,
                     label="# subjects same sign",
                     ticks=[0, 1, 2, 3])
        if atlas == "aal":
            _add_lobe_borders(ax_cons, sorted_cols)
        else:
            _add_network_borders(ax_cons, sorted_cols)

    out_path = os.path.join(OUT_DIR, "figG5_group_effect_map.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figG5 -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Group analysis  ->  {OUT_DIR}")
    print(f"Subjects: {SUBJECTS}\n")
    figG1_group_rest_matrices()
    figG2_group_global_metrics()
    figG3_group_within_network()
    figG4_group_condition_comparison()
    figG5_group_effect_map()
    print(f"\nAll group figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
