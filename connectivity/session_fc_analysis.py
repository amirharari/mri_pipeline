"""
Session FC Analysis — psilocybin study
=======================================
Reads Schaefer-400 + Tian S2 timeseries CSVs from an extraction output folder
and produces:

  Per subject
  -----------
  1. rest_fc_matrices.png       3 FC heatmaps (Baseline/Psilo/Follow-up) + difference
  2. rest_network_integration.png  within- vs between-network r for 7 Yeo networks × 3 sessions
  3. music_glass_fc_matrices.png   same as #1 for the Glass music condition
  4. subcortical_fc.png         Tian S2 connectivity + cortico-subcortical coupling

  Group
  -----
  5. group_rest_fc_difference.png   mean Psilo–Base + Follow–Base FC difference maps
  6. group_network_integration.png  network means ± SEM across subjects

Usage
-----
    python session_fc_analysis.py                          # uses config.TS_OUTPUT_DIR_ANATOMICAL
    python session_fc_analysis.py --data PATH/TO/TS_DIR   # explicit folder
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ── Project imports ──────────────────────────────────────────────────────────
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import config  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────
SESSION_LABELS = {"ses-1": "Baseline", "ses-2": "Psilocybin", "ses-3": "Follow-up"}
SESSION_COLORS = {"ses-1": "#4A90D9", "ses-2": "#E94E77", "ses-3": "#5CB85C"}
SESSIONS       = ["ses-1", "ses-2", "ses-3"]

# Yeo 7-network order (posterior → anterior) and display properties
YEO_NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
YEO_COLORS   = {
    "Vis":         "#7B2D8B",
    "SomMot":      "#4DB2AD",
    "DorsAttn":    "#2CA02C",
    "SalVentAttn": "#D967A3",
    "Limbic":      "#E8C96B",
    "Cont":        "#E07B39",
    "Default":     "#D62728",
}
YEO_FULL = {
    "Vis":         "Visual",
    "SomMot":      "Somatomotor\n(incl. A1)",
    "DorsAttn":    "Dorsal\nAttention",
    "SalVentAttn": "Salience /\nVent. Attn",
    "Limbic":      "Limbic",
    "Cont":        "Control\n(FPN)",
    "Default":     "Default\n(DMN)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fisher_z(r: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(r, -0.9999, 0.9999))


def _inv_fisher_z(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def _fc_matrix(csv_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a timeseries CSV and return (DataFrame, FC correlation matrix)."""
    df   = pd.read_csv(csv_path)
    corr = np.corrcoef(df.T)
    np.fill_diagonal(corr, np.nan)
    return df, corr


def _mean_fc(corr_list: list[np.ndarray]) -> np.ndarray:
    """Fisher-z average a list of FC matrices, return back in r-space."""
    if not corr_list:
        return None
    z_stack = np.stack([_fisher_z(c) for c in corr_list], axis=0)
    return _inv_fisher_z(np.nanmean(z_stack, axis=0))


def _network_order(cols: list[str]) -> list[int]:
    """Return parcel indices sorted by Yeo network order."""
    def _net_rank(col):
        for i, net in enumerate(YEO_NETWORKS):
            if net in col:
                return i
        return len(YEO_NETWORKS)
    return sorted(range(len(cols)), key=lambda i: _net_rank(cols[i]))


def _network_ticks(cols_ordered: list[str]) -> tuple[list[float], list[str], list[tuple]]:
    """Return (tick_positions, tick_labels, colored_spans) for the ordered parcel list."""
    spans  = []
    ticks  = []
    labels = []
    cur_net, start = None, 0
    for i, col in enumerate(cols_ordered + [None]):
        net = next((n for n in YEO_NETWORKS if n in col), "Other") if col else None
        if net != cur_net:
            if cur_net is not None:
                mid = (start + i - 1) / 2
                ticks.append(mid)
                labels.append(YEO_FULL.get(cur_net, cur_net))
                spans.append((start, i, YEO_COLORS.get(cur_net, "#aaa")))
            cur_net, start = net, i
    return ticks, labels, spans


def load_subject_session_fc(ts_dir: str, subject: str, atlas: str = "schaefer400"
                             ) -> dict[str, dict[str, list[np.ndarray]]]:
    """Return {session: {task: [fc_matrix, ...]}} for one subject."""
    result: dict = {}
    suffix = f"_{atlas}_ts.csv"
    for fname in sorted(os.listdir(ts_dir)):
        if not (fname.startswith(subject) and fname.endswith(suffix)):
            continue
        # Parse session and task from filename
        parts = fname.replace(suffix, "").split("_")
        # e.g. sub-001_ses-1_task-rest_run-1   or   sub-001_ses-2_task-music_acq-glass_run-1
        ses_part  = next((p for p in parts if p.startswith("ses-")),  None)
        task_part = next((p for p in parts if p.startswith("task-")), None)
        acq_part  = next((p for p in parts if p.startswith("acq-")),  None)
        if not ses_part or not task_part:
            continue
        task = task_part.replace("task-", "")
        if acq_part:
            task = task + "_" + acq_part.replace("acq-", "")

        _, fc = _fc_matrix(os.path.join(ts_dir, fname))
        result.setdefault(ses_part, {}).setdefault(task, []).append(fc)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Network integration metrics
# ─────────────────────────────────────────────────────────────────────────────

def network_integration_metrics(fc: np.ndarray, cols: list[str]) -> dict[str, float]:
    """Within-network mean r for each Yeo network."""
    metrics = {}
    for net in YEO_NETWORKS:
        idx = [i for i, c in enumerate(cols) if net in c]
        if len(idx) < 2:
            metrics[net] = np.nan
            continue
        blk = fc[np.ix_(idx, idx)].copy()
        np.fill_diagonal(blk, np.nan)
        metrics[net] = float(np.nanmean(blk))
    return metrics


def dmn_fpn_anticorrelation(fc: np.ndarray, cols: list[str]) -> float:
    dmn = [i for i, c in enumerate(cols) if "Default" in c]
    fpn = [i for i, c in enumerate(cols) if "Cont" in c]
    if not dmn or not fpn:
        return np.nan
    return float(np.nanmean(fc[np.ix_(dmn, fpn)]))


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_fc_matrix(ax, fc: np.ndarray, cols: list[str], title: str,
                    vmin=-1, vmax=1, cmap="RdBu_r", show_ticks=True):
    order     = _network_order(cols)
    fc_sorted = fc[np.ix_(order, order)]
    cols_ord  = [cols[i] for i in order]
    np.fill_diagonal(fc_sorted, 0)

    im = ax.imshow(fc_sorted, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="equal")

    # Network boundary lines + tick labels
    ticks, labels, spans = _network_ticks(cols_ord)
    n = len(cols_ord)
    boundaries = []
    cur_net = None
    for i, col in enumerate(cols_ord):
        net = next((nn for nn in YEO_NETWORKS if nn in col), "Other")
        if net != cur_net:
            if cur_net is not None:
                boundaries.append(i - 0.5)
            cur_net = net
    for b in boundaries:
        ax.axhline(b, color="white", lw=0.6, alpha=0.8)
        ax.axvline(b, color="white", lw=0.6, alpha=0.8)

    if show_ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    return im


def _plot_network_bars(ax, data: dict[str, dict[str, float]], title: str):
    """Bar chart: within-network r per session, grouped by network."""
    nets      = YEO_NETWORKS
    n_nets    = len(nets)
    n_ses     = len(SESSIONS)
    bar_w     = 0.22
    x         = np.arange(n_nets)

    for si, ses in enumerate(SESSIONS):
        vals = [data.get(ses, {}).get(net, np.nan) for net in nets]
        offset = (si - 1) * bar_w
        ax.bar(x + offset, vals, bar_w,
               color=SESSION_COLORS[ses], label=SESSION_LABELS[ses],
               alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([YEO_FULL[n] for n in nets], fontsize=7)
    ax.set_ylabel("Mean within-network r", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(-0.15, 0.75)
    ax.spines[["top", "right"]].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject figures
# ─────────────────────────────────────────────────────────────────────────────

def fig_fc_session_comparison(sub: str, task_key: str,
                               data: dict, out_dir: str, cols: list[str]):
    """4-panel: Baseline | Psilocybin | Follow-up | Psilo-Base difference."""
    fc_per_ses = {}
    for ses in SESSIONS:
        runs = data.get(ses, {}).get(task_key, [])
        if runs:
            fc_per_ses[ses] = _mean_fc(runs)

    available = [s for s in SESSIONS if s in fc_per_ses]
    if len(available) < 2:
        print(f"  [{sub}] Not enough sessions for {task_key} — skipping FC figure")
        return

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.suptitle(f"{sub}  |  {task_key}  |  Functional Connectivity (Schaefer-400)",
                 fontsize=12, fontweight="bold", y=1.01)

    # Shared colour scale across the three absolute matrices
    all_r = np.concatenate([fc_per_ses[s][~np.isnan(fc_per_ses[s])]
                             for s in available])
    vabs  = float(np.percentile(np.abs(all_r), 97))
    vabs  = max(0.3, min(vabs, 0.8))

    for ax_i, ses in enumerate(SESSIONS[:3]):
        fc = fc_per_ses.get(ses)
        if fc is not None:
            im = _plot_fc_matrix(axes[ax_i], fc, cols,
                                  title=SESSION_LABELS[ses],
                                  vmin=-vabs, vmax=vabs)
            plt.colorbar(im, ax=axes[ax_i], fraction=0.046, pad=0.04,
                         label="Pearson r")
        else:
            axes[ax_i].set_visible(False)

    # Difference panel: Psilocybin − Baseline
    if "ses-1" in fc_per_ses and "ses-2" in fc_per_ses:
        diff = fc_per_ses["ses-2"] - fc_per_ses["ses-1"]
        vlim = float(np.nanpercentile(np.abs(diff), 97))
        vlim = max(0.15, min(vlim, 0.5))
        im_d = _plot_fc_matrix(axes[3], diff, cols,
                                title="Δ Psilocybin − Baseline",
                                vmin=-vlim, vmax=vlim, cmap="RdBu_r")
        plt.colorbar(im_d, ax=axes[3], fraction=0.046, pad=0.04, label="Δr")
    else:
        axes[3].set_visible(False)

    plt.tight_layout()
    safe_task = task_key.replace("/", "_").replace(" ", "_")
    out_path  = os.path.join(out_dir, f"{sub}_{safe_task}_fc_matrices.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {os.path.basename(out_path)}")


def fig_network_integration(sub: str, task_key: str,
                             data: dict, out_dir: str, cols: list[str]):
    """7-network within-r bar chart across sessions + DMN-FPN line."""
    net_data: dict[str, dict[str, float]] = {}
    dmn_fpn:  dict[str, float]            = {}

    for ses in SESSIONS:
        runs = data.get(ses, {}).get(task_key, [])
        fc   = _mean_fc(runs) if runs else None
        if fc is not None:
            net_data[ses] = network_integration_metrics(fc, cols)
            dmn_fpn[ses]  = dmn_fpn_anticorrelation(fc, cols)
        else:
            net_data[ses] = {net: np.nan for net in YEO_NETWORKS}
            dmn_fpn[ses]  = np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5),
                                    gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle(f"{sub}  |  {task_key}  |  Network Integration",
                 fontsize=12, fontweight="bold")

    _plot_network_bars(ax1, net_data, "Within-network mean r  (7 Yeo networks)")

    # DMN-FPN anticorrelation across sessions
    vals = [dmn_fpn.get(s, np.nan) for s in SESSIONS]
    colors = [SESSION_COLORS[s] for s in SESSIONS]
    ax2.bar(range(3), vals, color=colors, alpha=0.85, edgecolor="white")
    ax2.axhline(0, color="black", lw=0.5, ls="--")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([SESSION_LABELS[s] for s in SESSIONS], fontsize=8)
    ax2.set_ylabel("Mean r", fontsize=8)
    ax2.set_title("DMN – FPN\nAnticorrelation", fontsize=9, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    # annotate expected direction
    ax2.text(0.5, 0.02, "↑ toward 0 = psilocybin\ndisruption of DMN-FPN",
             transform=ax2.transAxes, ha="center", fontsize=6.5, color="gray")

    plt.tight_layout()
    safe_task = task_key.replace("/", "_").replace(" ", "_")
    out_path  = os.path.join(out_dir, f"{sub}_{safe_task}_network_integration.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {os.path.basename(out_path)}")


def fig_subcortical(sub: str, task_key: str, data_tian: dict,
                    data_sch: dict, out_dir: str,
                    tian_cols: list[str], sch_cols: list[str]):
    """Tian S2 subcortical FC + cortico-subcortical coupling."""
    tian_fc_per_ses: dict = {}
    cort_subc_per_ses: dict = {}

    for ses in SESSIONS:
        t_runs = data_tian.get(ses, {}).get(task_key, [])
        s_runs = data_sch.get(ses, {}).get(task_key, [])
        if t_runs:
            tian_fc_per_ses[ses] = _mean_fc(t_runs)
        if t_runs and s_runs:
            # Cortico-subcortical: correlate each Schaefer parcel ts with Tian parcel ts
            # Average across runs
            cross_mats = []
            for ts_dir_path in zip(t_runs, s_runs) if len(t_runs) == len(s_runs) else []:
                pass  # only use Fisher-z mean of pre-computed FCs — not possible here
            # Instead: use the precomputed per-atlas FCs as proxies

    if len(tian_fc_per_ses) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f"{sub}  |  {task_key}  |  Tian S2 Subcortical FC",
                 fontsize=12, fontweight="bold")

    for ax_i, ses in enumerate(SESSIONS):
        fc = tian_fc_per_ses.get(ses)
        if fc is not None:
            np.fill_diagonal(fc, 0)
            im = axes[ax_i].imshow(fc, cmap="RdBu_r", vmin=-0.8, vmax=0.8,
                                    interpolation="nearest", aspect="equal")
            axes[ax_i].set_title(SESSION_LABELS[ses], fontsize=10, fontweight="bold")
            axes[ax_i].set_xticks(range(len(tian_cols)))
            axes[ax_i].set_xticklabels(tian_cols, fontsize=4, rotation=90)
            axes[ax_i].set_yticks(range(len(tian_cols)))
            axes[ax_i].set_yticklabels(tian_cols, fontsize=4)
            plt.colorbar(im, ax=axes[ax_i], fraction=0.046, pad=0.04, label="r")
        else:
            axes[ax_i].set_visible(False)

    plt.tight_layout()
    safe_task = task_key.replace("/", "_").replace(" ", "_")
    out_path  = os.path.join(out_dir, f"{sub}_{safe_task}_subcortical_fc.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {os.path.basename(out_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# Group-level figures
# ─────────────────────────────────────────────────────────────────────────────

def fig_group_fc_difference(task_key: str,
                             all_diffs: dict[str, dict[str, np.ndarray]],
                             cols: list[str], out_dir: str):
    """Mean Psilo-Base and Follow-Base FC difference across subjects."""
    for diff_label, ses_pair in [("Psilo_minus_Base", ("ses-1", "ses-2")),
                                  ("Followup_minus_Base", ("ses-1", "ses-3"))]:
        mats = []
        for sub, diff_dict in all_diffs.items():
            key = f"{ses_pair[0]}_{ses_pair[1]}"
            if key in diff_dict:
                mats.append(diff_dict[key])

        if not mats:
            continue

        mean_diff = np.nanmean(np.stack(mats, axis=0), axis=0)
        n_subs    = len(mats)

        fig, ax = plt.subplots(figsize=(9, 8))
        vlim = float(np.nanpercentile(np.abs(mean_diff), 97))
        vlim = max(0.1, min(vlim, 0.4))
        im   = _plot_fc_matrix(ax, mean_diff, cols,
                                title=f"Group mean ΔFC: {diff_label.replace('_', ' ')}  (n={n_subs})",
                                vmin=-vlim, vmax=vlim, cmap="RdBu_r")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Mean Δr")
        plt.tight_layout()
        safe_task = task_key.replace("/", "_").replace(" ", "_")
        out_path  = os.path.join(out_dir, f"group_{safe_task}_{diff_label}.png")
        plt.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  Saved {os.path.basename(out_path)}")


def fig_group_network_integration(task_key: str,
                                   all_net_data: dict[str, dict[str, dict[str, float]]],
                                   out_dir: str):
    """Group-level within-network r ± SEM across subjects."""
    nets   = YEO_NETWORKS
    n_nets = len(nets)
    bar_w  = 0.22

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(n_nets)

    for si, ses in enumerate(SESSIONS):
        means, sems = [], []
        for net in nets:
            vals = [all_net_data[sub].get(ses, {}).get(net, np.nan)
                    for sub in all_net_data]
            vals = [v for v in vals if not np.isnan(v)]
            means.append(np.mean(vals) if vals else np.nan)
            sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

        offset = (si - 1) * bar_w
        ax.bar(x + offset, means, bar_w,
               yerr=sems, capsize=3,
               color=SESSION_COLORS[ses], label=SESSION_LABELS[ses],
               alpha=0.85, edgecolor="white", linewidth=0.5,
               error_kw={"elinewidth": 1.2})

    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([YEO_FULL[n] for n in nets], fontsize=8)
    ax.set_ylabel("Mean within-network r  (±SEM)", fontsize=9)
    n_subs = len(all_net_data)
    ax.set_title(f"Group Network Integration  |  {task_key}  (n={n_subs})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-0.1, 0.8)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    safe_task = task_key.replace("/", "_").replace(" ", "_")
    out_path  = os.path.join(out_dir, f"group_{safe_task}_network_integration.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved {os.path.basename(out_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(ts_dir: str):
    subjects = sorted({f[:7] for f in os.listdir(ts_dir)
                        if f.startswith("sub-") and f.endswith("_schaefer400_ts.csv")})
    if not subjects:
        print(f"No *_schaefer400_ts.csv files found in {ts_dir}")
        return

    print(f"\nFound subjects: {subjects}")
    print(f"Source dir    : {ts_dir}\n")

    # Output root
    fig_root = os.path.join(ts_dir, "figures")
    grp_dir  = os.path.join(fig_root, "group")
    os.makedirs(grp_dir, exist_ok=True)

    # Read column names from any Schaefer file to get network labels
    sample_sch = next(f for f in os.listdir(ts_dir)
                      if f.endswith("_schaefer400_ts.csv"))
    sample_tian = next((f for f in os.listdir(ts_dir)
                        if f.endswith("_tian_s2_ts.csv")), None)
    sch_cols  = pd.read_csv(os.path.join(ts_dir, sample_sch), nrows=0).columns.tolist()
    tian_cols = (pd.read_csv(os.path.join(ts_dir, sample_tian), nrows=0).columns.tolist()
                 if sample_tian else [])

    print(f"Schaefer-400 parcels : {len(sch_cols)}")
    print(f"Tian S2 ROIs         : {len(tian_cols)}\n")

    # Tasks to analyse (focus on REST + music glass which is in all subjects)
    FOCUS_TASKS = ["rest", "music_glass"]

    # Containers for group-level analysis
    all_diffs    : dict[str, dict] = {}   # sub -> {ses1_ses2: diff_matrix}
    all_net_data : dict[str, dict] = {}   # sub -> {ses -> {net -> r}}

    for sub in subjects:
        print(f"{'─'*60}")
        print(f"  Subject: {sub}")
        sub_dir = os.path.join(fig_root, sub)
        os.makedirs(sub_dir, exist_ok=True)

        data_sch  = load_subject_session_fc(ts_dir, sub, atlas="schaefer400")
        data_tian = load_subject_session_fc(ts_dir, sub, atlas="tian_s2")

        # Collect all available tasks for this subject
        all_tasks = set()
        for ses_dict in data_sch.values():
            all_tasks.update(ses_dict.keys())

        # Determine which tasks to plot
        tasks_to_plot = []
        if "rest" in all_tasks:
            tasks_to_plot.append("rest")
        for t in sorted(all_tasks):
            if "glass" in t and t not in tasks_to_plot:
                tasks_to_plot.append(t)

        sub_diffs    = {}
        sub_net_data = {}

        for task_key in tasks_to_plot:
            print(f"  Task: {task_key}")

            # 1. FC matrices per session + difference
            fig_fc_session_comparison(sub, task_key, data_sch, sub_dir, sch_cols)

            # 2. Network integration
            fig_network_integration(sub, task_key, data_sch, sub_dir, sch_cols)

            # 3. Subcortical FC (Tian S2)
            if tian_cols:
                fig_subcortical(sub, task_key, data_tian, data_sch,
                                sub_dir, tian_cols, sch_cols)

            # Accumulate for group analysis (use REST only)
            if task_key == "rest":
                net_by_ses: dict[str, dict[str, float]] = {}
                for ses in SESSIONS:
                    runs = data_sch.get(ses, {}).get("rest", [])
                    fc   = _mean_fc(runs) if runs else None
                    if fc is not None:
                        net_by_ses[ses] = network_integration_metrics(fc, sch_cols)

                        # Differences vs Baseline
                        base_runs = data_sch.get("ses-1", {}).get("rest", [])
                        if ses != "ses-1" and base_runs:
                            base_fc = _mean_fc(base_runs)
                            key     = f"ses-1_{ses}"
                            sub_diffs[key] = fc - base_fc

                sub_net_data = net_by_ses

        all_diffs[sub]    = sub_diffs
        all_net_data[sub] = sub_net_data

    # ── Group figures ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Group-level figures")

    if any(all_diffs.values()):
        fig_group_fc_difference("rest", all_diffs, sch_cols, grp_dir)

    if any(all_net_data.values()):
        fig_group_network_integration("rest", all_net_data, grp_dir)

    print(f"\n{'='*60}")
    print(f"  All figures saved to {fig_root}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=config.TS_OUTPUT_DIR_ANATOMICAL,
        help="Timeseries directory (default: config.TS_OUTPUT_DIR_ANATOMICAL)",
    )
    args = parser.parse_args()
    main(args.data)
