"""
Subject Overview — full session × condition grid
=================================================
Produces one figure per subject:
  - Rows    = conditions (rest, glass, tundra, bailero, …)
  - Columns = sessions   (ses-1 Baseline | ses-2 Psilocybin | ses-3 Follow-up)
  - Each cell = Schaefer-400 FC matrix heatmap annotated with
                GCOR / DVARS / FD / scrub%
  - Gray "Not collected" for missing cells

Usage
-----
    cd connectivity
    python subject_overview.py                          # anatomical config
    python subject_overview.py --data PATH/TO/TS_DIR
    python subject_overview.py --data PATH --subjects sub-001 sub-002
"""
from __future__ import annotations

import argparse
import os
import re
import sys

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
SESSIONS       = ["ses-1", "ses-2", "ses-3"]
SESSION_LABELS = {"ses-1": "Baseline", "ses-2": "Psilocybin", "ses-3": "Follow-up"}
SESSION_COLORS = {"ses-1": "#4A90D9", "ses-2": "#E94E77", "ses-3": "#5CB85C"}

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

# Human-readable labels for task keys
TASK_LABELS = {
    "rest":              "Rest",
    "restchecktr2":      "Rest\n(checkTR)",
    "music_acq-glass":   "Glass",
    "music_acq-bailero": "Bailero",
    "music_acq-tundra":  "Tundra",
    "music_acq-personal":"Personal",
    "music_acq-personal1":"Personal 1",
    "music_acq-personal2":"Personal 2",
    "music_acq-agami":   "Agami",
    "music_acq-salome":  "Salomé",
}

# Canonical row order (rest first, then music pieces)
TASK_ORDER = [
    "rest", "restchecktr2",
    "music_acq-glass",
    "music_acq-bailero",
    "music_acq-tundra",
    "music_acq-personal",
    "music_acq-personal1", "music_acq-personal2",
    "music_acq-agami",
    "music_acq-salome",
]

# Non-canonical task keys → canonical key
# (same musical piece recorded under different BIDS task/acq conventions)
TASK_ALIASES: dict[str, str] = {
    "glass":             "music_acq-glass",    # task-glass == same piece as acq-glass
    "music_acq-baliero": "music_acq-bailero",  # BIDS typo in some sessions
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _task_key_from_filename(fname: str) -> str | None:
    """Extract the task key (= report task field) from a CSV filename.

    e.g. sub-001_ses-2_task-music_acq-glass_run-1_schaefer400_ts.csv
         → 'music_acq-glass'
    """
    m = re.search(r"task-(.+?)_run-", fname)
    return m.group(1) if m else None


def _session_from_filename(fname: str) -> str | None:
    m = re.search(r"(ses-\d+)", fname)
    return m.group(1) if m else None


def _subject_from_filename(fname: str) -> str | None:
    m = re.match(r"(sub-\d+)_", fname)
    return m.group(1) if m else None


def _network_sort_order(cols: list) -> list:
    """Indices that sort parcels by Yeo network."""
    def rank(col):
        for i, net in enumerate(YEO_NETWORKS):
            if net in col:
                return i
        return len(YEO_NETWORKS)
    return sorted(range(len(cols)), key=lambda i: rank(cols[i]))


def _network_boundaries(cols_ordered: list) -> list:
    """X/Y positions of network boundary lines."""
    boundaries = []
    cur = None
    for i, col in enumerate(cols_ordered):
        net = next((n for n in YEO_NETWORKS if n in col), "Other")
        if net != cur:
            if cur is not None:
                boundaries.append(i - 0.5)
            cur = net
    return boundaries


def _network_midpoints(cols_ordered: list) -> list[tuple]:
    """Return [(midpoint_index, network_name, color), ...] for each network block."""
    midpoints = []
    cur_net, start = None, 0
    for i, col in enumerate(cols_ordered + [None]):
        net = next((n for n in YEO_NETWORKS if n in col), "Other") if col else None
        if net != cur_net:
            if cur_net is not None:
                mid = (start + i - 1) / 2.0
                midpoints.append((mid, cur_net, YEO_COLORS.get(cur_net, "#888888")))
            cur_net, start = net, i
    return midpoints


# Short abbreviations for cramped axis labels
_NET_SHORT = {
    "Vis":         "Vis",
    "SomMot":      "SomMot\n(M1+A1)",
    "DorsAttn":    "DAttn",
    "SalVentAttn": "Sal",
    "Limbic":      "Limb",
    "Cont":        "Cont\n(FPN)",
    "Default":     "DMN",
}


def _add_network_axis_labels(ax, cols_ordered: list,
                              show_y: bool = True, show_x: bool = False):
    """Add network name tick-labels + colored left/bottom strips to a matrix cell."""
    midpoints = _network_midpoints(cols_ordered)
    n = len(cols_ordered)

    if show_y:
        ax.set_yticks([m[0] for m in midpoints])
        ax.set_yticklabels(
            [_NET_SHORT.get(m[1], m[1]) for m in midpoints],
            fontsize=4.8, va="center",
        )
        # Color each tick label to match the Yeo network
        for tick, (_, net, col) in zip(ax.get_yticklabels(), midpoints):
            tick.set_color(col)
            tick.set_fontweight("bold")
        ax.tick_params(axis="y", length=0, pad=1)

        # Thin colored strip on the left edge (one rectangle per network)
        boundaries = _network_boundaries(cols_ordered)
        starts = [-0.5] + [b for b in boundaries]
        ends   = [b for b in boundaries] + [n - 0.5]
        for (s, e), (_, net, col) in zip(zip(starts, ends), midpoints):
            ax.axhspan(s, e, xmin=0, xmax=0.018,
                       facecolor=col, alpha=0.85, clip_on=True, zorder=5)
    else:
        ax.set_yticks([])

    if show_x:
        ax.set_xticks([m[0] for m in midpoints])
        ax.set_xticklabels(
            [_NET_SHORT.get(m[1], m[1]) for m in midpoints],
            fontsize=4.8, ha="center", rotation=45,
        )
        for tick, (_, net, col) in zip(ax.get_xticklabels(), midpoints):
            tick.set_color(col)
            tick.set_fontweight("bold")
        ax.tick_params(axis="x", length=0, pad=1)
    else:
        ax.set_xticks([])


def _compute_fc(csv_path: str, cols_sorted: list) -> np.ndarray | None:
    """Load CSV, return FC matrix sorted by Yeo network order."""
    try:
        df   = pd.read_csv(csv_path)
        corr = np.corrcoef(df.values.T)
        np.fill_diagonal(corr, 0)
        order = _network_sort_order(list(df.columns))
        return corr[np.ix_(order, order)]
    except Exception as e:
        print(f"    [WARN] Could not load {os.path.basename(csv_path)}: {e}")
        return None


def _normalize_task(task: str) -> str:
    """Map non-canonical task keys to their canonical equivalent."""
    return TASK_ALIASES.get(task, task)


def _scan_directory(ts_dir: str) -> dict:
    """Return {subject: {session: {canonical_task_key: filepath}}} for all Schaefer CSVs."""
    index: dict = {}
    for fname in sorted(os.listdir(ts_dir)):
        if not fname.endswith("_schaefer400_ts.csv"):
            continue
        sub  = _subject_from_filename(fname)
        ses  = _session_from_filename(fname)
        task = _task_key_from_filename(fname)
        if not (sub and ses and task):
            continue
        canonical = _normalize_task(task)
        fpath = os.path.join(ts_dir, fname)
        index.setdefault(sub, {}).setdefault(ses, {})[canonical] = fpath
    return index


def _build_high_motion_set(report: pd.DataFrame | None) -> set:
    """Return set of (subject, session, canonical_task) tuples that were high-motion skipped."""
    if report is None:
        return set()
    skipped = report[report["high_motion_skip"] == True]  # noqa: E712
    result = set()
    for _, row in skipped.iterrows():
        canonical = _normalize_task(str(row["task"]))
        scrub = row.get("scrub_percent", np.nan)
        result.add((row["subject"], row["session"], canonical, scrub))
    return result


def _load_qc_report(ts_dir: str) -> pd.DataFrame | None:
    """Load scrubbing_report_filtered.csv if present."""
    path = os.path.join(ts_dir, "scrubbing_report_filtered.csv")
    if os.path.isfile(path):
        return pd.read_csv(path)
    return None


def _get_qc(report: pd.DataFrame | None, subject: str,
             session: str, task_key: str) -> dict:
    """Look up per-run QC metrics from the scrubbing report."""
    null = {"fd": np.nan, "gcor": np.nan, "dvars": np.nan, "scrub": np.nan}
    if report is None:
        return null
    mask = ((report["subject"] == subject) &
            (report["session"] == session) &
            (report["task"]    == task_key))
    rows = report[mask]
    if rows.empty:
        return null
    row = rows.iloc[0]
    return {
        "fd":    row.get("fd_mean_filtered", np.nan),
        "gcor":  row.get("gcor_schaefer",    np.nan),
        "dvars": row.get("dvars_post",        np.nan),
        "scrub": row.get("scrub_percent",     np.nan),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject overview figure
# ─────────────────────────────────────────────────────────────────────────────

def make_subject_overview(subject: str, ses_task_map: dict,
                           report: pd.DataFrame | None,
                           high_motion_set: set,
                           out_dir: str, cols: list):
    """Create the full subject overview grid and save."""

    # Collect all unique task keys and sort canonically
    all_tasks = set()
    for ses_dict in ses_task_map.values():
        all_tasks.update(ses_dict.keys())

    # Sort by canonical order, then alphabetically for unknowns
    def task_sort_key(t):
        try:
            return TASK_ORDER.index(t)
        except ValueError:
            return len(TASK_ORDER) + sorted(all_tasks).index(t)

    sorted_tasks = sorted(all_tasks, key=task_sort_key)

    n_rows = len(sorted_tasks)
    n_cols = len(SESSIONS)

    if n_rows == 0:
        print(f"  [{subject}] No data found — skipping")
        return

    # Figure layout: each cell = (cell_w × cell_h) inches
    # Extra left margin for Yeo network y-axis labels, extra bottom for x-axis labels
    cell_w, cell_h = 3.4, 3.8
    fig_w = cell_w * n_cols + 1.4   # +1.4 for row label column + y-tick labels
    fig_h = cell_h * n_rows + 1.2   # +1.2 for column headers + x-tick labels

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        f"{subject}  ·  Functional Connectivity Overview  "
        f"(Schaefer-400 cortical · Yeo-7 networks · anatomical denoising)",
        fontsize=13, fontweight="bold", y=0.995,
    )

    # Build a GridSpec: +1 col on left for row labels, +1 row on top for headers
    gs = fig.add_gridspec(
        n_rows + 1, n_cols + 1,
        left=0.10, right=0.97, top=0.97, bottom=0.04,
        hspace=0.04, wspace=0.06,
        width_ratios=[0.18] + [1.0] * n_cols,
        height_ratios=[0.06] + [1.0] * n_rows,
    )

    # Column headers (session labels)
    for ci, ses in enumerate(SESSIONS):
        ax_hdr = fig.add_subplot(gs[0, ci + 1])
        ax_hdr.set_facecolor(SESSION_COLORS[ses])
        ax_hdr.text(0.5, 0.5, SESSION_LABELS[ses],
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white",
                    transform=ax_hdr.transAxes)
        ax_hdr.set_xticks([]); ax_hdr.set_yticks([])
        for spine in ax_hdr.spines.values():
            spine.set_visible(False)

    # Top-left corner: blank
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tl.set_visible(False)

    # Color range: compute shared vabs across all available matrices
    all_r_vals = []
    for ri, task in enumerate(sorted_tasks):
        for ses in SESSIONS:
            fp = ses_task_map.get(ses, {}).get(task)
            if fp:
                try:
                    df_tmp = pd.read_csv(fp)
                    c = np.corrcoef(df_tmp.values.T)
                    np.fill_diagonal(c, np.nan)
                    all_r_vals.extend(c[~np.isnan(c)].ravel().tolist())
                except Exception:
                    pass
    vabs = float(np.percentile(np.abs(all_r_vals), 96)) if all_r_vals else 0.5
    vabs = max(0.25, min(vabs, 0.75))

    order = _network_sort_order(cols)
    cols_ordered = [cols[i] for i in order]
    boundaries = _network_boundaries(cols_ordered)

    for ri, task in enumerate(sorted_tasks):
        # Row label
        ax_lbl = fig.add_subplot(gs[ri + 1, 0])
        ax_lbl.set_facecolor("#F5F5F5")
        label = TASK_LABELS.get(task, task.replace("music_acq-", "").replace("_", " ").title())
        ax_lbl.text(0.5, 0.5, label,
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color="#333333",
                    rotation=0, transform=ax_lbl.transAxes,
                    wrap=True)
        ax_lbl.set_xticks([]); ax_lbl.set_yticks([])
        for spine in ax_lbl.spines.values():
            spine.set_color("#CCCCCC")

        for ci, ses in enumerate(SESSIONS):
            ax = fig.add_subplot(gs[ri + 1, ci + 1])
            fp = ses_task_map.get(ses, {}).get(task)

            if fp is None:
                # Check if this cell was high-motion skipped (run existed but was too noisy)
                hm_match = next(
                    ((sub, s, t, pct) for (sub, s, t, pct) in high_motion_set
                     if sub == subject and s == ses and t == task),
                    None,
                )
                if hm_match:
                    scrub_pct = hm_match[3]
                    ax.set_facecolor("#FFF0D0")
                    ax.text(0.5, 0.6, "HIGH MOTION",
                            ha="center", va="center", fontsize=8,
                            fontweight="bold", color="#CC6600",
                            transform=ax.transAxes)
                    ax.text(0.5, 0.35, "skipped",
                            ha="center", va="center", fontsize=7.5,
                            color="#CC6600", style="italic",
                            transform=ax.transAxes)
                    if not np.isnan(scrub_pct):
                        ax.text(0.5, 0.15, f"({scrub_pct:.0f}% scrubbed)",
                                ha="center", va="center", fontsize=7,
                                color="#AA4400", transform=ax.transAxes)
                    for spine in ax.spines.values():
                        spine.set_color("#CC6600")
                        spine.set_linewidth(1.8)
                else:
                    ax.set_facecolor("#EBEBEB")
                    ax.text(0.5, 0.5, "Not\ncollected",
                            ha="center", va="center", fontsize=8,
                            color="#AAAAAA", style="italic",
                            transform=ax.transAxes)
                    for spine in ax.spines.values():
                        spine.set_color("#CCCCCC")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            fc = _compute_fc(fp, cols)
            if fc is None:
                ax.set_facecolor("#FFE0E0")
                ax.text(0.5, 0.5, "Load\nerror",
                        ha="center", va="center", fontsize=8,
                        color="#CC4444", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            # FC heatmap
            ax.imshow(fc, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                      interpolation="nearest", aspect="equal",
                      origin="upper")

            # Network boundary lines
            for b in boundaries:
                ax.axhline(b, color="white", lw=0.4, alpha=0.7)
                ax.axvline(b, color="white", lw=0.4, alpha=0.7)

            # Network axis labels: y on leftmost col, x on bottom row
            _add_network_axis_labels(
                ax, cols_ordered,
                show_y=(ci == 0),
                show_x=(ri == n_rows - 1),
            )

            # Thin border in session colour
            for spine in ax.spines.values():
                spine.set_color(SESSION_COLORS[ses])
                spine.set_linewidth(1.8)

            # QC annotation (bottom of cell)
            qc = _get_qc(report, subject, ses, task)
            gcor_str  = f"GCOR={qc['gcor']:.3f}"  if not np.isnan(qc['gcor'])  else "GCOR=—"
            dvars_str = f"DVARS={qc['dvars']:.2f}" if not np.isnan(qc['dvars']) else "DVARS=—"
            fd_str    = f"FD={qc['fd']:.3f}"       if not np.isnan(qc['fd'])    else "FD=—"
            scrub_str = f"scrub={qc['scrub']:.1f}%"if not np.isnan(qc['scrub']) else ""

            # Colour-code GCOR: green / orange / red
            gcor_val = qc["gcor"]
            if np.isnan(gcor_val):
                gcor_col = "#555555"
            elif gcor_val > 0.3:
                gcor_col = "#CC3333"
            elif gcor_val > 0.2:
                gcor_col = "#E07B39"
            else:
                gcor_col = "#2A7A2A"

            # DVARS flag colour
            dvars_val = qc["dvars"]
            dvars_col = "#CC3333" if (not np.isnan(dvars_val) and dvars_val > 1.5) else "#555555"

            ann_lines = [
                (f"{fd_str}  {scrub_str}", "#444444"),
                (gcor_str,                  gcor_col),
                (dvars_str,                 dvars_col),
            ]
            n = len(fc)
            for li, (txt, col) in enumerate(ann_lines):
                y_pos = n - 1 - li * (n / 12)
                ax.text(n * 0.02, y_pos, txt,
                        fontsize=5.5, color="white",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.1",
                                  facecolor=col, alpha=0.75,
                                  edgecolor="none"),
                        va="top")

    # Shared colorbar (right side)
    cbar_ax = fig.add_axes([0.975, 0.08, 0.012, 0.82])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                                norm=plt.Normalize(vmin=-vabs, vmax=vabs))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Pearson r", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Network legend (bottom)
    legend_patches = [mpatches.Patch(color=c, label=n)
                      for n, c in YEO_COLORS.items()]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=7, fontsize=7, framealpha=0.8,
               title="Yeo 7-network order (left→right, top→bottom in each matrix)",
               title_fontsize=7,
               bbox_to_anchor=(0.5, -0.005))

    plt.savefig(os.path.join(out_dir, f"{subject}_full_overview.png"),
                dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {subject}_full_overview.png  "
          f"({n_rows} conditions × {n_cols} sessions)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(ts_dir: str, subjects: list | None = None):
    index  = _scan_directory(ts_dir)
    report = _load_qc_report(ts_dir)

    if subjects:
        index = {k: v for k, v in index.items() if k in subjects}

    if not index:
        print(f"No subjects found in {ts_dir}")
        return

    # Read column names from any Schaefer file
    sample = next(
        (os.path.join(ts_dir, f) for f in os.listdir(ts_dir)
         if f.endswith("_schaefer400_ts.csv")),
        None,
    )
    if sample is None:
        print("No Schaefer-400 CSV found.")
        return
    cols = pd.read_csv(sample, nrows=0).columns.tolist()

    out_dir = os.path.join(ts_dir, "figures", "overview")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nSource : {ts_dir}")
    print(f"Output : {out_dir}")
    print(f"{'─'*60}")

    high_motion_set = _build_high_motion_set(report)

    for sub in sorted(index):
        print(f"\n  {sub}")
        ses_task_map: dict = {}
        for ses, task_dict in index[sub].items():
            ses_task_map[ses] = task_dict

        make_subject_overview(sub, ses_task_map, report,
                              high_motion_set, out_dir, cols)

    print(f"\n{'='*60}")
    print(f"  Done.  Figures -> {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=config.TS_OUTPUT_DIR_ANATOMICAL,
        help="Timeseries directory",
    )
    parser.add_argument(
        "--subjects", nargs="*", default=None,
        help="Restrict to subjects e.g. sub-001 sub-002",
    )
    args = parser.parse_args()
    main(args.data, args.subjects)
