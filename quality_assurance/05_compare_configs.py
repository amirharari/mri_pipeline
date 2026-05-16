"""
QA 05: Compare all three denoising configurations side-by-side.

Reads outputs/quality_assurance/denoising_effect/config{1,2,3}_*/denoising_summary.csv
and produces:
  - comparison_table.csv        per-run metrics from all configs merged
  - comparison_grand_mean.csv   single-row summary per config
  - comparison_plot.png         grouped bar chart
  - scatter_per_run.png         per-run scatter by config + subject
  - comparison_results.xlsx     workbook: Grand Mean / Per-Run / Per-Session / Cross-Run

Usage:
  python 05_compare_configs.py                          # default denoising_effect folder
  python 05_compare_configs.py --base path/to/folder   # override output folder
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (  # noqa: E402
    QA_DENOISING_DIR,
    SESSION_LABELS as _CFG_SESSION_LABELS,
    ensure_nilearn_cache,
)

ensure_nilearn_cache()

_DEFAULT_BASE = QA_DENOISING_DIR

# Resolved at runtime via --base arg (see main())
BASE_OUT: str = _DEFAULT_BASE

# Session labels derived from config.py
SESSION_LABELS = {k: f"{k}  ({v})" for k, v in _CFG_SESSION_LABELS.items()}

CONFIGS = {
    "lean":                ("config1_lean",                 "Config 1: Lean"),
    "anatomical":          ("config2_anatomical",           "Config 2: Anatomical"),
    "global":              ("config3_global",               "Config 3: Global"),
    "research":            ("config4_research",             "Config 4: Research Std"),
    "lean_scrubbed":       ("config1_lean_scrubbed",        "Config 1: Lean+Scrub"),
    "anatomical_scrubbed": ("config2_anatomical_scrubbed",  "Config 2: Anat+Scrub"),
    "global_scrubbed":     ("config3_global_scrubbed",      "Config 3: Global+Scrub"),
}

METRICS = {
    # Signal quality
    "delta_tSNR_pct":         "tSNR Improvement (%)",
    "post_GCOR":              "Post-denoising GCOR",
    "variance_explained_pct": "Variance Explained (%)",
    # Ciric 2017 benchmarks
    "post_qcfc":              "QC-FC mean|r| (post)  [LOWER]",
    "post_dm_fc":             "DM-FC r (post)        [TOWARD 0]",
    "dof_lost":               "DoF Lost (regressors) [LOWER]",
    "post_modularity":        "Modularity contrast Q  [HIGHER]",
    "post_louvain_q":         "Louvain Q (BCT/Ciric) [HIGHER]",
    # Sanity checks
    "post_motor_LR":          "Motor L/R Corr (post) [HIGH ~0.9]",
    "post_dmn_da":            "DMN/DorsAttn (post)   [MED]",
    "post_v1_a1":             "V1/A1 neg-ctrl (post) [LOW]",
    # Motion / cost
    "mean_fd":                "Mean FD (mm)",
    "n_scrubbed":             "Scrubbed Volumes (mean)",
}


def load_config(folder: str) -> pd.DataFrame:
    csv = os.path.join(BASE_OUT, folder, "denoising_summary.csv")
    if not os.path.isfile(csv):
        print(f"  [MISSING] {csv}")
        return pd.DataFrame()
    return pd.read_csv(csv)


def load_cross_run(folder: str) -> dict:
    """Load cross-run QC-FC metrics from cross_run_qcfc.csv."""
    csv = os.path.join(BASE_OUT, folder, "cross_run_qcfc.csv")
    if not os.path.isfile(csv):
        return {}
    df = pd.read_csv(csv)
    return df.iloc[0].to_dict() if len(df) else {}


# Session labels that match the 'session' column in denoising_summary.csv
SESSION_ORDER = ["ses-1", "ses-2", "ses-3"]


def build_per_session_table(dfs: dict) -> pd.DataFrame:
    """Aggregate metrics by (config, session).

    ses-1 = Baseline, ses-2 = Psilocybin, ses-3 = Follow-up.
    Returns a DataFrame with one row per (config_label, session_label).
    """
    metric_cols = list(METRICS.keys())
    rows = []
    for name, df in dfs.items():
        dv = df.dropna(subset=["pre_tSNR"])
        if "session" not in dv.columns:
            continue
        for ses in SESSION_ORDER:
            sub = dv[dv["session"] == ses]
            if sub.empty:
                continue
            row = {
                "config":          CONFIGS[name][1],
                "session":         ses,
                "session_label":   SESSION_LABELS.get(ses, ses),
                "n_runs":          len(sub),
            }
            for col in metric_cols:
                row[col] = sub[col].dropna().mean() if col in sub else np.nan
            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


CONFIG_COLORS = {
    "lean":       "#2196F3",   # blue
    "anatomical": "#4CAF50",   # green
    "global":     "#F44336",   # red
    "research":   "#9C27B0",   # purple
}

SUBJECT_MARKERS = {
    "sub-001": "o",
    "sub-002": "s",
    "sub-003": "^",
    "sub-004": "D",
}

SCATTER_METRICS = [
    ("delta_tSNR_pct",         "tSNR Improvement (%)"),
    ("post_GCOR",              "Post-GCOR (lower = better)"),
    ("post_qcfc",              "QC-FC mean|r| (lower = better)"),
    ("post_dm_fc",             "DM-FC r (toward 0 = better)"),
    ("dof_lost",               "DoF Lost"),
    ("post_modularity",        "Network Modularity Q (higher = better)"),
    ("post_motor_LR",          "Motor L/R correlation"),
    ("post_dmn_da",            "DMN / DorsAttn correlation"),
    ("post_v1_a1",             "V1 / A1 neg-control"),
    ("variance_explained_pct", "Variance Explained (%)"),
    ("mean_fd",                "Mean FD (mm)"),
]


def _make_scatter_plots(df: pd.DataFrame, out_dir: str) -> None:
    """One subplot per metric; each dot = one run. Color = config, shape = subject."""
    subjects = sorted(df["subject"].dropna().unique())
    configs  = sorted(df["config"].dropna().unique())

    # Build legend handles
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    color_handles = [
        mpatches.Patch(color=CONFIG_COLORS.get(c, "#999"),
                       label=CONFIGS[c][1] if c in CONFIGS else c)
        for c in configs
    ]
    marker_handles = [
        mlines.Line2D([], [], color="gray",
                      marker=SUBJECT_MARKERS.get(s, "x"),
                      linestyle="None", markersize=7, label=s)
        for s in subjects
    ]

    n_plots = len(SCATTER_METRICS)
    ncols   = 4
    nrows   = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes = axes.flatten()

    for ax_idx, (col, ylabel) in enumerate(SCATTER_METRICS):
        ax = axes[ax_idx]
        if col not in df.columns:
            ax.axis("off")
            continue

        # x-axis: numeric config position with jitter per subject so dots don't overlap
        cfg_order = [c for c in ["lean", "anatomical", "global"] if c in configs]
        cfg_pos   = {c: i for i, c in enumerate(cfg_order)}
        n_subj    = max(len(subjects), 1)

        for _, row in df.dropna(subset=[col]).iterrows():
            cfg     = row["config"]
            subj    = row.get("subject", "")
            val     = row[col]
            color   = CONFIG_COLORS.get(cfg, "#999999")
            marker  = SUBJECT_MARKERS.get(subj, "x")
            s_idx   = subjects.index(subj) if subj in subjects else 0
            # spread subjects left-right around the config tick
            jitter  = (s_idx - (n_subj - 1) / 2) * 0.15
            x       = cfg_pos.get(cfg, 0) + jitter

            ax.scatter(x, val, color=color, marker=marker,
                       s=55, alpha=0.80, linewidths=0.5, edgecolors="k")

        # Config mean marker (horizontal line)
        for cfg, xpos in cfg_pos.items():
            vals = df.loc[df["config"] == cfg, col].dropna()
            if len(vals):
                ax.hlines(vals.mean(), xpos - 0.35, xpos + 0.35,
                          colors=CONFIG_COLORS.get(cfg, "#555"),
                          linewidths=2.5, zorder=5)

        ax.set_xticks(list(cfg_pos.values()))
        ax.set_xticklabels(
            [CONFIGS[c][1].replace("Config 1: ", "").replace("Config 2: ", "").replace("Config 3: ", "")
             for c in cfg_order],
            fontsize=8, rotation=10, ha="right")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=9, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Hide unused axes
    for ax in axes[n_plots:]:
        ax.axis("off")

    # Shared legend
    all_handles = color_handles + [
        mlines.Line2D([], [], linestyle="None", label="")  # spacer
    ] + marker_handles
    fig.legend(handles=all_handles, loc="lower right",
               ncol=2, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.98, 0.01))

    plt.suptitle("Per-Run Scatter — Config (color) × Subject (shape)\n"
                 "Horizontal bar = config mean",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    scatter_path = os.path.join(out_dir, "scatter_per_run.png")
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Scatter plot      -> {scatter_path}")


def main():
    global BASE_OUT

    parser = argparse.ArgumentParser(
        description="Compare denoising configurations (QA-05)")
    parser.add_argument(
        "--base", default=None,
        help="Override output folder (default: outputs/quality_assurance/denoising_effect)")
    args = parser.parse_args()

    if args.base:
        BASE_OUT = os.path.abspath(args.base)
    else:
        BASE_OUT = os.path.abspath(_DEFAULT_BASE)

    print(f"Reading from: {BASE_OUT}\n")

    dfs = {}
    cross_run = {}
    for name, (folder, label) in CONFIGS.items():
        df = load_config(folder)
        if df.empty:
            print(f"  Skipping {label} — CSV not found.")
            continue
        df["config"] = name
        df["config_label"] = label
        dfs[name] = df
        cross_run[name] = load_cross_run(folder)

    if not dfs:
        print("No config CSVs found. Run 04_denoising_effect.py first.")
        sys.exit(1)

    # -------------------------------------------------------------------
    # Grand mean table
    # -------------------------------------------------------------------
    rows = []
    for name, df in dfs.items():
        dv = df.dropna(subset=["pre_tSNR"])
        row = {"config": CONFIGS[name][1]}
        row["n_runs"]       = len(dv)
        row["n_regressors"] = dv["n_regressors"].dropna().mean() if "n_regressors" in dv else np.nan
        for col in METRICS:
            row[col] = dv[col].dropna().mean() if col in dv else np.nan
        rows.append(row)

    grand = pd.DataFrame(rows)
    csv_gm = os.path.join(BASE_OUT, "comparison_grand_mean.csv")
    grand.to_csv(csv_gm, index=False)
    print(f"\nGrand-mean table -> {csv_gm}")

    # Pretty-print
    print("\n" + "=" * 100)
    print(f"  DENOISING CONFIGURATION COMPARISON  (grand means across all runs)")
    print("=" * 100)
    col_w = 40
    print(f"  {'Metric':<{col_w}}", end="")
    for name in dfs:
        print(f"  {CONFIGS[name][1]:<26}", end="")
    print()
    print("-" * 100)
    for col, label in METRICS.items():
        print(f"  {label:<{col_w}}", end="")
        for name in dfs:
            val = grand.loc[grand["config"] == CONFIGS[name][1], col].values
            if val.size and not np.isnan(val[0]):
                print(f"  {val[0]:>26.3f}", end="")
            else:
                print(f"  {'N/A':>26}", end="")
        print()

    # Cross-run QC-FC section
    print("-" * 100)
    print(f"  {'--- Cross-run gold standard ---':<{col_w}}", end="")
    print()
    for cr_key, cr_label in [
        ("qcfc_mean_abs", "Cross-run QC-FC mean|r|  [LOWER]"),
        ("qcfc_pct_sig",  "Cross-run % sig edges    [LOWER]"),
        ("cross_run_dm_fc","Cross-run DM-FC r        [TOWARD 0]"),
    ]:
        print(f"  {cr_label:<{col_w}}", end="")
        for name in dfs:
            val = cross_run.get(name, {}).get(cr_key, np.nan)
            if not (isinstance(val, float) and np.isnan(val)):
                print(f"  {float(val):>26.3f}", end="")
            else:
                print(f"  {'N/A':>26}", end="")
        print()
    print("=" * 100)

    # -------------------------------------------------------------------
    # Per-session aggregation  (ses-1=Baseline, ses-2=Psilocybin, ses-3=Follow-up)
    # -------------------------------------------------------------------
    ses_df = build_per_session_table(dfs)
    if not ses_df.empty:
        print("\n" + "=" * 110)
        print("  PER-SESSION MEANS  (ses-1 = Baseline | ses-2 = Psilocybin | ses-3 = Follow-up)")
        print("=" * 110)
        key_metrics = [
            ("delta_tSNR_pct",         "tSNR Improvement (%)"),
            ("post_GCOR",              "Post-GCOR"),
            ("post_qcfc",              "QC-FC mean|r|"),
            ("post_dm_fc",             "DM-FC r"),
            ("post_modularity",        "Modularity Q (contrast)"),
            ("post_louvain_q",         "Louvain Q"),
            ("post_motor_LR",          "Motor L/R"),
            ("post_v1_a1",             "V1/A1 neg-ctrl"),
            ("mean_fd",                "Mean FD (mm)"),
            ("n_scrubbed",             "Scrubbed Vols"),
        ]
        col_w = 26
        # Group by config so each config block shows ses-1/2/3 side-by-side
        for name in dfs:
            cfg_label = CONFIGS[name][1]
            cfg_block = ses_df[ses_df["config"] == cfg_label]
            if cfg_block.empty:
                continue
            print(f"\n  -- {cfg_label} --")
            ses_cols = [row["session_label"] for _, row in cfg_block.iterrows()]
            print(f"  {'Metric':<{col_w}}", end="")
            for sc in ses_cols:
                print(f"  {sc:<28}", end="")
            print()
            print("  " + "-" * (col_w + 30 * len(ses_cols)))
            for col, lbl in key_metrics:
                if col not in cfg_block.columns:
                    continue
                print(f"  {lbl:<{col_w}}", end="")
                for _, row in cfg_block.iterrows():
                    val = row.get(col, np.nan)
                    if pd.notna(val):
                        print(f"  {float(val):>28.3f}", end="")
                    else:
                        print(f"  {'N/A':>28}", end="")
                print()
        print("=" * 110)

        csv_ses = os.path.join(BASE_OUT, "comparison_per_session.csv")
        try:
            ses_df.to_csv(csv_ses, index=False)
            print(f"Per-session table -> {csv_ses}")
        except PermissionError:
            print(f"  [SKIP] {csv_ses} is locked")

    # -------------------------------------------------------------------
    # Merged per-run table
    # -------------------------------------------------------------------
    all_df = pd.concat(dfs.values(), ignore_index=True)
    id_cols = ["subject", "session", "task", "acq", "run"]
    metric_cols = ["config", "config_label"] + list(METRICS.keys())
    keep = [c for c in id_cols + metric_cols if c in all_df.columns]
    all_df = all_df[keep]
    csv_all = os.path.join(BASE_OUT, "comparison_table.csv")
    try:
        all_df.to_csv(csv_all, index=False)
        print(f"Per-run merged table -> {csv_all}")
    except PermissionError:
        print(f"  [SKIP] {csv_all} is locked (close it in Excel and re-run to update)")

    # -------------------------------------------------------------------
    # Bar chart — 3 rows × 4 cols covering all key metrics
    # -------------------------------------------------------------------
    cfg_labels = [CONFIGS[n][1] for n in dfs]
    n_configs  = len(dfs)
    short_labels = [l.replace("Config 1: ", "").replace("Config 2: ", "").replace("Config 3: ", "")
                    for l in cfg_labels]

    plot_metrics = [
        # Row 1 — Signal quality
        ("delta_tSNR_pct",         "tSNR Improvement (%)",       "#4CAF50", None),
        ("post_GCOR",              "Post-GCOR (lower=better)",   "#2196F3", None),
        ("variance_explained_pct", "Variance Explained (%)",     "#795548", None),
        ("dof_lost",               "DoF Lost (regressors)",      "#607D8B", None),
        # Row 2 — Ciric 2017 benchmarks
        ("post_qcfc",              "QC-FC mean|r| (lower=better)","#FF5722", None),
        ("post_dm_fc",             "DM-FC r (toward 0=better)",  "#9C27B0", None),
        ("post_modularity",        "Modularity Q (higher=better)","#00BCD4", None),
        # Row 3 — Sanity checks
        ("post_motor_LR",          "Motor L/R (should be ~0.9)", "#FF9800", 0.9),
        ("post_dmn_da",            "DMN/DorsAttn (should be >0)","#E91E63", 0.0),
        ("post_v1_a1",             "V1/A1 neg-ctrl (lower=better)","#8BC34A",None),
        ("mean_fd",                "Mean FD (mm)",               "#9E9E9E", None),
    ]

    # Pad to fill a 3×4 grid
    while len(plot_metrics) < 12:
        plot_metrics.append(None)

    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    axes = axes.flatten()

    for ax_idx, entry in enumerate(plot_metrics):
        ax = axes[ax_idx]
        if entry is None:
            ax.axis("off")
            continue
        col, ylabel, color, hline = entry
        vals = []
        for name in dfs:
            dv  = dfs[name].dropna(subset=["pre_tSNR"])
            val = dv[col].dropna().mean() if col in dv else np.nan
            vals.append(val)

        bars = ax.bar(range(n_configs), vals, color=color, alpha=0.80, edgecolor="k")
        if hline is not None:
            ax.axhline(hline, color="red", lw=1.2, ls="--", alpha=0.7)
        ax.set_xticks(range(n_configs))
        ax.set_xticklabels(short_labels, fontsize=7, rotation=15, ha="right")
        ax.set_title(ylabel, fontsize=9, fontweight="bold")

        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + abs(bar.get_height()) * 0.02,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Denoising Configuration Comparison — All Ciric 2017 Benchmarks",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(BASE_OUT, "comparison_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot -> {plot_path}")

    # -------------------------------------------------------------------
    # Scatter plots — per-run dots, config=color, subject=marker shape
    # -------------------------------------------------------------------
    _make_scatter_plots(all_df, BASE_OUT)

    # -------------------------------------------------------------------
    # Excel workbook — 4 sheets
    # -------------------------------------------------------------------
    xlsx_path = os.path.join(BASE_OUT, "comparison_results.xlsx")
    try:
      _xlsx_open = open(xlsx_path, "a") if os.path.exists(xlsx_path) else None
      if _xlsx_open: _xlsx_open.close()
    except PermissionError:
      print(f"  [SKIP] {xlsx_path} is locked (close it in Excel and re-run to update)")
      print("\nDone.")
      return
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:

        # Sheet 1 — Grand mean (one row per config, human-readable labels)
        grand_renamed = grand.copy()
        grand_renamed.rename(columns={"config": "Configuration", **METRICS}, inplace=True)
        grand_renamed.to_excel(writer, sheet_name="Grand Mean", index=False)

        # Sheet 2 — Per-run merged table
        all_df.to_excel(writer, sheet_name="Per-Run", index=False)

        # Sheet 3 — Per-session means (ses-1=Baseline, ses-2=Psilocybin, ses-3=Follow-up)
        if not ses_df.empty:
            ses_renamed = ses_df.copy()
            ses_renamed.rename(columns={"config": "Configuration",
                                         "session_label": "Session",
                                         **METRICS}, inplace=True)
            # Put human-readable session label first, drop raw session col
            cols_first = ["Configuration", "Session", "n_runs"]
            remaining  = [c for c in ses_renamed.columns if c not in cols_first + ["session"]]
            ses_renamed = ses_renamed[cols_first + remaining]
            ses_renamed.to_excel(writer, sheet_name="Per-Session", index=False)

        # Sheet 4 — Cross-run gold standard
        cr_rows = []
        for name, (folder, label) in CONFIGS.items():
            if name not in cross_run:
                continue
            row = {"Configuration": label}
            row["Cross-run QC-FC mean|r|"]   = cross_run[name].get("qcfc_mean_abs", np.nan)
            row["Cross-run % sig edges (%)"] = cross_run[name].get("qcfc_pct_sig",  np.nan)
            row["Cross-run DM-FC r"]         = cross_run[name].get("cross_run_dm_fc", np.nan)
            cr_rows.append(row)
        pd.DataFrame(cr_rows).to_excel(writer, sheet_name="Cross-Run QC-FC", index=False)

    print(f"Excel workbook    -> {xlsx_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
