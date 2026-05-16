"""timeseries_sanity_check.py

Diagnostic figure: raw timeseries for key ROIs during Glass music + global mean.

For each subject × session shows 4 stacked panels:
  1. Global brain mean (all AAL ROIs averaged) — detects global signal inflation
  2. Bilateral auditory cortex (Heschl + STG)
  3. Bilateral NAcc
  4. Framewise Displacement (motion)

If panels 1, 2, and 3 all look the same → NAcc-auditory coupling is spurious global signal.
If they differ → the coupling reflects real region-specific dynamics.
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from typing import Optional

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from psilo_session_comparison import PLOT_STYLE
from subject_configs import CONFIGS

_REPO   = os.path.dirname(_HERE)
OUT_DIR = os.path.join(_REPO, "outputs", "timeseries_sanity")
os.makedirs(OUT_DIR, exist_ok=True)

SESSION_ORDER  = ["ses-1", "ses-2", "ses-3"]
SESSION_LABELS = {"ses-1": "Baseline", "ses-2": "Psilocybin", "ses-3": "Follow-up"}
SESSION_COLORS = {"ses-1": "#2196F3",  "ses-2": "#F44336",    "ses-3": "#4CAF50"}

SMOOTH = 8   # TRs for display smoothing

AUDITORY_COLS = [
    "Heschl's Gyrus (includes H1 and H2)",
    "Superior Temporal Gyrus, anterior division",
    "Superior Temporal Gyrus, posterior division",
    "Planum Temporale",
]
NACC_COLS = ["Left Accumbens", "Right Accumbens"]


def _load(cfg, condition, session, atlas) -> Optional[pd.DataFrame]:
    prefix = cfg["conditions"].get(condition, {}).get(session)
    if prefix is None:
        return None
    p = os.path.join(cfg["ts_dir"], f"{prefix}_{atlas}_ts.csv")
    return pd.read_csv(p) if os.path.isfile(p) else None


def zscore(x: np.ndarray) -> np.ndarray:
    mu, sd = np.nanmean(x), np.nanstd(x)
    return (x - mu) / (sd + 1e-8)


def smooth(x: np.ndarray, w: int = SMOOTH) -> np.ndarray:
    return uniform_filter1d(x, size=w)


def make_timeseries_figure(condition: str = "glass") -> None:
    plt.rcParams.update(PLOT_STYLE)
    subjects = list(CONFIGS.keys())

    for sub in subjects:
        cfg = CONFIGS[sub]
        available = [s for s in SESSION_ORDER
                     if cfg["conditions"].get(condition, {}).get(s) is not None]

        if not available:
            print(f"  {sub}: no {condition} data, skipping")
            continue

        n_ses = len(available)
        # 4 panels per session column: global, auditory, NAcc, FD
        n_rows = 4
        fig, axes = plt.subplots(
            n_rows, n_ses,
            figsize=(5.5 * n_ses, 2.8 * n_rows),
            constrained_layout=True,
            sharey="row",
        )
        if n_ses == 1:
            axes = axes.reshape(n_rows, 1)

        fig.suptitle(
            f"{sub} — Timeseries Sanity Check ({condition.upper()} condition)\n"
            "Row 1: Global mean (all AAL ROIs)  |  Row 2: Auditory cortex  |  "
            "Row 3: NAcc  |  Row 4: Motion (FD proxy from global variance)",
            fontsize=10, fontweight="bold",
        )

        row_labels = [
            "Global mean\n(all AAL)",
            "Auditory\n(Heschl + STG)",
            "Bilateral\nNAcc",
            "Mean squared\nrow-diff (motion proxy)",
        ]

        for col_i, ses in enumerate(available):
            df_aal = _load(cfg, condition, ses, "aal")
            df_sub = _load(cfg, condition, ses, "ho_subcortical")
            df_cor = _load(cfg, condition, ses, "ho_cortical")

            ses_label = SESSION_LABELS[ses]
            color     = SESSION_COLORS[ses]
            n_trs     = (df_aal.shape[0] if df_aal is not None
                         else df_sub.shape[0] if df_sub is not None else 0)
            t = np.arange(n_trs)

            # ── Row 0: Global mean across all AAL ROIs ───────────────────────
            ax = axes[0, col_i]
            if df_aal is not None:
                global_mean = df_aal.mean(axis=1).values
                global_z    = smooth(zscore(global_mean))
                ax.plot(t, global_z, color=color, lw=1.0, alpha=0.85)
                ax.fill_between(t, global_z, alpha=0.15, color=color)
                # overlay each ROI lightly to show variance
                for col_name in df_aal.columns[::4]:   # every 4th to avoid clutter
                    roi_z = smooth(zscore(df_aal[col_name].values))
                    ax.plot(t, roi_z, color="gray", lw=0.3, alpha=0.15)
            ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.4)
            if col_i == 0:
                ax.set_ylabel(row_labels[0], fontsize=8, fontweight="bold")
            ax.set_title(ses_label, fontsize=10, fontweight="bold", color=color)

            # ── Row 1: Auditory cortex ───────────────────────────────────────
            ax = axes[1, col_i]
            if df_cor is not None:
                for aud_col in AUDITORY_COLS:
                    if aud_col in df_cor.columns:
                        roi_z = smooth(zscore(df_cor[aud_col].values))
                        ax.plot(t, roi_z, lw=0.9, alpha=0.65, label=aud_col[:18])
                aud_avail = [c for c in AUDITORY_COLS if c in df_cor.columns]
                if aud_avail:
                    aud_mean = df_cor[aud_avail].mean(axis=1).values
                    ax.plot(t, smooth(zscore(aud_mean)),
                            color=color, lw=2.2, alpha=0.9, label="Mean auditory")
            ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.4)
            if col_i == 0:
                ax.set_ylabel(row_labels[1], fontsize=8, fontweight="bold")
                ax.legend(fontsize=5, loc="upper right")

            # ── Row 2: NAcc ──────────────────────────────────────────────────
            ax = axes[2, col_i]
            if df_sub is not None:
                for nacc_col in NACC_COLS:
                    if nacc_col in df_sub.columns:
                        roi_z = smooth(zscore(df_sub[nacc_col].values))
                        side = "L" if "Left" in nacc_col else "R"
                        ax.plot(t, roi_z, lw=1.0, alpha=0.65, label=f"NAcc-{side}")
                avail_nacc = [c for c in NACC_COLS if c in df_sub.columns]
                if avail_nacc:
                    nacc_mean = df_sub[avail_nacc].mean(axis=1).values
                    ax.plot(t, smooth(zscore(nacc_mean)),
                            color=color, lw=2.2, alpha=0.9, label="Bilateral mean")
            ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.4)
            if col_i == 0:
                ax.set_ylabel(row_labels[2], fontsize=8, fontweight="bold")
                ax.legend(fontsize=6, loc="upper right")

            # ── Row 3: Motion proxy (frame-to-frame variance) ────────────────
            ax = axes[3, col_i]
            if df_aal is not None:
                # Frame-difference RMS across all ROIs — rough in-scanner motion proxy
                arr   = df_aal.values.astype(float)
                fd_proxy = np.concatenate([[0], np.sqrt(np.mean(np.diff(arr, axis=0) ** 2, axis=1))])
                fd_sm    = smooth(fd_proxy, w=max(SMOOTH // 2, 3))
                ax.plot(t, fd_sm, color="darkorange", lw=1.2, alpha=0.85)
                ax.fill_between(t, fd_sm, alpha=0.25, color="darkorange")
                # mark spikes > 2 std
                thresh = fd_sm.mean() + 2 * fd_sm.std()
                spikes = np.where(fd_sm > thresh)[0]
                if len(spikes):
                    ax.scatter(spikes, fd_sm[spikes], color="red", s=18, zorder=4,
                               label=f"Spikes (>{thresh:.2f})")
                    ax.legend(fontsize=6)
            ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.4)
            if col_i == 0:
                ax.set_ylabel(row_labels[3], fontsize=8, fontweight="bold")
            ax.set_xlabel("Time (TRs)", fontsize=8)

        out = os.path.join(OUT_DIR, f"{sub}_{condition}_timeseries_sanity.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {os.path.basename(out)}")


def make_overlay_comparison(condition: str = "glass") -> None:
    """
    For each subject: overlay NAcc and Auditory mean timeseries in the same panel
    per session — directly shows whether they co-fluctuate (spurious global) or diverge.
    Also shows correlation r between them printed on panel.
    """
    plt.rcParams.update(PLOT_STYLE)
    subjects = list(CONFIGS.keys())

    fig, axes = plt.subplots(
        len(subjects), len(SESSION_ORDER),
        figsize=(5 * len(SESSION_ORDER), 3.5 * len(subjects)),
        constrained_layout=True,
    )

    fig.suptitle(
        f"NAcc vs Auditory Cortex — Direct Overlay ({condition.upper()})\n"
        "Blue=Auditory, Red=NAcc, Gray=Global mean.  r = Pearson correlation between them.",
        fontsize=10, fontweight="bold",
    )

    for row_i, sub in enumerate(subjects):
        cfg = CONFIGS[sub]
        for col_i, ses in enumerate(SESSION_ORDER):
            ax      = axes[row_i, col_i]
            color   = SESSION_COLORS[ses]

            df_aal = _load(cfg, condition, ses, "aal")
            df_sub = _load(cfg, condition, ses, "ho_subcortical")
            df_cor = _load(cfg, condition, ses, "ho_cortical")

            if df_aal is None and df_sub is None:
                ax.set_visible(False)
                continue

            n_trs = (df_aal.shape[0] if df_aal is not None else df_sub.shape[0])
            t = np.arange(n_trs)

            # Global mean
            if df_aal is not None:
                gm = smooth(zscore(df_aal.mean(axis=1).values))
                ax.plot(t, gm, color="lightgray", lw=1.8, alpha=0.7,
                        label="Global mean", zorder=1)

            # Auditory mean
            aud_ts = None
            if df_cor is not None:
                aud_avail = [c for c in AUDITORY_COLS if c in df_cor.columns]
                if aud_avail:
                    aud_ts = smooth(zscore(df_cor[aud_avail].mean(axis=1).values))
                    ax.plot(t, aud_ts, color="#1565C0", lw=1.8, alpha=0.85,
                            label="Auditory", zorder=2)

            # NAcc mean
            nacc_ts = None
            if df_sub is not None:
                nacc_avail = [c for c in NACC_COLS if c in df_sub.columns]
                if nacc_avail:
                    nacc_ts = smooth(zscore(df_sub[nacc_avail].mean(axis=1).values))
                    ax.plot(t, nacc_ts, color="#C62828", lw=1.8, alpha=0.85,
                            label="NAcc", zorder=3)

            # Pearson r between them
            if aud_ts is not None and nacc_ts is not None:
                r_val = np.corrcoef(aud_ts, nacc_ts)[0, 1]
                ax.text(0.97, 0.97, f"r = {r_val:.3f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=9, fontweight="bold",
                        color="#B71C1C" if abs(r_val) > 0.3 else "#555",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

            ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.4)
            ax.set_xlabel("Time (TRs)", fontsize=7)

            if col_i == 0:
                ax.set_ylabel(sub, fontsize=9, fontweight="bold")
                ax.legend(fontsize=6, loc="upper left")
            if row_i == 0:
                ax.set_title(SESSION_LABELS[ses], fontsize=10,
                             fontweight="bold", color=color)

    out = os.path.join(OUT_DIR, f"overlay_nacc_auditory_{condition}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out)}")


def make_roi_correlation_matrix(condition: str = "glass") -> None:
    """
    For a quick sanity check: compute the full AAL correlation matrix for each subject × session
    and show whether global signal inflation is driving results.
    Shows: mean off-diagonal r (GCOR) and within vs between network comparison.
    """
    plt.rcParams.update(PLOT_STYLE)
    subjects = list(CONFIGS.keys())

    fig, axes = plt.subplots(
        len(subjects), len(SESSION_ORDER),
        figsize=(4 * len(SESSION_ORDER), 3.5 * len(subjects)),
        constrained_layout=True,
    )
    fig.suptitle(
        f"AAL Connectivity Matrix w/ GCOR ({condition.upper()})\n"
        "Global signal inflation = uniformly positive matrix",
        fontsize=10, fontweight="bold",
    )

    for row_i, sub in enumerate(subjects):
        cfg = CONFIGS[sub]
        for col_i, ses in enumerate(SESSION_ORDER):
            ax    = axes[row_i, col_i]
            df    = _load(cfg, condition, ses, "aal")
            color = SESSION_COLORS[ses]

            if df is None:
                ax.set_visible(False)
                continue

            c = np.corrcoef(df.values.T)
            np.fill_diagonal(c, 0)
            gcor = float(np.nanmean(c[~np.eye(c.shape[0], dtype=bool)]))

            im = ax.imshow(c, cmap="RdBu_r", vmin=-0.8, vmax=0.8,
                           aspect="auto", interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label="r", format="%.1f")

            title = SESSION_LABELS[ses] if row_i == 0 else ""
            ax.set_title((title + "\n" if title else "") +
                         f"GCOR={gcor:.3f}",
                         fontsize=9, fontweight="bold", color=color)
            if col_i == 0:
                ax.set_ylabel(sub, fontsize=9, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    out = os.path.join(OUT_DIR, f"connectivity_matrix_gcor_{condition}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out)}")


if __name__ == "__main__":
    print(f"Output -> {OUT_DIR}\n")
    for cond in ["glass", "rest"]:
        print(f"--- {cond.upper()} ---")
        make_timeseries_figure(cond)
        make_overlay_comparison(cond)
        make_roi_correlation_matrix(cond)
        print()
    print("Done.")
