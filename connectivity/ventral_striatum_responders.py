"""ventral_striatum_responders.py

NAcc (nucleus accumbens) seed-based connectivity analysis
focused on the high vs low responder comparison.

Study context
-------------
sub-002, sub-004 : high psilocybin responders (strong valence + arousal during music)
sub-003          : lower responder (weaker subjective experience)

Hypothesis: Under psilocybin + music, high responders show larger increases in
  NAcc → Auditory cortex      (music reward pathway)
  NAcc → Insular Cortex       (interoceptive / bodily salience)
  NAcc → Frontal Medial / ACC (personal meaning / self-referential)
  NAcc → Amygdala             (emotional arousal)

Figures
-------
  figVS1_nacc_full_profile.png       — full HO-cortical NAcc connectivity bar chart × subject × session
  figVS2_targeted_circuits.png       — 4 key circuits × 3 sessions, high vs low responder
  figVS3_delta_psilo_baseline.png    — Δ(Psilo-Baseline) per circuit × condition × subject
  figVS4_nacc_temporal_dynamics.png  — smoothed NAcc timeseries + sliding-window NAcc-Auditory r

Reuses: conn_matrix, fisher_z_matrix, PLOT_STYLE, SESSION_LABELS/COLORS from psilo_session_comparison
"""

from __future__ import annotations

import os, sys, warnings
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines
import numpy  as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore", category=RuntimeWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from psilo_session_comparison import conn_matrix, PLOT_STYLE
from subject_configs import CONFIGS

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
_REPO   = os.path.dirname(_HERE)
OUT_DIR = os.path.join(_REPO, "outputs", "nacc_responders")
os.makedirs(OUT_DIR, exist_ok=True)

SESSION_ORDER  = ["ses-1", "ses-2", "ses-3"]
SESSION_LABELS = {"ses-1": "Baseline", "ses-2": "Psilocybin", "ses-3": "Follow-up"}
SESSION_COLORS = {"ses-1": "#2196F3",  "ses-2": "#F44336",    "ses-3": "#4CAF50"}

RESPONDER_GROUP = {"sub-002": "High",  "sub-003": "Low",  "sub-004": "High"}
RESPONDER_COLOR = {"High": "#E53935",  "Low": "#1E88E5"}

SUB_MARKERS = {"sub-002": "o", "sub-003": "s", "sub-004": "^"}
SUB_LABELS  = {"sub-002": "sub-002 (High)", "sub-003": "sub-003 (Low)", "sub-004": "sub-004 (High)"}

# ── NAcc seed ──────────────────────────────────────────────────────────────────
NACC_COLS = ["Left Accumbens", "Right Accumbens"]      # bilateral average

# ── Target circuits (HO cortical) ─────────────────────────────────────────────
CIRCUITS: Dict[str, List[str]] = {
    "Auditory\nCortex": [
        "Heschl's Gyrus (includes H1 and H2)",
        "Superior Temporal Gyrus, anterior division",
        "Superior Temporal Gyrus, posterior division",
        "Planum Temporale",
    ],
    "Insula": [
        "Insular Cortex",
    ],
    "Medial PFC\n(self-referential)": [
        "Frontal Medial Cortex",
        "Paracingulate Gyrus",
        "Cingulate Gyrus, anterior division",
    ],
    "Posterior\nCingulate (DMN)": [
        "Cingulate Gyrus, posterior division",
        "Precuneous Cortex",
    ],
}

# Amygdala is in HO-subcortical — handled separately
AMYGDALA_COLS = ["Left Amygdala", "Right Amygdala"]

# ── Conditions used in figures ──────────────────────────────────────────────────
# glass = available all subjects × all sessions
CONDITIONS_SHOW = {
    "rest":  "Rest",
    "glass": "Glass (music)",
}

SUBJECTS = list(CONFIGS.keys())  # ["sub-002", "sub-003", "sub-004"]

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load(cfg: dict, condition: str, session: str, atlas: str) -> Optional[pd.DataFrame]:
    prefix = cfg["conditions"].get(condition, {}).get(session)
    if prefix is None:
        return None
    p = os.path.join(cfg["ts_dir"], f"{prefix}_{atlas}_ts.csv")
    return pd.read_csv(p) if os.path.isfile(p) else None


def nacc_seed_r(
    df_sub: pd.DataFrame,          # HO subcortical timeseries
    df_cort: pd.DataFrame,         # HO cortical timeseries
    targets: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    For every column in df_cort (or a subset), return Pearson r with bilateral NAcc.
    NAcc = row-wise mean of Left+Right Accumbens (both must exist in df_sub).
    """
    avail = [c for c in NACC_COLS if c in df_sub.columns]
    if len(avail) < 1:
        return {}
    nacc = df_sub[avail].mean(axis=1).values

    cols = targets if targets is not None else list(df_cort.columns)
    out  = {}
    for col in cols:
        if col in df_cort.columns:
            r = np.corrcoef(nacc, df_cort[col].values)[0, 1]
            out[col] = float(np.clip(r, -1, 1))
    return out


def circuit_mean_r(
    df_sub: pd.DataFrame,
    df_cort: pd.DataFrame,
    circuit_cols: List[str],
) -> float:
    """Fisher-z mean of NAcc-r across all regions in a circuit."""
    r_dict = nacc_seed_r(df_sub, df_cort, circuit_cols)
    vals   = [r for r in r_dict.values() if not np.isnan(r)]
    if not vals:
        return float("nan")
    z_vals = np.arctanh(np.clip(vals, -0.999, 0.999))
    return float(np.tanh(np.mean(z_vals)))


def amygdala_nacc_r(df_sub: pd.DataFrame) -> float:
    """Pearson r between bilateral NAcc and bilateral Amygdala."""
    nacc_a = [c for c in NACC_COLS if c in df_sub.columns]
    amyg_a = [c for c in AMYGDALA_COLS if c in df_sub.columns]
    if not nacc_a or not amyg_a:
        return float("nan")
    nacc = df_sub[nacc_a].mean(axis=1).values
    amyg = df_sub[amyg_a].mean(axis=1).values
    r    = np.corrcoef(nacc, amyg)[0, 1]
    return float(np.clip(r, -1, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Figure VS1 — Full HO-cortical NAcc connectivity profile
# (horizontal bar chart: all 48 HO regions, sorted by Psilocybin value)
# 3 rows = 3 subjects, 3 session lines overlaid
# ─────────────────────────────────────────────────────────────────────────────

def figVS1_nacc_full_profile() -> None:
    plt.rcParams.update(PLOT_STYLE)
    condition = "glass"

    fig, axes = plt.subplots(1, len(SUBJECTS),
                             figsize=(7 * len(SUBJECTS), 14),
                             constrained_layout=True)
    fig.suptitle("NAcc Seed Connectivity — Full Cortical Profile (Glass music)\n"
                 "Sorted by Psilocybin session r-value",
                 fontsize=13, fontweight="bold")

    for col_i, sub in enumerate(SUBJECTS):
        ax  = axes[col_i]
        cfg = CONFIGS[sub]

        # Collect r-values across sessions
        session_r: Dict[str, Dict[str, float]] = {}
        col_order = None
        for ses in SESSION_ORDER:
            df_s = _load(cfg, condition, ses, "ho_subcortical")
            df_c = _load(cfg, condition, ses, "ho_cortical")
            if df_s is None or df_c is None:
                continue
            r_dict = nacc_seed_r(df_s, df_c)
            session_r[ses] = r_dict

        if not session_r:
            ax.set_visible(False)
            continue

        # Sort regions by psilocybin (ses-2) value; fall back to ses-1
        ref_ses = "ses-2" if "ses-2" in session_r else list(session_r.keys())[0]
        col_order = sorted(session_r[ref_ses].keys(),
                           key=lambda c: session_r[ref_ses][c])

        y_pos = np.arange(len(col_order))
        height = 0.25
        offsets = {"ses-1": -height, "ses-2": 0, "ses-3": height}

        for ses in SESSION_ORDER:
            if ses not in session_r:
                continue
            vals = [session_r[ses].get(c, float("nan")) for c in col_order]
            ax.barh(y_pos + offsets[ses], vals, height=height * 0.85,
                    color=SESSION_COLORS[ses], alpha=0.80,
                    label=SESSION_LABELS[ses], edgecolor="white", linewidth=0.3)

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(col_order, fontsize=6.5)
        ax.set_xlabel("r (NAcc → region)", fontsize=9)
        rgroup = RESPONDER_GROUP[sub]
        ax.set_title(f"{sub}  [{rgroup} responder]",
                     fontsize=10, fontweight="bold",
                     color=RESPONDER_COLOR[rgroup])
        if col_i == 0:
            ax.legend(fontsize=8, loc="lower right")

    out = os.path.join(OUT_DIR, "figVS1_nacc_full_profile.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figVS1 -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure VS2 — Targeted circuits × High vs Low responders
# Left panels: REST condition   Right panels: GLASS condition
# Rows = circuits (Auditory, Insula, mPFC, PCC, Amygdala)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_circuit_vals(condition: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns {circuit_name: {sub_id: {ses: r}}}
    """
    all_circuits = dict(CIRCUITS)
    all_circuits["Amygdala\n(emotional)"] = []   # handled separately

    out: Dict[str, Dict[str, Dict[str, float]]] = {c: {} for c in all_circuits}

    for sub in SUBJECTS:
        cfg = CONFIGS[sub]
        for cname in all_circuits:
            out[cname][sub] = {}
        for ses in SESSION_ORDER:
            df_s = _load(cfg, condition, ses, "ho_subcortical")
            df_c = _load(cfg, condition, ses, "ho_cortical")
            if df_s is None:
                continue
            for cname, cols in CIRCUITS.items():
                if df_c is None:
                    continue
                r = circuit_mean_r(df_s, df_c, cols)
                out[cname][sub][ses] = r
            # Amygdala: both in subcortical
            out["Amygdala\n(emotional)"][sub][ses] = amygdala_nacc_r(df_s)
    return out


def figVS2_targeted_circuits() -> None:
    plt.rcParams.update(PLOT_STYLE)

    circuit_names = list(CIRCUITS.keys()) + ["Amygdala\n(emotional)"]
    n_circuits    = len(circuit_names)
    n_cond        = len(CONDITIONS_SHOW)

    fig, axes = plt.subplots(n_circuits, n_cond,
                             figsize=(4.5 * n_cond, 3.5 * n_circuits),
                             constrained_layout=True)
    fig.suptitle("NAcc Seed Connectivity — Key Circuits\n"
                 "High responders: sub-002, sub-004  |  Low responder: sub-003",
                 fontsize=12, fontweight="bold")

    for col_i, (cond_key, cond_label) in enumerate(CONDITIONS_SHOW.items()):
        vals = _collect_circuit_vals(cond_key)

        for row_i, cname in enumerate(circuit_names):
            ax = axes[row_i, col_i]

            # Draw individual subject lines
            for sub in SUBJECTS:
                sub_dict = vals[cname][sub]
                xs = [i for i, ses in enumerate(SESSION_ORDER) if ses in sub_dict and not np.isnan(sub_dict[ses])]
                ys = [sub_dict[ses] for ses in SESSION_ORDER if ses in sub_dict and not np.isnan(sub_dict[ses])]
                rgroup = RESPONDER_GROUP[sub]
                ax.plot(xs, ys,
                        color=RESPONDER_COLOR[rgroup],
                        marker=SUB_MARKERS[sub], markersize=9,
                        linewidth=1.8, alpha=0.75,
                        linestyle="-" if rgroup == "High" else "--",
                        label=SUB_LABELS[sub], zorder=3)

            # Shade group means
            for gi, group in enumerate(["High", "Low"]):
                group_subs = [s for s in SUBJECTS if RESPONDER_GROUP[s] == group]
                for xi, ses in enumerate(SESSION_ORDER):
                    g_vals = [vals[cname][s].get(ses, float("nan"))
                              for s in group_subs]
                    g_clean = [v for v in g_vals if not np.isnan(v)]
                    if g_clean:
                        ax.scatter(xi, np.mean(g_clean),
                                   color=RESPONDER_COLOR[group],
                                   s=120, zorder=5, marker="_",
                                   linewidths=3)

            ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
            ax.axvspan(0.5, 1.5, color="#F44336", alpha=0.07, zorder=0)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(
                [SESSION_LABELS[s] for s in SESSION_ORDER],
                fontsize=8, rotation=20, ha="right"
            )
            ax.set_ylabel("NAcc r" if col_i == 0 else "", fontsize=8)
            if row_i == 0:
                ax.set_title(cond_label, fontsize=10, fontweight="bold")
            if col_i == 0:
                ax.text(-0.32, 0.5, cname, transform=ax.transAxes,
                        fontsize=8, fontweight="bold", va="center", ha="right",
                        rotation=0)

            if row_i == 0 and col_i == 0:
                handles = [
                    mlines.Line2D([], [], color=RESPONDER_COLOR["High"], linewidth=2,
                                  marker="o", markersize=7, label="High responder"),
                    mlines.Line2D([], [], color=RESPONDER_COLOR["Low"], linewidth=2,
                                  linestyle="--", marker="s", markersize=7, label="Low responder"),
                ] + [
                    mlines.Line2D([], [], color="gray", marker=SUB_MARKERS[s],
                                  markersize=6, linewidth=0, label=SUB_LABELS[s])
                    for s in SUBJECTS
                ]
                ax.legend(handles=handles, fontsize=7, loc="best")

    out = os.path.join(OUT_DIR, "figVS2_targeted_circuits.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figVS2 -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure VS3 — Δ(Psilocybin − Baseline) per circuit
# Bar chart: each subject as a bar, grouped by circuit, REST vs GLASS side by side
# This directly answers: "who changes, in which circuit, for which condition?"
# ─────────────────────────────────────────────────────────────────────────────

def figVS3_delta_map() -> None:
    plt.rcParams.update(PLOT_STYLE)

    circuit_names = list(CIRCUITS.keys()) + ["Amygdala\n(emotional)"]
    n_circuits    = len(circuit_names)

    fig, axes = plt.subplots(1, n_circuits,
                             figsize=(3.2 * n_circuits, 5),
                             constrained_layout=True)
    fig.suptitle("Psilocybin − Baseline  NAcc Connectivity Change\n"
                 "Δr per circuit per subject  |  REST vs GLASS (music)",
                 fontsize=12, fontweight="bold")

    bar_width = 0.3
    cond_offsets = {"rest": -bar_width / 2 - 0.05, "glass": bar_width / 2 + 0.05}
    cond_hatches = {"rest": "", "glass": "//"}

    for ci, cname in enumerate(circuit_names):
        ax = axes[ci]

        for sub_i, sub in enumerate(SUBJECTS):
            cfg   = CONFIGS[sub]
            rgrp  = RESPONDER_GROUP[sub]
            color = RESPONDER_COLOR[rgrp]

            for cond_key in CONDITIONS_SHOW:
                cond_vals = _collect_circuit_vals(cond_key)
                r_base  = cond_vals[cname][sub].get("ses-1", float("nan"))
                r_psilo = cond_vals[cname][sub].get("ses-2", float("nan"))
                delta   = r_psilo - r_base if not (np.isnan(r_base) or np.isnan(r_psilo)) else float("nan")

                x = sub_i + cond_offsets[cond_key]
                if not np.isnan(delta):
                    ax.bar(x, delta, width=bar_width * 0.85,
                           color=color, alpha=0.80 if cond_key == "glass" else 0.45,
                           hatch=cond_hatches[cond_key],
                           edgecolor="black", linewidth=0.5)
                    ax.text(x, delta + (0.005 if delta >= 0 else -0.015),
                            f"{delta:+.2f}", ha="center", va="bottom" if delta >= 0 else "top",
                            fontsize=6.5)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(SUBJECTS)))
        ax.set_xticklabels([s.replace("sub-", "s") for s in SUBJECTS],
                           fontsize=8)
        ax.set_title(cname, fontsize=9, fontweight="bold")
        if ci == 0:
            ax.set_ylabel("Δr (Psilo − Baseline)", fontsize=9)

        # shade psilocybin region (just label marker)
        ax.annotate("Psilo+Music\n(solid fill)" if ci == 0 else "",
                    xy=(0, 0), fontsize=6, alpha=0)

    # Legend
    legend_els = [
        mpatches.Patch(facecolor=RESPONDER_COLOR["High"], alpha=0.45,
                       edgecolor="black", label="High responder — REST"),
        mpatches.Patch(facecolor=RESPONDER_COLOR["High"], alpha=0.80,
                       hatch="//", edgecolor="black", label="High responder — GLASS"),
        mpatches.Patch(facecolor=RESPONDER_COLOR["Low"], alpha=0.45,
                       edgecolor="black", label="Low responder — REST"),
        mpatches.Patch(facecolor=RESPONDER_COLOR["Low"], alpha=0.80,
                       hatch="//", edgecolor="black", label="Low responder — GLASS"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=2,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.08))

    out = os.path.join(OUT_DIR, "figVS3_delta_psilo_baseline.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figVS3 -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure VS4 — NAcc temporal dynamics during Glass music
# Top row:   smoothed NAcc timeseries per session (z-scored, each subject)
# Bottom row: sliding-window NAcc ↔ Auditory r (window = 30 TRs)
# ─────────────────────────────────────────────────────────────────────────────

def figVS4_temporal_dynamics(window_trs: int = 30, smooth_trs: int = 15) -> None:
    plt.rcParams.update(PLOT_STYLE)
    condition = "glass"
    AUDITORY  = "Heschl's Gyrus (includes H1 and H2)"

    fig, axes = plt.subplots(2, len(SUBJECTS),
                             figsize=(5.5 * len(SUBJECTS), 8),
                             constrained_layout=True)
    fig.suptitle("NAcc Temporal Dynamics During Glass (music)\n"
                 "Top: smoothed bilateral NAcc signal  |  "
                 f"Bottom: sliding-window NAcc–Auditory r  (w={window_trs} TRs)",
                 fontsize=11, fontweight="bold")

    for col_i, sub in enumerate(SUBJECTS):
        cfg    = CONFIGS[sub]
        rgrp   = RESPONDER_GROUP[sub]
        ax_ts  = axes[0, col_i]
        ax_sw  = axes[1, col_i]

        for ses in SESSION_ORDER:
            df_s = _load(cfg, condition, ses, "ho_subcortical")
            df_c = _load(cfg, condition, ses, "ho_cortical")
            if df_s is None:
                continue

            avail_nacc = [c for c in NACC_COLS if c in df_s.columns]
            if not avail_nacc:
                continue

            nacc_raw = df_s[avail_nacc].mean(axis=1).values
            # z-score
            nacc_z   = (nacc_raw - nacc_raw.mean()) / (nacc_raw.std() + 1e-8)
            # smooth
            nacc_sm  = uniform_filter1d(nacc_z, size=smooth_trs)

            t = np.arange(len(nacc_sm))
            color = SESSION_COLORS[ses]
            label = SESSION_LABELS[ses]

            ax_ts.plot(t, nacc_sm, color=color, linewidth=1.2,
                       alpha=0.85, label=label)

            # Sliding-window NAcc–Auditory correlation
            if df_c is not None and AUDITORY in df_c.columns:
                aud = df_c[AUDITORY].values
                n   = len(nacc_raw)
                sw_r  = np.full(n, float("nan"))
                half  = window_trs // 2
                for t_i in range(half, n - half):
                    w_nacc = nacc_raw[t_i - half: t_i + half]
                    w_aud  = aud[t_i - half: t_i + half]
                    if np.std(w_nacc) > 1e-8 and np.std(w_aud) > 1e-8:
                        sw_r[t_i] = np.corrcoef(w_nacc, w_aud)[0, 1]
                sw_sm = uniform_filter1d(np.nan_to_num(sw_r), size=smooth_trs // 2)
                ax_sw.plot(t, sw_sm, color=color, linewidth=1.2, alpha=0.85)

        ax_ts.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
        ax_sw.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)

        for ax in [ax_ts, ax_sw]:
            ax.set_xlabel("Time (TRs)", fontsize=8)

        ax_ts.set_ylabel("NAcc signal (z-score, smoothed)", fontsize=8)
        ax_sw.set_ylabel("NAcc–Auditory r\n(sliding window)", fontsize=8)

        ax_ts.set_title(
            f"{sub}  [{rgrp} responder]",
            fontsize=10, fontweight="bold",
            color=RESPONDER_COLOR[rgrp]
        )
        if col_i == 0:
            ax_ts.legend(fontsize=8, loc="upper right")

    out = os.path.join(OUT_DIR, "figVS4_nacc_temporal_dynamics.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figVS4 -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure VS5 — Summary scatter: responder group vs NAcc connectivity change
# x-axis: Δ NAcc–Auditory (music psilocybin − baseline)
# y-axis: Δ NAcc–Insula
# Each dot = one subject; color = responder group
# Quadrants annotated with interpretation
# ─────────────────────────────────────────────────────────────────────────────

def figVS5_responder_scatter() -> None:
    plt.rcParams.update(PLOT_STYLE)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    fig.suptitle("Responder Profile — NAcc Connectivity Change Under Psilocybin\n"
                 "Δr = Psilocybin minus Baseline",
                 fontsize=12, fontweight="bold")

    auditory_cols = CIRCUITS["Auditory\nCortex"]
    insula_cols   = CIRCUITS["Insula"]
    mpfc_cols     = CIRCUITS["Medial PFC\n(self-referential)"]

    for ax_i, (cond_key, cond_label) in enumerate(CONDITIONS_SHOW.items()):
        ax = axes[ax_i]

        for sub in SUBJECTS:
            cfg   = CONFIGS[sub]
            rgrp  = RESPONDER_GROUP[sub]

            vals = {}
            for ses_key in ["ses-1", "ses-2"]:
                df_s = _load(cfg, cond_key, ses_key, "ho_subcortical")
                df_c = _load(cfg, cond_key, ses_key, "ho_cortical")
                if df_s is None or df_c is None:
                    vals[ses_key] = None
                    continue
                vals[ses_key] = {
                    "aud":  circuit_mean_r(df_s, df_c, auditory_cols),
                    "ins":  circuit_mean_r(df_s, df_c, insula_cols),
                    "mpfc": circuit_mean_r(df_s, df_c, mpfc_cols),
                    "amyg": amygdala_nacc_r(df_s),
                }

            if vals.get("ses-1") is None or vals.get("ses-2") is None:
                continue

            delta_aud  = vals["ses-2"]["aud"]  - vals["ses-1"]["aud"]
            delta_ins  = vals["ses-2"]["ins"]  - vals["ses-1"]["ins"]
            delta_mpfc = vals["ses-2"]["mpfc"] - vals["ses-1"]["mpfc"]
            delta_amyg = vals["ses-2"]["amyg"] - vals["ses-1"]["amyg"]

            # Bubble: size = magnitude of mPFC change, color = responder
            bubble_size = max(abs(delta_mpfc) * 3000, 200)
            ax.scatter(delta_aud, delta_ins,
                       color=RESPONDER_COLOR[rgrp],
                       s=bubble_size, alpha=0.75,
                       marker=SUB_MARKERS[sub],
                       edgecolors="black", linewidths=1.2, zorder=3)
            ax.annotate(
                f"{sub}\namyg Δ={delta_amyg:+.2f}",
                (delta_aud + 0.003, delta_ins + 0.003),
                fontsize=7, color="black"
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

        # Quadrant labels
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        ax.text(0.97, 0.97, "music reward\n+ body salience ↑",
                transform=ax.transAxes, fontsize=7, ha="right", va="top",
                color="#B71C1C", alpha=0.6)
        ax.text(0.03, 0.97, "body ↑,\nreward ↓",
                transform=ax.transAxes, fontsize=7, ha="left", va="top",
                color="#555", alpha=0.6)
        ax.text(0.97, 0.03, "reward ↑,\nbody ↓",
                transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                color="#555", alpha=0.6)
        ax.text(0.03, 0.03, "both ↓",
                transform=ax.transAxes, fontsize=7, ha="left", va="bottom",
                color="#555", alpha=0.6)

        ax.set_xlabel("Δ NAcc–Auditory r  (reward × music pathway)", fontsize=9)
        ax.set_ylabel("Δ NAcc–Insula r  (interoceptive salience)", fontsize=9)
        ax.set_title(cond_label, fontsize=10, fontweight="bold")

    legend_els = [
        mpatches.Patch(facecolor=RESPONDER_COLOR["High"], label="High responder",
                       edgecolor="black"),
        mpatches.Patch(facecolor=RESPONDER_COLOR["Low"],  label="Low responder",
                       edgecolor="black"),
        mlines.Line2D([], [], color="none", marker="o", markersize=7,
                      label="Bubble size ∝ |Δ NAcc–mPFC|", markerfacecolor="gray"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, -0.06))

    out = os.path.join(OUT_DIR, "figVS5_responder_scatter.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figVS5 -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"NAcc responder analysis  ->  {OUT_DIR}\n")
    print(f"Responder groups: {RESPONDER_GROUP}\n")
    figVS1_nacc_full_profile()
    figVS2_targeted_circuits()
    figVS3_delta_map()
    figVS4_temporal_dynamics()
    figVS5_responder_scatter()
    print(f"\nAll figures -> {OUT_DIR}")


if __name__ == "__main__":
    main()
