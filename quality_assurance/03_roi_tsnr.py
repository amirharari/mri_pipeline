"""
Quality Assurance 3: ROI-based tSNR in DMN and Auditory Cortex.

tSNR = mean / std of raw BOLD signal per ROI across time.
Uses NiftiLabelsMasker (standardize=False) on one run at a time to keep RAM low.

ROI groups:
  DMN      -- Schaefer100 parcels with "Default" in name
  Auditory -- Harvard-Oxford cortical: "Planum Temporale", "Heschl's Gyrus",
              "Superior Temporal Gyrus, anterior/posterior division"

Saves: roi_tsnr_summary.csv + roi_tsnr_rest_vs_music.png
"""
import gc
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from scipy import stats

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "extraction"))
sys.path.insert(0, _REPO)
from fmri_timeseries_extractor import get_tr_seconds  # noqa: E402
from config import FMRIPREP_ROOT, QA_OUTPUT_DIR, ensure_nilearn_cache  # noqa: E402

ensure_nilearn_cache()
OUT_DIR = os.path.join(QA_OUTPUT_DIR, "roi_tsnr")
os.makedirs(OUT_DIR, exist_ok=True)


def load_atlases():
    """Load atlases once; return (dmn_spec, auditory_spec).

    Both use separate atlases:
      DMN      — Schaefer 100 parcels labelled "Default"
      Auditory — Harvard-Oxford cortical (Heschl's Gyrus, Planum Temporale,
                 Superior Temporal Gyrus anterior + posterior divisions)
    """
    # --- DMN: Schaefer 100 ---
    sch = datasets.fetch_atlas_schaefer_2018(n_rois=100)
    labels_sch = [l.decode() if isinstance(l, bytes) else l for l in sch.labels]
    dmn_indices = [i + 1 for i, l in enumerate(labels_sch) if "Default" in l]

    # --- Auditory: Harvard-Oxford cortical (no AAL dependency) ---
    ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    labels_ho = list(ho.labels)
    AUD_KEYWORDS = ("Heschl", "Planum Temporale", "Superior Temporal Gyrus")
    aud_indices = [i for i, l in enumerate(labels_ho) if any(k in l for k in AUD_KEYWORDS)]

    print(f"DMN parcels  (Schaefer-100)     : {len(dmn_indices)}")
    aud_names = [labels_ho[i] for i in aud_indices]
    print(f"Auditory ROIs (Harvard-Oxford)  : {len(aud_indices)}")
    for n in aud_names:
        print(f"  - {n}")
    return (sch.maps, dmn_indices), (ho.maps, aud_indices)


def compute_roi_tsnr(bold_path, atlas_maps, roi_indices):
    """Extract raw timeseries for ROI subset, return mean tSNR.

    Uses get_tr_seconds for reliable TR reading (header or JSON sidecar).
    """
    img = nib.load(bold_path)
    tr  = get_tr_seconds(bold_path, img)   # robust TR: header or JSON sidecar

    masker = NiftiLabelsMasker(
        labels_img = atlas_maps,
        standardize = False,
        detrend     = False,
        t_r         = tr,
        verbose     = 0,
    )
    ts = masker.fit_transform(img)       # (T, n_rois)
    del img                              # free NIfTI proxy
    gc.collect()

    all_labels = list(masker.labels_)

    # Select only target ROI columns
    col_mask = np.array([l in roi_indices for l in all_labels])
    if not col_mask.any():
        return np.nan

    ts_roi = ts[:, col_mask]             # (T, n_target)
    m = ts_roi.mean(axis=0)
    s = np.maximum(ts_roi.std(axis=0), 1e-6)
    return float((m / s).mean())


def find_bold_files():
    """Yield dict per BOLD run."""
    for subj in sorted(p for p in os.listdir(FMRIPREP_ROOT) if p.startswith("sub-")):
        subj_path = os.path.join(FMRIPREP_ROOT, subj)
        if not os.path.isdir(subj_path):
            continue
        for ses in sorted(p for p in os.listdir(subj_path) if p.startswith("ses-")):
            func_dir = os.path.join(subj_path, ses, "func")
            if not os.path.isdir(func_dir):
                continue
            for f in sorted(os.listdir(func_dir)):
                if not (f.endswith(".nii.gz") and "preproc_bold" in f and "MNI" in f):
                    continue
                task_m = re.search(r"task-([a-zA-Z0-9]+)", f)
                acq_m  = re.search(r"acq-([a-zA-Z0-9]+)", f)
                run_m  = re.search(r"run-([a-zA-Z0-9]+)", f)
                yield {
                    "bold_path": os.path.join(func_dir, f),
                    "subject": subj,
                    "session": ses,
                    "task": task_m.group(1) if task_m else "",
                    "acq":  acq_m.group(1)  if acq_m  else "",
                    "run":  f"run-{run_m.group(1)}" if run_m else "run-1",
                }


def main():
    print("Loading atlases...")
    (dmn_atlas, dmn_idx), (aud_atlas, aud_idx) = load_atlases()

    results = []
    for info in find_bold_files():
        acq_str = f"_{info['acq']}" if info["acq"] else ""
        label = f"{info['subject']}_{info['session']}_{info['task']}{acq_str}_{info['run']}"
        print(f"  {label}...", end=" ", flush=True)
        try:
            tsnr_dmn = compute_roi_tsnr(info["bold_path"], dmn_atlas, dmn_idx)
            gc.collect()
            tsnr_aud = compute_roi_tsnr(info["bold_path"], aud_atlas, aud_idx)
            gc.collect()
            print(f"DMN={tsnr_dmn:.1f}  Aud={tsnr_aud:.1f}")
        except Exception as e:
            print(f"ERROR: {e}")
            tsnr_dmn = tsnr_aud = np.nan

        cond = "rest" if info["task"] in ("rest", "restchecktr2") else "music"
        results.append({**{k: info[k] for k in ("subject","session","task","acq","run")},
                        "condition": cond,
                        "tSNR_DMN": round(tsnr_dmn, 2),
                        "tSNR_Auditory": round(tsnr_aud, 2) if not np.isnan(tsnr_aud) else np.nan})

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, "roi_tsnr_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # Bar plot
    rest  = df[df["condition"] == "rest"]
    music = df[df["condition"] == "music"]
    subj_colors = {"sub-001": "#2196F3", "sub-002": "#F44336",
                   "sub-003": "#4CAF50", "sub-004": "#FF9800"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (roi, title) in zip(axes, [("tSNR_DMN", "Default Mode Network"),
                                        ("tSNR_Auditory", "Auditory Cortex")]):
        r_vals = rest[roi].dropna()
        m_vals = music[roi].dropna()
        ax.bar([0, 1], [r_vals.mean(), m_vals.mean()],
               yerr=[r_vals.sem(), m_vals.sem()],
               capsize=6, color=["#607D8B", "#E91E63"], width=0.5, alpha=0.85)

        for subj in df["subject"].unique():
            for xi, cond in [(0, "rest"), (1, "music")]:
                vals = df[(df["subject"] == subj) & (df["condition"] == cond)][roi].dropna()
                ax.scatter([xi] * len(vals), vals, color=subj_colors.get(subj, "grey"),
                           zorder=5, s=40, alpha=0.8)

        if len(r_vals) >= 2 and len(m_vals) >= 2:
            t, p = stats.ttest_ind(r_vals, m_vals)
            ax.set_title(f"{title}\nRest={r_vals.mean():.1f} vs Music={m_vals.mean():.1f}"
                         f"  (t={t:.2f}, p={p:.3f})")
        else:
            ax.set_title(f"{title}")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Rest", "Music"])
        ax.set_ylabel("Mean tSNR")
        ax.grid(True, axis="y", alpha=0.3)

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                           markersize=8, label=s) for s, c in subj_colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, title="Subject")
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig_path = os.path.join(OUT_DIR, "roi_tsnr_rest_vs_music.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path}")

    print("\n-- Mean tSNR by subject --")
    print(df.groupby("subject")[["tSNR_DMN", "tSNR_Auditory"]].mean().round(1))
    print("\n-- Mean tSNR by condition --")
    print(df.groupby("condition")[["tSNR_DMN", "tSNR_Auditory"]].mean().round(1))


if __name__ == "__main__":
    main()
