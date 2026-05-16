"""
============================================================================
VISUAL BRAIN ILLUSTRATION: ENTROPY UNDER PSILOCYBIN VS BASELINE
============================================================================

Purpose
-------
Create striking brain visualizations comparing entropy by region between two
conditions (e.g., baseline vs psilocybin), suitable for social media (portrait
1080x1920). Uses AAL atlas and per-ROI time-series CSVs.

Inputs
------
- Two AAL time-series CSVs or folders containing AAL CSVs:
  - CONDITION_A_CSV or FOLDER (baseline)
  - CONDITION_B_CSV or FOLDER (psilocybin)

Processing
----------
- Compute per-ROI spectral entropy (normalized to [0,1]) from time series
- Aggregate per condition (mean across files if a folder)
- Map ROI values onto AAL atlas and render three panels:
  1) Baseline entropy
  2) Psilocybin entropy
  3) Difference (Psilo − Base)

Outputs
-------
- High-resolution portrait PNG (1080x1920) with side-by-side brain maps
- Individual panel PNGs (optional)

Notes
-----
- Designed for clear, vibrant visuals (social-first color, fonts, titles)
- Assumes CSV columns match AAL labels (e.g., 'Precentral_L', 'Amygdala_R')
"""

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import datasets, plotting, image
import nibabel as nib
from scipy.signal import welch


# =============================================================================
# CONFIG
# =============================================================================

# Provide EITHER file paths OR directories (script will detect which)
CONDITION_A_PATH = r"C:\kpeSoundPath\rest-results-itamar\sub-001_ses-1_task-rest_aal_ts.csv"  # Baseline
CONDITION_B_PATH = r"C:\kpeSoundPath\rest-results-itamar\sub-001_ses-3_task-rest_aal_ts.csv"  # Psilocybin

OUTPUT_DIR = "psilocybin_brain_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Visual style
FIG_WIDTH_PX, FIG_HEIGHT_PX = 1080, 1920  # portrait for TikTok
DPI = 150
FIG_SIZE_IN = (FIG_WIDTH_PX / DPI, FIG_HEIGHT_PX / DPI)

BASELINE_TITLE = "Baseline"
PSILO_TITLE = "Psilocybin"
DIFF_TITLE = "Δ Entropy (Psy − Base)"

# Entropy parameters
TR_SEC = 1.0               # TR in seconds
SPECTRAL_ENTROPY_NORMALIZE = True  # Normalize to [0,1]

# Colormaps
CMAP_ENTROPY = "plasma"
CMAP_DIFF = "coolwarm"


# =============================================================================
# UTILITIES
# =============================================================================

def spectral_entropy(time_series: np.ndarray, fs: float, normalize: bool = True) -> float:
    """Compute spectral entropy of a 1D time series using Welch's method.

    Returns:
        float: entropy value (0..1 if normalize)
    """
    if time_series is None or len(time_series) < 10:
        return np.nan
    freqs, psd = welch(time_series, fs=fs, nperseg=min(256, max(16, len(time_series) // 4)))
    psd = np.asarray(psd)
    psd = psd / np.sum(psd) if np.sum(psd) > 0 else psd
    psd = psd[psd > 0]
    if psd.size == 0:
        return np.nan
    H = -np.sum(psd * np.log2(psd))
    if normalize and psd.size > 1:
        H /= np.log2(psd.size)
    return float(H)


def load_aal_entropy_from_csv(csv_path: str, fs: float) -> Dict[str, float]:
    """Load an AAL timeseries CSV and compute per-ROI spectral entropy.

    Keeps numeric columns only and ignores non-numeric identifiers.
    """
    df = pd.read_csv(csv_path)
    df = df.select_dtypes(include=[np.number])
    # Drop constant/all-NaN columns
    keep_cols = [c for c in df.columns if df[c].std(ddof=0) > 0 and not df[c].isna().all()]
    df = df[keep_cols]
    roi_to_entropy: Dict[str, float] = {}
    for col in df.columns:
        ts = df[col].dropna().values
        roi_to_entropy[col] = spectral_entropy(ts, fs=fs, normalize=SPECTRAL_ENTROPY_NORMALIZE)
    return roi_to_entropy


def aggregate_entropy(path: str, fs: float) -> Dict[str, float]:
    """Aggregate per-ROI entropy across a single file or all CSVs in a folder."""
    if os.path.isdir(path):
        entropies: List[Dict[str, float]] = []
        for fname in sorted(os.listdir(path)):
            if fname.endswith("_aal_ts.csv"):
                entropies.append(load_aal_entropy_from_csv(os.path.join(path, fname), fs))
        if not entropies:
            raise FileNotFoundError(f"No _aal_ts.csv files found in folder: {path}")
        # Compute mean across files for overlapping ROIs
        keys = set().union(*[e.keys() for e in entropies])
        return {k: np.nanmean([e.get(k, np.nan) for e in entropies]) for k in keys}
    else:
        return load_aal_entropy_from_csv(path, fs)


def build_label_to_index(label_list: List[bytes], index_list: List[int]) -> Dict[str, int]:
    """Map AAL label strings to voxel values in the atlas image using provided indices."""
    mapping: Dict[str, int] = {}
    for i, raw in enumerate(label_list):
        label = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        mapping[label] = int(index_list[i])
    return mapping


def project_values_to_atlas(values_by_label: Dict[str, float], atlas_img: nib.Nifti1Image,
                            label_to_index: Dict[str, int]) -> nib.Nifti1Image:
    """Create a NIfTI image with voxel values assigned per labeled region."""
    atlas_data = atlas_img.get_fdata()
    out_data = np.zeros_like(atlas_data)

    # AAL atlas is integer-labeled; fill regions by label index
    for label, val in values_by_label.items():
        idx = label_to_index.get(label)
        if idx is None:
            continue
        out_data[atlas_data == idx] = 0.0 if val is None or np.isnan(val) else float(val)

    return image.new_img_like(atlas_img, out_data)


def align_to_aal_labels(values: Dict[str, float], aal_labels: List[bytes]) -> Dict[str, float]:
    """Best-effort alignment of CSV ROI names to AAL label strings."""
    # Most CSVs from AAL pipelines already match names like 'Precentral_L'
    available = {k: v for k, v in values.items() if isinstance(k, str)}
    return available


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("PSILOCYBIN VS BASELINE: BRAIN ENTROPY VISUALIZATION (AAL)")
    print("=" * 80)
    print(f"Condition A: {CONDITION_A_PATH}")
    print(f"Condition B: {CONDITION_B_PATH}")
    print(f"Output dir : {OUTPUT_DIR}")

    # Load AAL atlas
    print("Loading AAL atlas...")
    # Use SPM12 variant which provides explicit indices mapping label->voxel value
    aal = datasets.fetch_atlas_aal(version="SPM12")
    aal_img = nib.load(aal["maps"])
    aal_labels = aal["labels"]  # list of bytes/str
    aal_indices = list(aal["indices"]) if "indices" in aal else sorted(int(v) for v in np.unique(aal_img.get_fdata()) if v > 0)[:len(aal_labels)]
    label_to_index = build_label_to_index(aal_labels, aal_indices)

    # Aggregate per-ROI entropy for each condition
    fs = 1.0 / TR_SEC
    ent_A_raw = aggregate_entropy(CONDITION_A_PATH, fs)
    ent_B_raw = aggregate_entropy(CONDITION_B_PATH, fs)

    # Align to AAL label names
    ent_A = align_to_aal_labels(ent_A_raw, aal_labels)
    ent_B = align_to_aal_labels(ent_B_raw, aal_labels)

    # Build images
    img_A = project_values_to_atlas(ent_A, aal_img, label_to_index)
    img_B = project_values_to_atlas(ent_B, aal_img, label_to_index)

    # Compute difference (B - A)
    data_A = img_A.get_fdata()
    data_B = img_B.get_fdata()
    data_D = data_B - data_A
    img_D = image.new_img_like(aal_img, data_D)

    # Normalize display ranges (robust percentiles for nice visuals)
    def robust_vmin_vmax(data: np.ndarray, lo=5, hi=95):
        vals = data[np.isfinite(data)]
        if vals.size == 0:
            return 0.0, 1.0
        return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))

    vmin_A, vmax_A = robust_vmin_vmax(data_A)
    vmin_B, vmax_B = robust_vmin_vmax(data_B)
    vmax_abs_D = np.max(np.abs(data_D[np.isfinite(data_D)])) if np.isfinite(data_D).any() else 1.0

    # Create portrait figure (1080x1920)
    fig = plt.figure(figsize=FIG_SIZE_IN, dpi=DPI)
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])

    # Panel 1: Baseline
    ax1 = fig.add_subplot(gs[0])
    plotting.plot_stat_map(
        img_A,
        display_mode="ortho",
        cut_coords=(0, 0, 0),
        cmap=CMAP_ENTROPY,
        vmin=vmin_A, vmax=vmax_A,
        annotate=False,
        colorbar=True,
        axes=ax1,
        title=BASELINE_TITLE,
    )

    # Panel 2: Psilocybin
    ax2 = fig.add_subplot(gs[1])
    plotting.plot_stat_map(
        img_B,
        display_mode="ortho",
        cut_coords=(0, 0, 0),
        cmap=CMAP_ENTROPY,
        vmin=vmin_B, vmax=vmax_B,
        annotate=False,
        colorbar=True,
        axes=ax2,
        title=PSILO_TITLE,
    )

    # Panel 3: Difference
    ax3 = fig.add_subplot(gs[2])
    plotting.plot_stat_map(
        img_D,
        display_mode="ortho",
        cut_coords=(0, 0, 0),
        cmap=CMAP_DIFF,
        vmin=-vmax_abs_D, vmax=vmax_abs_D,
        annotate=False,
        colorbar=True,
        axes=ax3,
        title=DIFF_TITLE,
    )

    fig.suptitle("Regional Spectral Entropy (AAL)", y=0.995, fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = os.path.join(OUTPUT_DIR, "brain_entropy_psilo_vs_base_portrait.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Optional: also save individual panels as separate images
    dispA = plotting.plot_glass_brain(img_A, cmap=CMAP_ENTROPY, display_mode='lyrz', colorbar=True,
                                      vmin=vmin_A, vmax=vmax_A, title=BASELINE_TITLE)
    dispA.savefig(os.path.join(OUTPUT_DIR, "baseline_glass.png"), dpi=200)
    dispB = plotting.plot_glass_brain(img_B, cmap=CMAP_ENTROPY, display_mode='lyrz', colorbar=True,
                                      vmin=vmin_B, vmax=vmax_B, title=PSILO_TITLE)
    dispB.savefig(os.path.join(OUTPUT_DIR, "psilocybin_glass.png"), dpi=200)
    dispD = plotting.plot_glass_brain(img_D, cmap=CMAP_DIFF, display_mode='lyrz', colorbar=True,
                                      vmin=-vmax_abs_D, vmax=vmax_abs_D, title=DIFF_TITLE)
    dispD.savefig(os.path.join(OUTPUT_DIR, "difference_glass.png"), dpi=200)

    print("All visuals generated.")


if __name__ == "__main__":
    main()


