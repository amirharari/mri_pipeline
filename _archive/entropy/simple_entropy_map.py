"""
Simple per-ROI entropy brain map (AAL, volumetric)

- Computes spectral entropy per AAL ROI from one CSV
- Projects to AAL atlas volume and renders a single map
- No surfaces; avoids null patches seen on surface sampling
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, plotting, image
from scipy.signal import welch


CSV_PATH = r"C:\kpeSoundPath\rest-results-itamar\sub-001_ses-1_task-rest_aal_ts.csv"
TR_SEC = 1.0
OUTPUT_PNG = "simple_entropy_map.png"


def spectral_entropy(ts: np.ndarray, fs: float, normalize: bool = True) -> float:
    if ts is None or len(ts) < 10:
        return np.nan
    freqs, psd = welch(ts, fs=fs, nperseg=min(256, max(16, len(ts)//4)))
    s = np.sum(psd)
    if s <= 0:
        return np.nan
    psd = psd / s
    psd = psd[psd > 0]
    if psd.size == 0:
        return np.nan
    H = -np.sum(psd * np.log2(psd))
    if normalize and psd.size > 1:
        H /= np.log2(psd.size)
    return float(H)


def main():
    print("Computing simple AAL entropy map...")
    df = pd.read_csv(CSV_PATH).select_dtypes(include=[np.number])
    keep = [c for c in df.columns if df[c].std(ddof=0) > 0 and not df[c].isna().all()]
    df = df[keep]

    fs = 1.0 / TR_SEC
    roi_to_entropy = {}
    for col in df.columns:
        ts = df[col].dropna().values
        roi_to_entropy[col] = spectral_entropy(ts, fs=fs, normalize=True)

    # Fetch AAL with indices for accurate mapping
    aal = datasets.fetch_atlas_aal(version="SPM12")
    aal_img = nib.load(aal["maps"])
    labels = [lab.decode("utf-8") if isinstance(lab, bytes) else lab for lab in aal["labels"]]
    indices = list(aal["indices"]) if "indices" in aal else sorted(int(v) for v in np.unique(aal_img.get_fdata()) if v>0)[:len(labels)]
    label_to_index = {labels[i]: int(indices[i]) for i in range(len(labels))}

    # Build data volume
    atlas_data = aal_img.get_fdata()
    out_data = np.zeros_like(atlas_data, dtype=float)

    # Fill region values; missing ROIs remain 0 (background)
    for label, idx in label_to_index.items():
        val = roi_to_entropy.get(label, np.nan)
        if np.isnan(val):
            continue
        out_data[atlas_data == idx] = val

    out_img = image.new_img_like(aal_img, out_data)

    # Robust display limits
    vals = out_data[np.isfinite(out_data) & (out_data>0)]
    vmin, vmax = (0.0, 1.0)
    if vals.size:
        vmin = float(np.percentile(vals, 5))
        vmax = float(np.percentile(vals, 95))

    display = plotting.plot_stat_map(
        out_img,
        display_mode="ortho",
        cut_coords=(0,0,0),
        cmap="plasma",
        vmin=vmin, vmax=vmax,
        colorbar=True,
        title="AAL Spectral Entropy (normalized)"
    )
    display.savefig(OUTPUT_PNG, dpi=200)
    display.close()
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()


