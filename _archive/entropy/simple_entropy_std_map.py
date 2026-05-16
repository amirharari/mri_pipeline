"""
Simple entropy-STD map (AAL, volumetric)

- Computes Sample Entropy (SampEn) per ROI in sliding windows
- Aggregates per-ROI standard deviation (STD) of entropy across time
- Projects to AAL atlas volume and renders a single map (PNG)
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, plotting, image


CSV_PATH = r"C:\kpeSoundPath\rest-results-itamar\sub-001_ses-1_task-rest_aal_ts.csv"
TR_SEC = 1.0
WINDOW_SEC = 30
STEP_SEC = 5
OUTPUT_PNG = "simple_entropy_std_map.png"


def sample_entropy(ts: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    """Sample Entropy (SampEn) excluding self-matches; r = r_frac * std(ts)."""
    x = np.asarray(ts, dtype=float)
    if x.size < m + 2 or np.allclose(np.std(x), 0.0):
        return np.nan
    r = r_frac * np.std(x)
    N = x.size

    def _phi(mm: int) -> float:
        patterns = np.array([x[i:i+mm] for i in range(N-mm)], dtype=float)
        if patterns.size == 0:
            return 0.0
        count = 0
        total = 0
        for i in range(len(patterns)):
            for j in range(len(patterns)):
                if i == j:
                    continue
                if np.max(np.abs(patterns[i] - patterns[j])) <= r:
                    count += 1
                total += 1
        return (count / total) if total > 0 else 0.0

    Bm = _phi(m)
    Bm1 = _phi(m+1)
    if Bm == 0 or Bm1 == 0:
        return np.nan
    return float(-np.log(Bm1 / Bm))


def sliding_windows(n: int, w: int, s: int):
    for start in range(0, n - w + 1, s):
        yield start, start + w


def main():
    print("Computing simple AAL entropy-STD map (SampEn, sliding windows)...")
    df = pd.read_csv(CSV_PATH).select_dtypes(include=[np.number])
    # Drop constant/all-NaN columns
    keep = [c for c in df.columns if df[c].std(ddof=0) > 0 and not df[c].isna().all()]
    df = df[keep]

    w_pts = max(1, int(round(WINDOW_SEC / TR_SEC)))
    s_pts = max(1, int(round(STEP_SEC / TR_SEC)))

    # Per-ROI list of entropies over time
    roi_to_series = {c: [] for c in df.columns}
    T = df.shape[0]
    for a, b in sliding_windows(T, w_pts, s_pts):
        sub = df.iloc[a:b, :]
        for c in df.columns:
            ts = sub[c].dropna().values
            roi_to_series[c].append(sample_entropy(ts, m=2, r_frac=0.2))

    # Per-ROI STD across windows
    roi_to_std = {}
    for roi, series in roi_to_series.items():
        arr = np.asarray(series, dtype=float)
        arr = arr[np.isfinite(arr)]
        roi_to_std[roi] = float(np.std(arr)) if arr.size else np.nan

    # Fetch AAL with indices for accurate mapping
    aal = datasets.fetch_atlas_aal(version="SPM12")
    aal_img = nib.load(aal["maps"])
    labels = [lab.decode("utf-8") if isinstance(lab, bytes) else lab for lab in aal["labels"]]
    indices = list(aal["indices"]) if "indices" in aal else sorted(int(v) for v in np.unique(aal_img.get_fdata()) if v>0)[:len(labels)]
    label_to_index = {labels[i]: int(indices[i]) for i in range(len(labels))}

    # Build data volume
    atlas_data = aal_img.get_fdata()
    out_data = np.zeros_like(atlas_data, dtype=float)
    for label, idx in label_to_index.items():
        val = roi_to_std.get(label, np.nan)
        if np.isnan(val):
            continue
        out_data[atlas_data == idx] = val

    out_img = image.new_img_like(aal_img, out_data)

    vals = out_data[np.isfinite(out_data) & (out_data>0)]
    if vals.size:
        vmin = float(np.percentile(vals, 5))
        vmax = float(np.percentile(vals, 95))
    else:
        vmin, vmax = 0.0, 1.0

    display = plotting.plot_stat_map(
        out_img,
        display_mode="ortho",
        cut_coords=(0,0,0),
        cmap="Turbo",
        vmin=vmin, vmax=vmax,
        colorbar=True,
        title=f"AAL SampEn STD (w={WINDOW_SEC}s, step={STEP_SEC}s)"
    )
    display.savefig(OUTPUT_PNG, dpi=200)
    display.close()
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()


