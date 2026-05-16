"""
Correlation strength vs inter-ROI Euclidean distance (motion artifact check).

If short-distance correlations are significantly higher than long-distance
ones after cleaning, it suggests residual motion is driving short-range
connectivity inflation (Ciric et al. 2017 DM-FC criterion).

Uses Schaefer-100 parcels (matches the DEFAULT_ATLASES list).

Usage
-----
    python plot_distance_dependent_correlation.py          # default output dir
    python plot_distance_dependent_correlation.py --dir D  # custom TS dir
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import datasets
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TS_OUTPUT_DIR, OUTPUTS_DIR, ensure_nilearn_cache  # noqa: E402

ensure_nilearn_cache()


def schaefer_centroids_mm(n_rois: int = 400) -> np.ndarray:
    """Return (n_rois, 3) MNI centroid coordinates for each Schaefer parcel."""
    sch = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)
    img = nib.load(sch.maps)
    data = img.get_fdata()
    aff  = img.affine
    coords = []
    for lbl in range(1, n_rois + 1):
        vox = np.argwhere(data == lbl)
        if len(vox):
            mni = nib.affines.apply_affine(aff, vox.mean(axis=0))
        else:
            mni = np.array([0.0, 0.0, 0.0])
        coords.append(mni)
    return np.array(coords)


def _find_example_csv(ts_dir: str, suffix: str = "_schaefer400_ts.csv") -> str:
    """Return path of the first timeseries CSV matching *suffix* in *ts_dir*."""
    cands = sorted(f for f in os.listdir(ts_dir) if f.endswith(suffix))
    if not cands:
        raise FileNotFoundError(f"No *{suffix} files found in {ts_dir}")
    return os.path.join(ts_dir, cands[0])


def main(ts_dir: str) -> None:
    import pandas as pd

    csv_path = _find_example_csv(ts_dir)
    print(f"Using: {csv_path}")

    df   = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.strip()]
    ts   = df[cols].values                          # (T, n_rois)
    corr = np.corrcoef(ts.T)                        # (n_rois, n_rois)

    n_rois     = corr.shape[0]
    centroids  = schaefer_centroids_mm(n_rois)
    dists      = squareform(pdist(centroids, "euclidean"))

    i_upper = np.triu_indices(n_rois, k=1)
    d_flat  = dists[i_upper]
    r_flat  = corr[i_upper]

    # Median split
    d_median   = np.median(d_flat)
    r_short    = np.mean(r_flat[d_flat <= d_median])
    r_long     = np.mean(r_flat[d_flat >  d_median])
    full_r, _  = __import__("scipy.stats", fromlist=["pearsonr"]).pearsonr(d_flat, r_flat)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(d_flat, r_flat, alpha=0.25, s=5, color="steelblue", rasterized=True)
    ax.axhline(r_short, color="red",  lw=1.5, label=f"Mean r (short dist <= {d_median:.0f} mm): {r_short:.3f}")
    ax.axhline(r_long,  color="blue", lw=1.5, label=f"Mean r (long  dist  > {d_median:.0f} mm): {r_long:.3f}")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Euclidean distance between parcel centroids (mm)")
    ax.set_ylabel("Pearson r")
    ax.set_title(
        f"Correlation vs Distance (Schaefer-{n_rois})\n"
        f"r(FC, distance) = {full_r:+.3f}  "
        f"({'distance-dependent artifact detected' if r_short > r_long + 0.05 else 'no strong distance-dependence'})"
    )
    ax.legend(fontsize=9)

    out_path = os.path.join(ts_dir, "correlation_vs_distance_gsr_off.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    print(f"r(FC, distance)     = {full_r:+.3f}")
    print(f"Mean r (short dist) = {r_short:.3f}")
    print(f"Mean r (long dist)  = {r_long:.3f}")
    if r_short > r_long + 0.05:
        print("-> Short-distance correlations significantly higher: possible residual motion artifact.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        default=os.environ.get("TS_GSR_OFF_DIR", TS_OUTPUT_DIR),
        help="Directory containing timeseries CSVs",
    )
    args = parser.parse_args()
    main(args.dir)
