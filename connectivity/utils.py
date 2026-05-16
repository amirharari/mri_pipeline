"""
Shared utilities for all connectivity analysis scripts.
Import from here instead of duplicating across files.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YEO_NETWORKS: Dict[str, str] = {
    "Visual":             "Vis",
    "Somatomotor":        "SomMot",
    "Dorsal_Attention":   "DorsAttn",
    "Ventral_Attention":  "SalVentAttn",
    "Limbic":             "Limbic",
    "Control":            "Cont",
    "Default":            "Default",
}

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_timeseries(filepath: str) -> Optional[pd.DataFrame]:
    """Load a single timeseries CSV. Returns None on failure."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"  [WARN] File not found: {filepath}")
        return None
    except Exception as e:
        print(f"  [WARN] Error reading {filepath}: {e}")
        return None


def load_and_concatenate(file_list: List[str]) -> Optional[pd.DataFrame]:
    """Load multiple timeseries CSVs and concatenate row-wise."""
    dfs = []
    for path in file_list:
        df = load_timeseries(path)
        if df is not None:
            dfs.append(df)
            print(f"  Loaded {os.path.basename(path)} ({df.shape[0]} timepoints)")
        else:
            print(f"  Missing: {os.path.basename(path)}")
    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {combined.shape[0]} total timepoints")
    return combined

# ---------------------------------------------------------------------------
# Atlas / network helpers
# ---------------------------------------------------------------------------

def identify_network_columns(df: pd.DataFrame, keyword: str) -> List[str]:
    """Return column names that contain *keyword*."""
    return [c for c in df.columns if keyword in str(c)]


# ---------------------------------------------------------------------------
# Connectivity math
# ---------------------------------------------------------------------------

def pearson_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of a timeseries DataFrame."""
    return df.corr()


def fisher_z_mean(r_vals: np.ndarray) -> float:
    """
    Proper average of correlations via Fisher z-transformation.
    clip → arctanh (z) → mean z → tanh back to r.
    """
    clipped = np.clip(r_vals, -0.999, 0.999)
    z_vals = np.arctanh(clipped)
    return float(np.tanh(np.mean(z_vals)))


def within_network_connectivity(df: pd.DataFrame, cols: List[str]) -> Optional[Dict]:
    """
    Compute within-network mean connectivity using Fisher z-transform.
    Returns dict with keys: mean, median, std, n_connections, n_regions, values, fisher_z_values.
    Returns None if fewer than 2 columns.
    """
    if len(cols) < 2:
        return None
    corr = df[cols].corr()
    n = len(cols)
    idx = np.triu_indices(n, k=1)
    r_vals = corr.values[idx]
    z_vals = np.arctanh(np.clip(r_vals, -0.999, 0.999))
    mean_r = float(np.tanh(np.mean(z_vals)))
    return {
        "mean":           mean_r,
        "median":         float(np.median(r_vals)),
        "std":            float(np.std(z_vals)),
        "n_connections":  len(r_vals),
        "n_regions":      n,
        "values":         r_vals,
        "fisher_z_values": z_vals,
        "matrix":         corr,
    }

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_connectivity_matrix(
    corr: np.ndarray,
    title: str,
    out_path: str,
    labels: Optional[List[str]] = None,
    vmin: float = -0.5,
    vmax: float = 0.5,
) -> None:
    """Save a connectivity heatmap to *out_path*."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, label="r")
    if labels and len(labels) <= 20:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_session_heatmap(
    conn: pd.DataFrame,
    title: str,
    ax: plt.Axes,
    short_labels: Optional[List[str]] = None,
) -> None:
    """Draw a seaborn heatmap on *ax* for one session connectivity matrix."""
    sns.heatmap(
        conn, ax=ax, cmap="coolwarm", center=0,
        vmin=-1, vmax=1, square=True,
        cbar_kws={"label": "Correlation"},
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    if short_labels:
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short_labels, rotation=0, fontsize=8)

# ---------------------------------------------------------------------------
# AAL atlas + brain surface helpers (shared by 3D visualisation scripts)
# ---------------------------------------------------------------------------

def fetch_aal_spm12() -> Tuple:
    """Load AAL SPM12 atlas. Returns (aal, atlas_img, atlas_data, name_to_index)."""
    from nilearn import datasets as _datasets
    aal = _datasets.fetch_atlas_aal(version="SPM12")
    atlas_img = nib.load(aal["maps"])
    atlas = np.rint(atlas_img.get_fdata()).astype(np.int32)
    labels = [lab.decode("utf-8") if isinstance(lab, bytes) else lab for lab in aal["labels"]]
    if "indices" in aal:
        indices_raw = list(aal["indices"])
    else:
        indices_raw = sorted(int(v) for v in np.unique(atlas) if v > 0)[: len(labels)]
    name_to_index = {labels[i]: int(indices_raw[i]) for i in range(len(labels))}
    return aal, atlas_img, atlas, name_to_index


def fetch_fsaverage_surfaces() -> Tuple[str, str, str, str]:
    """Return (pial_left, pial_right, white_left, white_right) paths."""
    from nilearn import datasets as _datasets
    fsavg = _datasets.fetch_surf_fsaverage()
    return fsavg["pial_left"], fsavg["pial_right"], fsavg["white_left"], fsavg["white_right"]


def sample_labels(atlas_img: nib.Nifti1Image, pial_mesh: str) -> np.ndarray:
    """Sample integer AAL labels onto a surface mesh."""
    from nilearn import surface as _surface
    try:
        lbl = _surface.vol_to_surf(
            atlas_img, pial_mesh,
            inner_mesh=None, kind="line", n_samples=25, interpolation="nearest",
        )
    except TypeError:
        lbl = _surface.vol_to_surf(atlas_img, pial_mesh)
    lbl = np.rint(np.asarray(lbl)).astype(np.int32)
    lbl[lbl < 0] = 0
    return lbl


def build_vertex_data(
    surf_mesh: str,
    labels_on_vertices: np.ndarray,
    roi_index_to_val: Dict[int, float],
    index_to_name: Optional[Dict[int, str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (coords, faces, intensities) for surface rendering."""
    from nilearn import surface as _surface
    coords, faces = _surface.load_surf_mesh(surf_mesh)
    intens = np.zeros(coords.shape[0], dtype=float)
    for idx_val, v in roi_index_to_val.items():
        mask = labels_on_vertices == idx_val
        if np.any(mask):
            intens[mask] = 0.0 if v is None or np.isnan(v) else float(v)
    return coords, faces, intens


def map_roi_to_indices(roi_to_val: Dict[str, float], name_to_index: Dict[str, int]) -> Dict[int, float]:
    """Convert {roi_name: value} → {atlas_index: value}."""
    return {
        name_to_index[roi]: v
        for roi, v in roi_to_val.items()
        if roi in name_to_index
    }
