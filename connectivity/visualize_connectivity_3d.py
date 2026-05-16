"""
3D Brain Visualization of Connectivity Differences: Psilocybin - Baseline.

Interactive 6-view brain surface rendered with Plotly, showing mean
connectivity difference per ROI (psilocybin minus baseline) mapped onto
the fsaverage cortical surface.

Requires: AAL-labelled timeseries CSVs for ses-1 (baseline) and ses-2 (psilocybin).
The AAL atlas is not in DEFAULT_ATLASES by default — you can re-add it in
extraction/atlases.py and re-run the pipeline if needed.

Usage
-----
    python visualize_connectivity_3d.py --baseline PATH --psilocybin PATH
    python visualize_connectivity_3d.py --dir TS_DIR --subject sub-001
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE, ".."))
from utils import (fetch_aal_spm12, fetch_fsaverage_surfaces,   # noqa: E402
                   sample_labels, build_vertex_data, map_roi_to_indices,
                   pearson_matrix)
from config import TS_OUTPUT_DIR, OUTPUTS_DIR  # noqa: E402

DEFAULT_OUT = os.path.join(OUTPUTS_DIR, "connectivity_3d")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_mean_connectivity(df: pd.DataFrame) -> pd.Series:
    """Mean absolute connectivity strength per ROI."""
    corr = pearson_matrix(df)
    return pd.Series({
        roi: corr.loc[:, roi].drop(roi).abs().mean()
        for roi in corr.columns
    })


def create_3d_visualization(baseline_csv: str, psilocybin_csv: str, out_dir: str) -> None:
    """Render a 6-view 3D difference map and save as PNG."""
    print(f"  Baseline  : {baseline_csv}")
    print(f"  Psilocybin: {psilocybin_csv}")

    df_base  = pd.read_csv(baseline_csv)
    df_psilo = pd.read_csv(psilocybin_csv)

    diff = compute_mean_connectivity(df_psilo) - compute_mean_connectivity(df_base)
    print(f"  Diff range: [{diff.min():.4f}, {diff.max():.4f}]")

    _aal, atlas_img, _atlas_data, name_to_index = fetch_aal_spm12()
    pial_l, pial_r, _wl, _wr = fetch_fsaverage_surfaces()

    diff_map = map_roi_to_indices(diff.to_dict(), name_to_index)
    labels_l = sample_labels(atlas_img, pial_l)
    labels_r = sample_labels(atlas_img, pial_r)
    coords_l, faces_l, intens_l = build_vertex_data(pial_l, labels_l, diff_map)
    coords_r, faces_r, intens_r = build_vertex_data(pial_r, labels_r, diff_map)

    nonzero = np.concatenate([intens_l[intens_l != 0], intens_r[intens_r != 0]])
    vmax    = float(np.percentile(np.abs(nonzero), 95)) if len(nonzero) else 0.1

    coords_both = np.vstack([coords_l, coords_r + 100])
    faces_both  = np.vstack([faces_l,  faces_r + len(coords_l)])
    intens_both = np.hstack([intens_l, intens_r])

    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{"type": "mesh3d"}] * 3, [{"type": "mesh3d"}] * 3],
        subplot_titles=("Left Lateral", "Posterior", "Right Lateral",
                        "Superior",    "Anterior",  "Inferior"),
        horizontal_spacing=0.05, vertical_spacing=0.10,
    )

    views = [
        (0,    0,  1, 1), (90,    0,  1, 2), (180,  0,  1, 3),
        (90,  90,  2, 1), (-90,   0,  2, 2), (-90, -90, 2, 3),
    ]
    for az, el, row, col in views:
        fig.add_trace(
            go.Mesh3d(
                x=coords_both[:, 0], y=coords_both[:, 1], z=coords_both[:, 2],
                i=faces_both[:, 0],  j=faces_both[:, 1],  k=faces_both[:, 2],
                intensity=intens_both,
                colorscale="RdBu_r",
                showscale=False,
                intensitymode="vertex",
            ),
            row=row, col=col,
        )
        fig.update_scenes(
            xaxis=dict(showbackground=False, visible=False),
            yaxis=dict(showbackground=False, visible=False),
            zaxis=dict(showbackground=False, visible=False),
            aspectmode="cube",
            camera=dict(eye=dict(
                x=1.5 * np.cos(np.radians(az)) * np.cos(np.radians(el)),
                y=1.5 * np.sin(np.radians(az)) * np.cos(np.radians(el)),
                z=1.5 * np.sin(np.radians(el)),
            )),
            row=row, col=col,
        )

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=0, color=[-vmax, vmax], colorscale="RdBu_r", showscale=True,
                    colorbar=dict(title="Connectivity<br>Difference", len=0.6, y=0.5, x=1.15)),
    ))
    fig.update_layout(
        title_text="Connectivity Differences: Psilocybin - Baseline (6 Views)",
        title_font_size=20, showlegend=False, width=1800, height=1200,
    )

    os.makedirs(out_dir, exist_ok=True)
    tag  = os.path.basename(baseline_csv).replace("_ses-1_task-rest_aal_ts.csv", "")
    path = os.path.join(out_dir, f"{tag}_connectivity_differences_3d.png")
    fig.write_image(path, width=1800, height=1200)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Auto-discovery helper
# ---------------------------------------------------------------------------

def _find_aal_pair(ts_dir: str, subject: str):
    """Return (baseline_csv, psilocybin_csv) for *subject* using AAL files."""
    def _find(ses):
        pat = os.path.join(ts_dir, f"{subject}_{ses}_task-rest_aal_ts.csv")
        hits = sorted(glob.glob(pat))
        return hits[0] if hits else None
    return _find("ses-1"), _find("ses-2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",   default=None, help="Path to baseline AAL timeseries CSV")
    parser.add_argument("--psilocybin", default=None, help="Path to psilocybin AAL timeseries CSV")
    parser.add_argument("--dir",        default=TS_OUTPUT_DIR, help="Timeseries directory (auto-discover)")
    parser.add_argument("--subject",    default="sub-001", help="Subject ID for auto-discovery")
    parser.add_argument("--out",        default=DEFAULT_OUT, help="Output directory")
    args = parser.parse_args()

    if args.baseline and args.psilocybin:
        baseline_csv   = args.baseline
        psilocybin_csv = args.psilocybin
    else:
        baseline_csv, psilocybin_csv = _find_aal_pair(args.dir, args.subject)
        if not baseline_csv or not psilocybin_csv:
            print(f"Could not find AAL rest timeseries for {args.subject} in {args.dir}")
            print("Make sure the AAL atlas is in DEFAULT_ATLASES (extraction/atlases.py) and re-run extraction.")
            raise SystemExit(1)

    create_3d_visualization(baseline_csv, psilocybin_csv, args.out)
