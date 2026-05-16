"""
Quality Assurance 1: tSNR maps for every subject and run.

Computes voxel-wise tSNR = mean / std across time, masked by brain.
Saves a tSNR NIfTI and a summary PNG for each BOLD file.
"""
import gc
import os
import re
import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FMRIPREP_ROOT, QA_OUTPUT_DIR  # noqa: E402

OUT_DIR = os.path.join(QA_OUTPUT_DIR, "tsnr_maps")
os.makedirs(OUT_DIR, exist_ok=True)


def find_bold_and_mask(func_dir):
    """Yield (bold_path, mask_path) for each preprocessed BOLD in func_dir."""
    for f in sorted(os.listdir(func_dir)):
        if not (f.endswith(".nii.gz") and "preproc_bold" in f and "MNI" in f):
            continue
        bold_path = os.path.join(func_dir, f)
        mask_path = bold_path.replace("_desc-preproc_bold.nii.gz", "_desc-brain_mask.nii.gz")
        if not os.path.isfile(mask_path):
            mask_path = None  # will use non-zero voxels as fallback
        yield bold_path, mask_path


def compute_tsnr_map(bold_path, mask_path=None):
    """Compute voxel-wise tSNR = mean / std. Returns (tsnr_3d, mean_tsnr_brain)."""
    img  = nib.load(bold_path)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    del img   # release NIfTI proxy; data is already in memory

    mean_t = np.mean(data, axis=-1)
    std_t  = np.std(data,  axis=-1)
    std_t  = np.maximum(std_t, 1e-6)   # safe clamp, no in-place aliasing risk
    tsnr   = mean_t / std_t

    if mask_path and os.path.isfile(mask_path):
        mask = nib.load(mask_path).get_fdata() > 0.5
    else:
        mask = mean_t > 50   # reuse already-computed mean, no extra pass over data

    del data   # free 4D BOLD
    gc.collect()

    tsnr_masked = np.where(mask, tsnr, np.nan)
    mean_tsnr   = float(np.nanmean(tsnr_masked))
    return tsnr, mean_tsnr, affine, mask


def save_tsnr_figure(tsnr, mask, out_path, title, mean_tsnr):
    """Save a 3-slice view of tSNR."""
    tsnr_display = np.where(mask, tsnr, np.nan)
    x, y, z = np.array(tsnr_display.shape[:3]) // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (slc, lbl) in zip(axes, [(x, "x"), (y, "y"), (z, "z")]):
        if lbl == "x":
            im = ax.imshow(tsnr_display[slc, :, :].T, origin="lower", cmap="viridis",
                           vmin=0, vmax=np.nanpercentile(tsnr_display, 99))
        elif lbl == "y":
            im = ax.imshow(tsnr_display[:, slc, :].T, origin="lower", cmap="viridis",
                           vmin=0, vmax=np.nanpercentile(tsnr_display, 99))
        else:
            im = ax.imshow(tsnr_display[:, :, slc].T, origin="lower", cmap="viridis",
                           vmin=0, vmax=np.nanpercentile(tsnr_display, 99))
        ax.set_title(f"Slice {lbl}={slc}")
        plt.colorbar(im, ax=ax, label="tSNR")
    fig.suptitle(f"{title}\nMean tSNR (brain) = {mean_tsnr:.1f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    results = []
    for subj in sorted(p for p in os.listdir(FMRIPREP_ROOT) if p.startswith("sub-")):
        subj_path = os.path.join(FMRIPREP_ROOT, subj)
        if not os.path.isdir(subj_path):
            continue
        for ses in sorted(p for p in os.listdir(subj_path) if p.startswith("ses-")):
            func_dir = os.path.join(subj_path, ses, "func")
            if not os.path.isdir(func_dir):
                continue

            for bold_path, mask_path in find_bold_and_mask(func_dir):
                fname = os.path.basename(bold_path)
                # Parse entities
                task_m = re.search(r"task-([a-zA-Z0-9]+)", fname)
                acq_m = re.search(r"acq-([a-zA-Z0-9]+)", fname)
                run_m = re.search(r"run-([a-zA-Z0-9]+)", fname)
                task = task_m.group(1) if task_m else "unknown"
                acq = acq_m.group(1) if acq_m else ""
                run = f"run-{run_m.group(1)}" if run_m else "run-1"

                label = f"{subj}_{ses}_{task}"
                if acq:
                    label += f"_{acq}"
                label += f"_{run}"

                print(f"Processing {label}...")
                try:
                    tsnr, mean_tsnr, aff, mask = compute_tsnr_map(bold_path, mask_path)

                    # Save tSNR NIfTI
                    tsnr_nii = nib.Nifti1Image(np.where(mask, tsnr, 0).astype(np.float32), aff)
                    nii_path = os.path.join(OUT_DIR, f"{label}_tsnr.nii.gz")
                    nib.save(tsnr_nii, nii_path)

                    # Save figure
                    png_path = os.path.join(OUT_DIR, f"{label}_tsnr.png")
                    save_tsnr_figure(tsnr, mask, png_path, label, mean_tsnr)

                    results.append({
                        "subject": subj,
                        "session": ses,
                        "task": task,
                        "acq": acq or "",
                        "run": run,
                        "mean_tSNR": round(mean_tsnr, 2),
                    })
                except Exception as e:
                    print(f"  [ERROR] {e}")
                    results.append({
                        "subject": subj, "session": ses, "task": task, "acq": acq or "", "run": run,
                        "mean_tSNR": np.nan,
                    })

    # Save summary CSV
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, "tsnr_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved tSNR summary -> {csv_path}")
    print(f"Total runs: {len(df)}, Mean tSNR: {df['mean_tSNR'].mean():.1f}")


if __name__ == "__main__":
    main()
