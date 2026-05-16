"""
Run all Quality Assurance scripts in order.

Outputs saved to: outputs/quality_assurance/

Script overview
---------------
  01_tsnr_maps.py          — voxel-wise tSNR maps, per run
  02_motion_vs_cleaning.py — scrub % vs variance explained scatter
  03_roi_tsnr.py           — ROI tSNR for DMN and auditory cortex
  04_denoising_effect.py   — Ciric 2017 benchmarks for all 3 denoising configs
                             (run separately for full control — see below)
  05_compare_configs.py    — compare all 3 config outputs side-by-side

Note: Script 04 is long-running and parallelisable.
Run it separately with:
    python 04_denoising_effect.py --config all --workers 4
Then run this script (or just script 05) to see the comparison.
"""
import os
import subprocess
import sys

REPO    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA_DIR  = os.path.join(REPO, "quality_assurance")
OUT_DIR = os.path.join(REPO, "outputs", "quality_assurance")
os.makedirs(OUT_DIR, exist_ok=True)

# Scripts 01–03 run quickly; 04 is skipped here (run separately with --workers).
# Script 05 compares the outputs of 04.
QUICK_SCRIPTS = [
    ("01_tsnr_maps.py",          "tSNR maps (raw BOLD, voxel-level)"),
    ("02_motion_vs_cleaning.py", "Motion vs cleaning correlation"),
    ("03_roi_tsnr.py",           "ROI tSNR — DMN & Auditory (Schaefer + HO)"),
]

COMPARISON_SCRIPT = ("05_compare_configs.py", "Config comparison table & plots")


if __name__ == "__main__":
    print("=" * 60)
    print("  Quality Assurance Pipeline")
    print("  Output ->", OUT_DIR)
    print("=" * 60)

    for script, desc in QUICK_SCRIPTS:
        path = os.path.join(QA_DIR, script)
        print(f"\n--- {desc} ---")
        ret = subprocess.call([sys.executable, path], cwd=REPO)
        if ret != 0:
            print(f"[WARNING] {script} exited with code {ret}")

    # Run comparison only if at least one denoising output exists
    denoising_dir = os.path.join(OUT_DIR, "denoising_effect")
    config_dirs   = ["config1_lean", "config2_anatomical", "config3_global"]
    any_ready     = any(
        os.path.isfile(os.path.join(denoising_dir, d, "denoising_summary.csv"))
        for d in config_dirs
    )

    if any_ready:
        script, desc = COMPARISON_SCRIPT
        print(f"\n--- {desc} ---")
        path = os.path.join(QA_DIR, script)
        ret  = subprocess.call([sys.executable, path], cwd=REPO)
        if ret != 0:
            print(f"[WARNING] {script} exited with code {ret}")
    else:
        print(
            "\n[SKIP] 05_compare_configs.py — no denoising outputs found yet.\n"
            "  Run first:  python quality_assurance/04_denoising_effect.py --config all --workers 4"
        )

    print("\n" + "=" * 60)
    print("  QA complete.")
    print("=" * 60)
