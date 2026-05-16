"""Discovery script for the psilocybin study fMRIPrep output tree.

Structure assumed
----------------
<STUDY_ROOT>/
  sub-{id}/
    ses-{N}/
      func/
        *_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
        *_desc-confounds_timeseries.tsv

Tasks discovered: task-rest, task-music (all acquisitions / runs)
Sessions:         ses-1, ses-2, ses-3  (configurable via SESSIONS)

Outputs
-------
  <OUT_DIR>/run_manifest.csv   — every discovered BOLD/confounds pair
  <OUT_DIR>/run_manifest.json  — same, JSON format for programmatic use

Usage
-----
  # preview only (prints manifest, no extraction)
  python discover_psilo_study.py

  # then pass the discovered runs to the extractor:
  from discover_psilo_study import discover_psilo_runs
  from atlases import DEFAULT_ATLASES
  from fmri_timeseries_extractor import create_time_series

  runs = discover_psilo_runs(STUDY_ROOT)
  create_time_series(runs, output_dir="...", atlases=DEFAULT_ATLASES, use_gsr=False)
"""

import json
import os
import re
import sys
from typing import List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FMRIPREP_ROOT, SESSIONS, OUTPUTS_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration — derived from config.py (edit paths there, not here)
# ---------------------------------------------------------------------------
STUDY_ROOT = FMRIPREP_ROOT
OUT_DIR    = os.path.join(OUTPUTS_DIR, "manifests")

# Pattern that identifies a preprocessed BOLD in MNI space
_BOLD_PATTERN = re.compile(
    r"space-MNI152NLin2009cAsym_res-2_desc-preproc_bold\.nii\.gz$"
)

# BIDS entity extractors
_RE_SUBJECT = re.compile(r"(sub-[^_]+)")
_RE_SESSION = re.compile(r"(ses-[^_]+)")
_RE_TASK    = re.compile(r"task-([^_]+)")
_RE_ACQ     = re.compile(r"acq-([^_]+)")
_RE_RUN     = re.compile(r"(run-[^_]+)")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _confounds_path_from_bold(bold_path: str) -> Optional[str]:
    """Derive the confounds TSV path from a preproc BOLD path.

    The BOLD filename up to the first space/res/desc entity is the shared
    BIDS prefix used by the confounds file:

      sub-002_ses-1_task-music_acq-agami_run-3  ← shared prefix
        _space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz  ← BOLD suffix
        _desc-confounds_timeseries.tsv                              ← confounds suffix

    Fallback: some fMRIPrep outputs have a spurious extra entity before
    _space- (e.g. _acq-glass_run-1_glass_space-...).  We progressively strip
    trailing BIDS-like tokens before _space- until we find a match.
    """
    folder   = os.path.dirname(bold_path)
    basename = os.path.basename(bold_path)

    # Strip the MNI-space suffix to recover the candidate BIDS prefix
    prefix = re.sub(r"_space-.*$", "", basename)

    # Try the prefix directly, then progressively drop trailing _token segments
    # (handles e.g. the stray '_glass' in sub-002_ses-1_..._acq-glass_run-1_glass_space-...)
    for _ in range(5):
        conf = os.path.join(folder, prefix + "_desc-confounds_timeseries.tsv")
        if os.path.isfile(conf):
            return conf
        # Drop the last _<token> segment and retry
        new_prefix = re.sub(r"_[^_]+$", "", prefix)
        if new_prefix == prefix:
            break
        prefix = new_prefix

    return None


def discover_psilo_runs(
    study_root: str = STUDY_ROOT,
    sessions:   List[str] = None,
    subjects:   Optional[List[str]] = None,
) -> "List":
    """Return a list of BoldRun objects for every paired BOLD+confounds found.

    Parameters
    ----------
    study_root : root of the fMRIPrep output tree (contains sub-* folders)
    sessions   : session labels to include (default: ses-1, ses-2, ses-3)
    subjects   : restrict to these subject IDs (e.g. ["sub-002"]); None = all
    """
    # Import here so this module can also be used without the full extraction env
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fmri_timeseries_extractor import BoldRun  # noqa: PLC0415

    if sessions is None:
        sessions = SESSIONS

    runs: List[BoldRun] = []

    for subj in sorted(os.listdir(study_root)):
        if not subj.startswith("sub-"):
            continue
        if subjects and subj not in subjects:
            continue

        subj_path = os.path.join(study_root, subj)
        if not os.path.isdir(subj_path):
            continue

        for ses in sessions:
            func_dir = os.path.join(subj_path, ses, "func")
            if not os.path.isdir(func_dir):
                continue

            for fname in sorted(os.listdir(func_dir)):
                if not _BOLD_PATTERN.search(fname):
                    continue

                bold_path = os.path.join(func_dir, fname)
                conf_path = _confounds_path_from_bold(bold_path)

                if conf_path is None:
                    print(f"  [WARN] No confounds for {fname} — skipping")
                    continue

                # Parse BIDS entities
                task_m = _RE_TASK.search(fname)
                acq_m  = _RE_ACQ.search(fname)
                run_m  = _RE_RUN.search(fname)

                task = task_m.group(1) if task_m else "unknown"
                acq  = acq_m.group(1)  if acq_m  else None
                run  = run_m.group(0)  if run_m  else "run-1"

                # Encode acq into the task string so it survives into output filenames:
                #   task-music_acq-agami_run-3_aal_ts.csv
                task_label = f"{task}_acq-{acq}" if acq else task

                runs.append(BoldRun(
                    subject       = subj,
                    session       = ses,
                    task          = task_label,
                    run           = run,
                    bold_path     = bold_path,
                    confounds_path= conf_path,
                ))

    return runs


def save_manifest(runs: "List", out_dir: str = OUT_DIR) -> str:
    """Write run_manifest.csv and run_manifest.json to out_dir.  Returns CSV path."""
    os.makedirs(out_dir, exist_ok=True)

    rows = [
        {
            "subject":       r.subject,
            "session":       r.session,
            "task":          r.task,
            "run":           r.run,
            "bold_path":     r.bold_path,
            "confounds_path": r.confounds_path,
        }
        for r in runs
    ]

    csv_path  = os.path.join(out_dir, "run_manifest.csv")
    json_path = os.path.join(out_dir, "run_manifest.json")

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(json_path, "w") as fh:
        json.dump(rows, fh, indent=2)

    return csv_path


# ---------------------------------------------------------------------------
# Entry point — preview only, no extraction
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Scanning: {STUDY_ROOT}")
    print(f"Sessions: {SESSIONS}\n")

    runs = discover_psilo_runs()

    if not runs:
        print("No runs found — check STUDY_ROOT and SESSIONS.")
        sys.exit(1)

    # Print summary table
    rows = [
        {"subject": r.subject, "session": r.session,
         "task": r.task, "run": r.run,
         "bold": os.path.basename(r.bold_path)}
        for r in runs
    ]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\nTotal: {len(runs)} run(s)")

    # Save manifest
    csv_path = save_manifest(runs)
    print(f"\nManifest saved -> {csv_path}")
    print(f"               -> {csv_path.replace('.csv', '.json')}")
