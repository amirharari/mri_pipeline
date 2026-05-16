"""
Main pipeline — runs all stages in order.

Usage
-----
    python pipeline.py --config anatomical
    python pipeline.py --config strict --no-qc --subjects sub-001 sub-002
    python pipeline.py --config lean   --no-extraction   # rerun QC/connectivity only
    python pipeline.py --config anatomical --no-extraction --no-qc --no-connectivity
                                           # figures only

Stages
------
    1  Timeseries extraction   (--no-extraction  to skip)
    2  QC  dashboard + motion  (--no-qc          to skip)
    3  Connectivity heatmaps   (--no-connectivity to skip)
    4  Session FC figures +
       Per-subject overview +
       LOO psilo similarity    (--no-figures      to skip)

Paths come from config.py — only edit that file when changing machines.
"""
import argparse
import datetime
import os
import subprocess
import sys

import config  # project-wide paths and defaults

_CONFIGS = {
    "lean":       "18 regressors  (12 motion + 6 aCompCor combined WM/CSF)",
    "anatomical": "22 regressors  (12 motion + 5 WM + 5 CSF CompCor)",
    "global":     "20 regressors  (12 motion + 6 aCompCor + GSR + deriv)",
    "strict":     "44 regressors  (24 Friston motion + 10 WM + 10 CSF CompCor)",
}

_parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("--config", choices=list(_CONFIGS), default="lean",
                     help="Denoising config (default: lean)")
_parser.add_argument("--workers",   type=int, default=4,
                     help="Parallel workers for extraction (default: 4)")
_parser.add_argument("--subjects",  nargs="*", default=None,
                     help="Restrict to subjects e.g. sub-001 sub-002")
_parser.add_argument("--no-extraction",   dest="run_extraction",   action="store_false",
                     help="Skip Stage 1 (use existing timeseries CSVs)")
_parser.add_argument("--no-qc",           dest="run_qc",           action="store_false",
                     help="Skip Stage 2 QC scripts")
_parser.add_argument("--no-connectivity", dest="run_connectivity", action="store_false",
                     help="Skip Stage 3 connectivity heatmaps")
_parser.add_argument("--no-figures",      dest="run_figures",      action="store_false",
                     help="Skip Stage 4 session FC + subject overview figures")
_parser.set_defaults(run_extraction=True, run_qc=True, run_connectivity=True,
                     run_figures=True)
_args = _parser.parse_args()

DENOISING_CONFIG = _args.config
RUN_EXTRACTION   = _args.run_extraction
RUN_QC           = _args.run_qc
RUN_CONNECTIVITY = _args.run_connectivity
RUN_FIGURES      = _args.run_figures
N_WORKERS        = _args.workers
MAX_SUBJECTS     = _args.subjects   # None = all subjects

ROOT        = config.PROJECT_ROOT
EXT_DIR     = os.path.join(ROOT, "extraction")
DIAG_DIR    = os.path.join(ROOT, "diagnostics")
CONN_DIR    = os.path.join(ROOT, "connectivity")
OUTPUTS_DIR = config.OUTPUTS_DIR


# ---------------------------------------------------------------------------
# Tee logger — mirrors stdout/stderr to a per-batch log file
# ---------------------------------------------------------------------------
class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def _banner(label: str) -> None:
    width = 64
    print(f"\n{'=' * width}")
    print(f"  {label}")
    print(f"{'=' * width}")


def _run_script(label: str, script: str, env_extra: dict = None,
                extra_args: list = None) -> None:
    """Run a script in its own directory so relative imports resolve correctly."""
    _banner(label)
    env = {**os.environ, **(env_extra or {})}
    cmd = [sys.executable, script] + (extra_args or [])
    result = subprocess.run(cmd, cwd=os.path.dirname(script), env=env)
    if result.returncode != 0:
        print(f"[WARNING] '{label}' exited with code {result.returncode}")


# ---------------------------------------------------------------------------
# All execution inside this guard so Windows multiprocessing can re-import
# this module safely without running the pipeline.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if DENOISING_CONFIG not in config.TS_CONFIG_DIRS:
        raise ValueError(f"Unknown DENOISING_CONFIG '{DENOISING_CONFIG}'. "
                         f"Choose from: {list(config.TS_CONFIG_DIRS)}")

    ts_out_dir = config.TS_CONFIG_DIRS[DENOISING_CONFIG]
    use_gsr    = (DENOISING_CONFIG == "global")
    os.makedirs(ts_out_dir, exist_ok=True)

    # Set up per-batch log
    _stamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = os.path.join(ts_out_dir, f"pipeline_log_{_stamp}.txt")
    _log_fh   = open(_log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    sys.stderr = _Tee(sys.__stderr__, _log_fh)
    print(f"Batch started     : {_stamp}")
    print(f"Denoising config  : {DENOISING_CONFIG}  —  {_CONFIGS[DENOISING_CONFIG]}")
    print(f"FMRIPREP_ROOT     : {config.FMRIPREP_ROOT}")
    print(f"FD threshold      : {config.FD_THRESHOLD} mm  (current + next vol scrubbed)")
    print(f"Output dir        : {ts_out_dir}")
    print(f"Subjects          : {MAX_SUBJECTS or 'all'}")
    print(f"Log               -> {_log_path}\n")

    # ── Stage 1: Timeseries extraction ──────────────────────────────────────
    if RUN_EXTRACTION:
        _banner(f"Stage 1: Timeseries extraction  [{DENOISING_CONFIG}]")

        sys.path.insert(0, EXT_DIR)
        import fmri_timeseries_extractor as ext                    # noqa: E402
        import atlases                                             # noqa: E402
        from discover_psilo_study import discover_psilo_runs      # noqa: E402

        runs = discover_psilo_runs(config.FMRIPREP_ROOT,
                                   subjects=MAX_SUBJECTS)
        print(f"  Discovered {len(runs)} total runs across all tasks/sessions.")

        ext.create_time_series(
            runs, ts_out_dir,
            atlases   = atlases.DEFAULT_ATLASES,
            use_gsr   = use_gsr,
            n_workers = N_WORKERS,
            config    = DENOISING_CONFIG,
        )

    # ── Stages 2–3: QC and connectivity ─────────────────────────────────────
    env_extra = {"TS_DIR": ts_out_dir, "DENOISING_CONFIG": DENOISING_CONFIG}

    if RUN_QC:
        _run_script(
            "Stage 2a: GCOR + scrubbing dashboard",
            os.path.join(DIAG_DIR, "visualize_metrics.py"),
            env_extra=env_extra,
        )
        _run_script(
            "Stage 2b: Distance-dependent correlation (motion check)",
            os.path.join(DIAG_DIR, "plot_distance_dependent_correlation.py"),
            env_extra=env_extra,
        )

    if RUN_CONNECTIVITY:
        _run_script(
            "Stage 3: Connectivity matrix heatmaps",
            os.path.join(CONN_DIR, "show_connectivity_matrices.py"),
            env_extra=env_extra,
        )

    if RUN_FIGURES:
        # Pass --data so all figure scripts use the correct config's timeseries dir
        data_args = ["--data", ts_out_dir]
        subj_args = (["--subjects"] + MAX_SUBJECTS) if MAX_SUBJECTS else []

        _run_script(
            "Stage 4a: Session FC analysis figures (rest + music, group diffs)",
            os.path.join(CONN_DIR, "session_fc_analysis.py"),
            env_extra=env_extra,
            extra_args=data_args,
        )
        _run_script(
            "Stage 4b: Per-subject overview grids (all tasks × sessions)",
            os.path.join(CONN_DIR, "subject_overview.py"),
            env_extra=env_extra,
            extra_args=data_args + subj_args,
        )
        _run_script(
            "Stage 4c: LOO psilocybin signature similarity (music imprint analysis)",
            os.path.join(CONN_DIR, "music_psychedelic_similarity.py"),
            env_extra=env_extra,
            extra_args=data_args + subj_args,
        )

    print(f"\n{'=' * 64}")
    print(f"  Pipeline complete  [{DENOISING_CONFIG}]")
    print(f"  Outputs -> {ts_out_dir}")
    print(f"  Log     -> {_log_path}")
    print(f"{'=' * 64}\n")
    _log_fh.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
