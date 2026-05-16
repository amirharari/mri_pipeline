"""run_psilo_subject.py — extract timeseries for a single psilo-study subject.

Usage
-----
  python run_psilo_subject.py --subject sub-004 --out psilo_sub004_GSR_off
  python run_psilo_subject.py --subject sub-002 --out psilo_sub002_GSR_off_v2
"""
import argparse
import os
import sys

_HERE     = os.path.dirname(os.path.abspath(__file__))
_REPO     = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from discover_psilo_study import discover_psilo_runs, save_manifest
from atlases import DEFAULT_ATLASES
import fmri_timeseries_extractor as ext


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True,
                        help="Subject ID, e.g. sub-004")
    parser.add_argument("--out", required=True,
                        help="Output folder name under outputs/, e.g. psilo_sub004_GSR_off")
    parser.add_argument("--gsr", action="store_true", default=False,
                        help="Enable global signal regression (default: off)")
    args = parser.parse_args()

    out_dir = os.path.join(_REPO, "outputs", args.out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Subject  : {args.subject}")
    print(f"GSR      : {'on' if args.gsr else 'off'}")
    print(f"Output   : {out_dir}\n")

    runs = discover_psilo_runs(subjects=[args.subject])
    if not runs:
        print("No runs found — check subject ID and STUDY_ROOT in discover_psilo_study.py")
        sys.exit(1)

    print(f"Discovered {len(runs)} run(s):")
    for r in runs:
        print(f"  {r.subject}  {r.session}  {r.task}  {r.run}")

    manifest_dir = os.path.join(_REPO, "outputs", "manifests", args.subject)
    save_manifest(runs, out_dir=manifest_dir)
    print(f"\nManifest -> {manifest_dir}\n")

    ext.create_time_series(
        runs, out_dir,
        atlases=DEFAULT_ATLASES,
        use_gsr=args.gsr,
    )
    print(f"\nDone — timeseries saved to {out_dir}")


if __name__ == "__main__":
    main()
