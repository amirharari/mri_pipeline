"""Run this file in PyCharm to start the eight minute forest experiment."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
MOVIE = ROOT / "forest_41m58s_to_49m58s_1080p.mp4"
EXPERIMENT = ROOT / "forest_viewing_compat.py"


def main():
    for path in [MOVIE, EXPERIMENT]:
        if not path.is_file():
            raise FileNotFoundError(
                "Keep the experiment and its movie together. After cloning, run git lfs pull: "
                + str(path)
            )
    if MOVIE.stat().st_size < 1024:
        raise RuntimeError("The movie has not been downloaded. Run git lfs pull in the repository.")

    # Respect a PsychoPy environment already selected in PyCharm or Coder.
    candidates = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "PsychoPy" / "python.exe",
        Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        / "Programs" / "PsychoPy" / "python.exe",
    ]
    interpreter = Path(sys.executable) if importlib.util.find_spec("psychopy") is not None else None
    if interpreter is None:
        interpreter = next((path for path in candidates if path.is_file()), None)
    if interpreter is None:
        raise RuntimeError(
            "PsychoPy was not found. Select a Python interpreter with PsychoPy installed "
            "or open forest_viewing.psyexp in PsychoPy Builder."
        )

    print("Python:", interpreter, flush=True)
    print("Press 5 after the participant dialog. Escape exits.", flush=True)
    try:
        return subprocess.call([str(interpreter), str(EXPERIMENT), *sys.argv[1:]], cwd=str(ROOT))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
