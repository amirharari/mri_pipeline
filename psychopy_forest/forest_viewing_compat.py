"""Forest viewing with classic PsychoPy APIs and a MovieStim3 fallback.

Run in PsychoPy Coder, or use run_forest.py in PyCharm.
The Builder export remains available separately for PsychoPy 2026.1.3.
"""
import csv
from datetime import datetime
import inspect
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
MOVIE = ROOT / "forest_41m58s_to_49m58s_1080p.mp4"
BLACK_SECONDS = 30.0
MOVIE_SECONDS = 480.0


class UserAbort(Exception):
    """Escape was pressed."""


def create_movie(visual, win, filename, size, volume=1.0):
    """Use modern ffpyplayer if present, otherwise the classic MovieStim3 API."""
    common = dict(
        filename=str(filename), units="pix", size=size,
        loop=False, noAudio=False, volume=volume,
    )
    errors = []
    modern = getattr(visual, "MovieStim", None)
    if modern is not None and "movieLib" in inspect.signature(modern.__init__).parameters:
        try:
            options = dict(common, movieLib="ffpyplayer")
            if "autoStart" in inspect.signature(modern.__init__).parameters:
                options["autoStart"] = False
            return modern(win, **options), "MovieStim/ffpyplayer"
        except Exception as exc:
            errors.append("MovieStim: " + str(exc))
    legacy = getattr(visual, "MovieStim3", None)
    if legacy is not None:
        try:
            return legacy(win, **common), "MovieStim3"
        except Exception as exc:
            errors.append("MovieStim3: " + str(exc))
    raise RuntimeError(
        "No working movie backend was found in this PsychoPy installation. "
        "Use a Standalone PsychoPy installation with movie support. "
        + "; ".join(errors)
    )


class SessionLog:
    """Flush event timestamps immediately, including on Escape or playback errors."""

    def __init__(self, directory, participant, version, clock):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        participant = re.sub(r"[^\w-]", "_", participant)[:80] or "anonymous"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = directory / (participant + "_forest_" + stamp + ".csv")
        self.clock = clock
        self.trigger_time = None
        self.file = self.path.open("w", newline="", encoding="utf-8-sig")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "participant", "psychopy_version", "event", "session_seconds",
            "seconds_from_trigger", "detail",
        ])
        self.participant = participant
        self.version = version

    def record(self, name, timestamp=None, detail=""):
        timestamp = self.clock.getTime() if timestamp is None else timestamp
        relative = "" if self.trigger_time is None else "%.6f" % (timestamp - self.trigger_time)
        self.writer.writerow([
            self.participant, self.version, name, "%.6f" % timestamp, relative, detail,
        ])
        self.file.flush()

    def close(self):
        self.file.close()


def wait_for_trigger(win, message, event, journal):
    event.clearEvents(eventType="keyboard")
    while True:
        message.draw()
        win.flip()
        keys = event.getKeys(keyList=["5", "escape"], timeStamped=journal.clock)
        if any(key == "escape" for key, timestamp in keys):
            raise UserAbort()
        for key, timestamp in keys:
            if key == "5":
                journal.trigger_time = timestamp
                journal.record("trigger_5", timestamp=timestamp)
                return


def check_escape(event):
    if event.getKeys(keyList=["escape"]):
        raise UserAbort()


def black_period(win, seconds, name, core, event, journal):
    timer = core.Clock()
    win.callOnFlip(timer.reset)
    win.callOnFlip(journal.record, name + ".started")
    win.flip()
    while timer.getTime() < seconds:
        check_escape(event)
        win.flip()
    journal.record(name + ".stopped")


def run_experiment(participant, fullscr=True, black_seconds=BLACK_SECONDS,
                   movie_seconds=MOVIE_SECONDS, data_dir=None, volume=1.0):
    import psychopy
    from psychopy import core, event, visual
    from psychopy.constants import FINISHED

    if not MOVIE.is_file() or MOVIE.stat().st_size < 1024:
        raise FileNotFoundError(
            "Movie missing or still a Git LFS pointer. Run git lfs pull, "
            "then keep " + MOVIE.name + " beside this script."
        )
    journal = SessionLog(
        data_dir or ROOT / "data", participant, psychopy.__version__, core.Clock(),
    )
    win = movie = None
    try:
        win = visual.Window(
            size=(960, 540), fullscr=fullscr, screen=0, winType="pyglet",
            allowGUI=False, color="black", units="pix", checkTiming=False,
        )
        win.mouseVisible = False
        # Fit 16:9 inside the display without cropping or stretching.
        width = min(float(win.size[0]), float(win.size[1]) * 16 / 9)
        movie, backend = create_movie(visual, win, MOVIE, (width, width * 9 / 16), volume)
        journal.record("prepared", detail=backend + "; " + MOVIE.name)
        message = visual.TextStim(win, text="waiting signal", color="white", height=28)
        wait_for_trigger(win, message, event, journal)
        black_period(win, black_seconds, "black_start", core, event, journal)

        timer = core.Clock()
        movie.play()
        movie.draw()
        win.callOnFlip(timer.reset)
        win.callOnFlip(journal.record, "movie.started")
        win.flip()
        while timer.getTime() < movie_seconds:
            check_escape(event)
            if movie.status == FINISHED:
                if timer.getTime() < movie_seconds - 0.1:
                    raise RuntimeError("Movie ended before the requested viewing duration.")
                break
            movie.draw()
            win.flip()
        # MovieStim.stop() reloads the decoder in some versions and delays the black screen.
        movie.pause()
        journal.record("movie.stopped")
        black_period(win, black_seconds, "black_end", core, event, journal)
        journal.record("completed")
        return journal.path
    except UserAbort:
        journal.record("aborted", detail="Escape")
        return journal.path
    except Exception as exc:
        journal.record("error", detail=str(exc))
        raise
    finally:
        try:
            if movie is not None:
                movie.stop()
        finally:
            try:
                if win is not None:
                    win.close()
            finally:
                journal.close()


def main():
    from psychopy import gui
    info = {"participant": ""}
    dialog = gui.DlgFromDict(info, title="Yakushima forest, eight minutes", sortKeys=False)
    if dialog.OK:
        path = run_experiment(info["participant"])
        print("Session log:", path)


if __name__ == "__main__":
    main()
