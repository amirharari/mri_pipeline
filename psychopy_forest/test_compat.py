"""Checks for backend selection and fresh scanner triggers without requiring PsychoPy."""
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from forest_viewing_compat import SessionLog, UserAbort, create_movie, wait_for_trigger


class LegacyMovie:
    def __init__(self, win, filename, units, size, loop, noAudio, volume):
        self.filename = filename


class MovieBackendTests(unittest.TestCase):
    def test_classic_installation_uses_moviestim3(self):
        class OldMovieStim:
            def __init__(self, win, filename):
                raise AssertionError("The old AVbin backend should not be selected.")

        visual = SimpleNamespace(MovieStim=OldMovieStim, MovieStim3=LegacyMovie)
        movie, backend = create_movie(visual, object(), Path("clip.mp4"), (960, 540))
        self.assertIsInstance(movie, LegacyMovie)
        self.assertEqual(backend, "MovieStim3")

    def test_modern_movie_is_preloaded_without_starting(self):
        class ModernMovie:
            def __init__(self, win, movieLib, autoStart=True, **kwargs):
                self.autoStart = autoStart
                self.movieLib = movieLib

        movie, backend = create_movie(SimpleNamespace(MovieStim=ModernMovie), object(), "clip.mp4", (960, 540))
        self.assertFalse(movie.autoStart)
        self.assertEqual(movie.movieLib, "ffpyplayer")
        self.assertEqual(backend, "MovieStim/ffpyplayer")

    def test_unavailable_ffpyplayer_falls_back_to_legacy(self):
        class UnavailableMovie:
            def __init__(self, win, movieLib, **kwargs):
                raise ImportError("ffpyplayer unavailable")

        visual = SimpleNamespace(MovieStim=UnavailableMovie, MovieStim3=LegacyMovie)
        movie, backend = create_movie(visual, object(), "clip.mp4", (960, 540))
        self.assertIsInstance(movie, LegacyMovie)

    def test_missing_backend_reports_a_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "No working movie backend"):
            create_movie(SimpleNamespace(), object(), "clip.mp4", (960, 540))


class TriggerTests(unittest.TestCase):
    def test_stale_key_is_discarded_and_fresh_timestamp_is_saved(self):
        class Events:
            def __init__(self):
                self.keys = [[("5", 1.0)], [], [("5", 3.25)]]

            def clearEvents(self, eventType):
                self.keys.pop(0)

            def getKeys(self, keyList, timeStamped):
                return self.keys.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            journal = SessionLog(directory, "test", "test-version", SimpleNamespace(getTime=lambda: 9.0))
            try:
                wait_for_trigger(SimpleNamespace(flip=lambda: None), SimpleNamespace(draw=lambda: None), Events(), journal)
                self.assertEqual(journal.trigger_time, 3.25)
            finally:
                journal.close()
            with journal.path.open(encoding="utf-8-sig", newline="") as handle:
                row = list(csv.DictReader(handle))[0]
            self.assertEqual(row["event"], "trigger_5")
            self.assertEqual(row["session_seconds"], "3.250000")
            self.assertEqual(row["seconds_from_trigger"], "0.000000")

    def test_escape_takes_priority_over_trigger_in_same_frame(self):
        events = SimpleNamespace(
            clearEvents=lambda eventType: None,
            getKeys=lambda keyList, timeStamped: [("5", 1.0), ("escape", 1.01)],
        )
        journal = SimpleNamespace(clock=object(), trigger_time=None)
        with self.assertRaises(UserAbort):
            wait_for_trigger(SimpleNamespace(flip=lambda: None), SimpleNamespace(draw=lambda: None), events, journal)
        self.assertIsNone(journal.trigger_time)


if __name__ == "__main__":
    unittest.main()
