"""Brief local playback check, not an experimental run."""
from pathlib import Path
from psychopy import visual, core

root = Path(__file__).resolve().parent
win = visual.Window((640, 360), fullscr=False, units='pix', color='black')
try:
    for filename in ['stimulus_twice.mp4', 'stimulus_three_times.mp4']:
        movie = visual.MovieStim(win, str(root / filename), size=(640, 360),
                                 loop=False, noAudio=False, volume=0)
        movie.play()
        clock = core.Clock()
        frames = 0
        while clock.getTime() < 2:
            movie.draw()
            win.flip()
            frames += 1
        assert not movie.isFinished, 'Movie ended prematurely'
        win.getMovieFrame(buffer='front')
        win.saveMovieFrames(str(root / (Path(filename).stem + '_check.png')))
        movie.stop()
        print(filename, 'rendered', frames, 'display frames', flush=True)
finally:
    win.close()
