"""Build the forest viewing experiment using the installed PsychoPy."""
from pathlib import Path

from psychopy.experiment import Experiment
from psychopy.experiment.components.keyboard import KeyboardComponent
from psychopy.experiment.components.movie import MovieComponent
from psychopy.experiment.components.polygon import PolygonComponent
from psychopy.experiment.components.text import TextComponent


ROOT = Path(__file__).resolve().parent
MOVIE_NAME = "forest_41m58s_to_49m58s_1080p.mp4"
ROUTINES = ["wait_for_5", "black_start", "viewing", "black_end"]


def build():
    exp = Experiment()
    settings = {
        "expName": "Yakushima forest 41m58s to 49m58s",
        "Experiment info": "{'participant': ''}",
        "Full-screen window": True,
        "Show info dlg": True,
        "Enable Escape": True,
        "Show mouse": False,
        "measureFrameRate": False,
        "frameRate": 60,
        "Units": "height",
        "color": "black",
        "Data filename": "'data/%s_%s' % (expInfo['participant'], expInfo['date'])",
    }
    for key, value in settings.items():
        exp.settings.params[key].val = value

    for name in ROUTINES:
        routine = exp.addRoutine(name)
        if name == "wait_for_5":
            routine.addComponent(TextComponent(
                exp, name, name="waiting_message", text="waiting signal",
                units="height", color="white", pos=(0, 0), letterHeight=0.05,
                startVal=0, stopVal="",
            ))
            component = KeyboardComponent(
                exp, name, name="start_trigger", allowedKeys="'5'",
                registerOn="press", store="first key", forceEndRoutine=True,
                discardPrev=True, startVal=0, stopVal="",
            )
        elif name == "viewing":
            component = MovieComponent(
                exp, name, name="forest_movie", movie=MOVIE_NAME,
                units="height", size=(16 / 9, 1),
                stopType="duration (s)", stopVal=480,
                forceEndRoutine=True, loop=False, noAudio=False,
            )
        else:
            component = PolygonComponent(
                exp, name, name=name + "_screen", shape="rectangle",
                units="norm", size=(2, 2), fillColor="black", lineColor="black",
                startVal=0, stopType="duration (s)", stopVal=30,
            )
        routine.addComponent(component)
        exp.flow.addRoutine(routine, len(exp.flow))

    path = ROOT / "forest_viewing.psyexp"
    exp.saveToXML(str(path))
    loaded = Experiment()
    loaded.loadFromXML(str(path))
    assert [routine.name for routine in loaded.flow] == ROUTINES
    for name in ["black_start", "black_end"]:
        component = next(c for c in loaded.routines[name] if c.type == "Polygon")
        assert float(component.params["stopVal"].val) == 30
    movie = next(c for c in loaded.routines["viewing"] if c.type == "Movie")
    assert movie.params["movie"].val == MOVIE_NAME
    assert not movie.params["loop"].val
    assert not movie.params["No audio"].val
    assert float(movie.params["stopVal"].val) == 480
    script = loaded.writeScript(expPath=str(path), target="PsychoPy")
    code = script if isinstance(script, str) else script.getvalue()
    code = "\n".join(line.rstrip() for line in code.splitlines()) + "\n"
    compile(code, str(path.with_suffix(".py")), "exec")
    path.with_suffix(".py").write_text(code, encoding="utf-8-sig")
    print("Saved, reloaded and compiled:", path)


if __name__ == "__main__":
    build()
