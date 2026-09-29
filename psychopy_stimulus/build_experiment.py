"""Generate the editable PsychoPy Builder experiment with installed PsychoPy."""
from pathlib import Path
from psychopy.experiment import Experiment
from psychopy.experiment.components.movie import MovieComponent
from psychopy.experiment.components.polygon import PolygonComponent
from psychopy.experiment.components.keyboard import KeyboardComponent

root = Path(__file__).resolve().parent
exp = Experiment()
settings = {
    'expName': 'Stimulus repeated viewing',
    'Experiment info': "{'participant': '', 'repetitions': [2, 3]}",
    'Full-screen window': True,
    'Show info dlg': True,
    'Enable Escape': True,
    'Show mouse': False,
    'measureFrameRate': False,
    'frameRate': 60,
    'Units': 'height',
    'color': 'black',
    'Data filename': "'data/%s_%s' % (expInfo['participant'], expInfo['date'])",
}
for key, value in settings.items():
    exp.settings.params[key].val = value

def add_black_screen(name, position):
    black = exp.addRoutine(name)
    black.addComponent(PolygonComponent(
        exp, name, name=name + '_screen', shape='rectangle',
        units='norm', size=(2, 2), fillColor='black', lineColor='black',
        startVal=0, stopType='duration (s)', stopVal=30,
    ))
    exp.flow.addRoutine(black, position)

wait = exp.addRoutine('wait_for_5')
wait.addComponent(KeyboardComponent(
    exp, 'wait_for_5', name='start_trigger', allowedKeys="'5'",
    registerOn='press', store='first key', forceEndRoutine=True,
    discardPrev=True, startVal=0, stopVal='',
))
exp.flow.addRoutine(wait, 0)
add_black_screen('black_start', 1)
routine = exp.addRoutine('viewing')
movie = MovieComponent(
    exp, 'viewing', name='stimulus',
    movie="$'stimulus_twice.mp4' if int(expInfo['repetitions']) == 2 else 'stimulus_three_times.mp4'",
    units='height', size=(1.7777777778, 1),
    stopVal='', forceEndRoutine=True, loop=False, noAudio=False,
)
routine.addComponent(movie)
exp.flow.addRoutine(routine, 2)
add_black_screen('black_end', 3)
path = root / 'repeated_viewing.psyexp'
exp.saveToXML(str(path))
loaded = Experiment()
loaded.loadFromXML(str(path))
assert len(loaded.routines['viewing']) >= 1
assert [r.name for r in loaded.flow] == ['wait_for_5', 'black_start', 'viewing', 'black_end']
for name in ['black_start', 'black_end']:
    screen = next(c for c in loaded.routines[name] if c.type == 'Polygon')
    assert float(screen.params['stopVal'].val) == 30
script = loaded.writeScript(expPath=str(path), target='PsychoPy')
code = script if isinstance(script, str) else script.getvalue()
compile(code, str(path.with_suffix('.py')), 'exec')
path.with_suffix('.py').write_text(code, encoding='utf-8-sig')
print('Saved and compiled:', path)
