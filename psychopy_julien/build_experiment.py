"""Create an editable PsychoPy experiment for the trimmed Julien movie."""
from pathlib import Path
from psychopy.experiment import Experiment
from psychopy.experiment.components.movie import MovieComponent
from psychopy.experiment.components.polygon import PolygonComponent
from psychopy.experiment.components.keyboard import KeyboardComponent

root = Path(__file__).resolve().parent
exp = Experiment()
for key, value in {
    'expName': 'Julien 01m00s to 10m05s',
    'Experiment info': "{'participant': ''}",
    'Full-screen window': True,
    'Show info dlg': True,
    'Enable Escape': True,
    'Show mouse': False,
    'measureFrameRate': False,
    'frameRate': 60,
    'Units': 'height',
    'color': 'black',
    'Data filename': "'data/%s_%s' % (expInfo['participant'], expInfo['date'])",
}.items():
    exp.settings.params[key].val = value

for name in ['wait_for_5', 'black_start', 'viewing', 'black_end']:
    routine = exp.addRoutine(name)
    if name == 'wait_for_5':
        component = KeyboardComponent(
            exp, name, name='start_trigger', allowedKeys="'5'",
            registerOn='press', store='first key', forceEndRoutine=True,
            discardPrev=True, startVal=0, stopVal='',
        )
    elif name == 'viewing':
        component = MovieComponent(
            exp, name, name='julien_movie',
            movie='julien_01m00s_to_10m05s.mp4',
            units='height', size=(16/9, 1),
            stopVal='', forceEndRoutine=True, loop=False, noAudio=False,
        )
    else:
        component = PolygonComponent(
            exp, name, name=name + '_screen', shape='rectangle',
            units='norm', size=(2, 2), fillColor='black', lineColor='black',
            startVal=0, stopType='duration (s)', stopVal=30,
        )
    routine.addComponent(component)
    exp.flow.addRoutine(routine, len(exp.flow))

path = root / 'julien_viewing.psyexp'
exp.saveToXML(str(path))
loaded = Experiment()
loaded.loadFromXML(str(path))
assert [r.name for r in loaded.flow] == ['wait_for_5', 'black_start', 'viewing', 'black_end']
for name in ['black_start', 'black_end']:
    component = next(c for c in loaded.routines[name] if c.type == 'Polygon')
    assert float(component.params['stopVal'].val) == 30
movie = next(c for c in loaded.routines['viewing'] if c.type == 'Movie')
assert not movie.params['loop'].val
assert movie.params['movie'].val == 'julien_01m00s_to_10m05s.mp4'
script = loaded.writeScript(expPath=str(path), target='PsychoPy')
code = script if isinstance(script, str) else script.getvalue()
compile(code, str(path.with_suffix('.py')), 'exec')
path.with_suffix('.py').write_text(code, encoding='utf-8-sig')
print('Saved, reloaded and compiled:', path)
