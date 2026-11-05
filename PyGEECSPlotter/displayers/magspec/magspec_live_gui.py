# Live magspec display: the generic live GUI (PyGEECSPlotter.displayers.live_gui)
# with the magspec settings. The watcher sends each shot as soon as it is
# analysed (MagSpecEBeamAnalyzer.live_payload); the GUI process draws it.
#
#   gui = start_gui()
#   gui.send(payload)        # never blocks; drops the message if the GUI is behind
#   gui.stop()

from PyGEECSPlotter.displayers.live_gui import CMAPS, GEECS_CMAPS, LiveGui, colormap  # noqa: F401
from PyGEECSPlotter.displayers import live_gui

TRENDS = ['charge_pC', 'peakMomentum_GeV/c', 'maxMomentum_GeV/c', 'eYAngle_mrad']


def magspec_options(prefix='py-magspec'):
    """The generic GUI's options for magspec (see live_gui's module comment)."""
    return {
        'settings_name': 'magspec',
        'trend_prefix': prefix,
        'trends': TRENDS,
        'n_images': 1,
        'labels': {'x': 'Momentum [GeV/c]', 'y': 'mrad', 'image': 'pC/mrad/(GeV/c)',
                   'line': 'pC/GeV', 'wf': 'pC/GeV'},
        'x_range': (0.0, 2.5),
        'log_line': True,
        'cmap': 'white_blue_yellow',
        'stats': ('Q = {charge_pC:.3g} pC,  peak = {peakMomentum_GeV/c:.3g} GeV/c,  '
                  'max = {maxMomentum_GeV/c:.3g} GeV/c,  eY = {eYAngle_mrad:.2f} mrad'),
    }


def start_gui(prefix='py-magspec', title='Live magspec', maxsize=16, log_path=None, timeout=60.0):
    """Start the live magspec GUI process; returns a ``LiveGui``. ``prefix``:
    the result-name prefix of the scalars (``'<prefix> charge_pC'``).
    ``log_path``: the GUI appends ``scan, shot, t_sent, t_received, t_drawn``
    per drawn shot."""
    return live_gui.start_gui('magspec', title=title, options=magspec_options(prefix), maxsize=maxsize,
                              log_path=log_path, timeout=timeout)
