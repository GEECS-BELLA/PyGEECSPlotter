# Live display: a PyQt6 window in its own process, fed one message per shot.
#
# The watcher sends each shot as soon as it is analysed (the analyzer's
# live_payload); the GUI process draws it, so rendering never slows the
# analysis, and the window (with empty plots) is up before the first shot.
#
#   gui = start_gui('CAM-HPD-CCD', title='Live CAM-HPD-CCD')
#   gui.send(payload)        # never blocks; drops the message if the GUI is behind
#   gui.stop()
#
# The GUI runs as ``python -m PyGEECSPlotter.displayers.live_gui`` (not
# multiprocessing's spawn, which re-runs the caller's script) with PYTHONPATH
# set to this package's repo, and receives messages over a local
# authenticated socket. Qt is only imported in the GUI process.
#
# Messages (dicts, 'kind' first):
#   shot:   scan, shot, and optionally
#           images  [{'name', 'image' (2-D float32), 'extent'}], or image / extent
#           line    {'x', 'y'}       this shot's 1-D view (spectrum, lineout)
#           wf      {'x', 'y'}       its waterfall row (a fixed grid is best)
#           results {name: float}    scalars (trends)
#           context {name: float}    .tdms device scalars (trends)
#           param   {'name', 'value'} scan parameter (trend x-axis)
#           t_sent
#   status: text (status bar)
#   grab:   path (save a screenshot)
#   quit
#
# Window options (``options`` of start_gui; a preset such as magspec's sets
# them): trends (default trend names), trend_prefix (prefix stripped from
# result names), labels {'x', 'y', 'image', 'line', 'wf'}, x_range (fixed
# x min/max), log_line, n_images (image panels to lay out at start-up), stats
# (title format of the line panel, e.g.
# 'Q = {charge_pC:.3g} pC'), settings_name (QSettings key).

import json
import os
import queue as _queue
import secrets
import subprocess
import sys
import threading
import time

# geecs_cmaps factories first, then matplotlib names
GEECS_CMAPS = ['white_blue_yellow', 'white_blue_yellow_black', 'enhance_low', 'bwr_enhance_low',
               'blue_yellow_red', 'inferno_pic', 'wby_grey']
CMAPS = GEECS_CMAPS + ['jet', 'viridis', 'inferno', 'turbo', 'gray', 'RdBu']


def colormap(name, _cache={}):
    """A matplotlib colormap: one of PyGEECSPlotter.utils.geecs_cmaps, else a matplotlib name."""
    if name in GEECS_CMAPS:
        if name not in _cache:
            import PyGEECSPlotter.utils.geecs_cmaps as gcm
            _cache[name] = getattr(gcm, name)()
        return _cache[name]
    return name


class LiveGui:
    """Handle on the GUI process (see ``start_gui``)."""

    def __init__(self, process, conn, maxsize):
        self.process = process
        self._conn = conn
        self._out = _queue.Queue(maxsize)
        self.dropped = 0
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        while True:
            msg = self._out.get()
            if msg is None:
                return
            try:
                self._conn.send(msg)
            except (OSError, EOFError, ValueError):
                return                      # the window was closed

    def alive(self):
        return self.process.poll() is None and self._thread.is_alive()

    def send(self, msg):
        """Queue ``msg`` for the GUI without blocking. False if it was dropped."""
        if not self.alive():
            return False
        try:
            self._out.put_nowait(msg)
            return True
        except _queue.Full:
            self.dropped += 1
            return False

    def stop(self, timeout=5.0):
        self.send({'kind': 'quit'})
        try:
            self._out.put(None, timeout=1.0)
        except _queue.Full:
            pass
        self._thread.join(timeout)
        try:
            self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
        try:
            self._conn.close()
        except OSError:
            pass


def package_env():
    """os.environ with PYTHONPATH starting at this PyGEECSPlotter's repo, so a
    child process imports this same copy (e.g. a checkout), not an installed one."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (repo, os.environ.get('PYTHONPATH')) if p))


def start_gui(name='live', title='Live analysis', options=None, maxsize=16, log_path=None, timeout=60.0):
    """Start the GUI process and return a ``LiveGui``. ``name``: the
    analysis name (window title, settings key); ``options``: see the module
    comment. ``log_path``: the GUI appends ``scan, shot, t_sent, t_received,
    t_drawn`` per drawn shot."""
    from multiprocessing.connection import Listener
    key = secrets.token_bytes(16)
    listener = Listener(('127.0.0.1', 0), authkey=key)
    options = dict(options or {})
    options.setdefault('settings_name', name)
    args = [sys.executable, '-m', 'PyGEECSPlotter.displayers.live_gui', str(listener.address[1]),
            key.hex(), title, json.dumps(options), log_path or '']
    process = subprocess.Popen(args, env=package_env())
    conn = [None]
    t = threading.Thread(target=lambda: conn.__setitem__(0, listener.accept()), daemon=True)
    t.start()
    t.join(timeout)
    listener.close()
    if conn[0] is None:
        process.kill()
        raise RuntimeError(f'the live GUI did not start within {timeout:g} s')
    return LiveGui(process, conn[0], maxsize)


def _run(port, key_hex, title, options_json, log_path):
    """The GUI process: connect back, read messages in a thread, run the window."""
    from multiprocessing.connection import Client
    conn = Client(('127.0.0.1', int(port)), authkey=bytes.fromhex(key_hex))
    q = _queue.Queue()

    def reader():
        while True:
            try:
                msg = conn.recv()
                if isinstance(msg, dict):
                    msg['t_received'] = time.time()     # off the pipe (the window draws later)
                q.put(msg)
            except (EOFError, OSError):
                q.put({'kind': 'quit'})
                return
    threading.Thread(target=reader, daemon=True).start()
    import matplotlib
    matplotlib.use('QtAgg')
    from PyQt6 import QtWidgets
    from PyGEECSPlotter.displayers.live_gui_window import window_class
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = window_class()(q, title, json.loads(options_json), log_path or None)
    window.show()
    app.exec()


if __name__ == '__main__':
    _run(*sys.argv[1:6])
