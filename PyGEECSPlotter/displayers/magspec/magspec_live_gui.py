# Live magspec display: a PyQt6 window in its own process.
#
# The watcher sends each shot as soon as it is analysed (see
# MagSpecEBeamAnalyzer.live_payload); the GUI process draws it. Rendering
# therefore never slows the analysis, and the window (with empty plots) is up
# before the first shot arrives.
#
#   gui = start_gui()
#   gui.send(payload)        # never blocks; drops the message if the GUI is behind
#   gui.stop()
#
# The GUI runs as ``python -m PyGEECSPlotter.displayers.magspec.magspec_live_gui``
# (so the caller's script is not re-run, as multiprocessing's spawn would) and
# receives messages over a local authenticated socket.
#
# Messages: {'kind': 'shot', ...} (live_payload, plus 't_sent'),
# {'kind': 'status', 'text': ...}, {'kind': 'grab', 'path': ...} (save a
# screenshot), {'kind': 'quit'}. Qt is only imported in the GUI process.

import os
import queue as _queue
import secrets
import subprocess
import sys
import threading
import time

TRENDS = ['charge_pC', 'peakMomentum_GeV/c', 'maxMomentum_GeV/c', 'eYAngle_mrad']
CMAPS = ['jet', 'viridis', 'inferno', 'turbo', 'gray']


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


def start_gui(prefix='py-magspec', title='Live magspec', maxsize=16, log_path=None, timeout=60.0):
    """Start the GUI process and return a ``LiveGui``. ``prefix``: the
    result-name prefix of the scalars (``'<prefix> charge_pC'``).
    ``log_path``: if given, the GUI appends ``scan, shot, t_sent,
    t_received, t_drawn`` per drawn shot."""
    from multiprocessing.connection import Listener
    key = secrets.token_bytes(16)
    listener = Listener(('127.0.0.1', 0), authkey=key)
    # the GUI must import this same PyGEECSPlotter (e.g. a checkout), not an installed copy
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (repo, os.environ.get('PYTHONPATH')) if p))
    args = [sys.executable, '-m', 'PyGEECSPlotter.displayers.magspec.magspec_live_gui',
            str(listener.address[1]), key.hex(), prefix, title, log_path or '']
    process = subprocess.Popen(args, env=env)
    conn = [None]
    t = threading.Thread(target=lambda: conn.__setitem__(0, listener.accept()), daemon=True)
    t.start()
    t.join(timeout)
    listener.close()
    if conn[0] is None:
        process.kill()
        raise RuntimeError(f'the live GUI did not start within {timeout:g} s')
    return LiveGui(process, conn[0], maxsize)


def _run(port, key_hex, prefix, title, log_path):
    """The GUI process: connect back, read messages in a thread, run the window."""
    from multiprocessing.connection import Client
    conn = Client(('127.0.0.1', int(port)), authkey=bytes.fromhex(key_hex))
    q = _queue.Queue()

    def reader():
        while True:
            try:
                q.put(conn.recv())
            except (EOFError, OSError):
                q.put({'kind': 'quit'})
                return
    threading.Thread(target=reader, daemon=True).start()
    import matplotlib
    matplotlib.use('QtAgg')
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = _window_class()(q, prefix, title, log_path or None)
    window.show()
    app.exec()


def _window_class():
    """The window class (built here so Qt is only imported in the GUI process)."""
    import numpy as np
    from PyQt6 import QtCore, QtWidgets
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    class LiveMagspecWindow(QtWidgets.QMainWindow):
        def __init__(self, q, prefix, title, log_path):
            super().__init__()
            self.q, self.prefix, self.log_path = q, prefix, log_path
            self.setWindowTitle(title)
            self.settings = QtCore.QSettings('BELLA', 'PyGEECSPlotter-live-magspec')
            self.scan = None
            self.latest = None              # newest shot message
            self.rows = {}                  # shot -> scalars
            self.spectra = {}               # shot -> spectrum on the fixed grid
            self.grid = None
            self.dirty_agg = False
            self.last_agg = 0.0
            self.trend_cols = None
            self._build_plots()
            self._build_controls()
            self.status = QtWidgets.QLabel('waiting for the first shot')
            self.statusBar().addWidget(self.status)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._tick)
            self.timer.start(30)
            self.applied = {}               # colormap / scale last set (re-applying them is slow)
            # lay the figures out once now, then keep that layout: re-running
            # constrained layout on every draw costs ~0.1 s per shot
            QtCore.QTimer.singleShot(0, self._freeze_layouts)
            geometry = self.settings.value('geometry')
            if geometry is not None:
                self.restoreGeometry(geometry)
            else:
                self.resize(1500, 850)

        # ------------------------------------------------------------ layout
        def _canvas(self, figsize):
            fig = Figure(figsize=figsize, constrained_layout=True)
            return fig, FigureCanvasQTAgg(fig)

        def _build_plots(self):
            self.fig_shot, self.can_shot = self._canvas((8, 6))
            self.ax_img, self.ax_sp = self.fig_shot.subplots(
                2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            self.ax_img.set_ylabel('mrad')
            self.ax_sp.set_xlabel('GeV/c')
            self.ax_sp.set_ylabel('pC/GeV')
            self.im = self.ax_img.imshow(np.zeros((2, 2)), aspect='auto', origin='lower',
                                         interpolation='nearest', extent=(0, 5, -1.3, 1.3))
            self.fig_shot.colorbar(self.im, ax=self.ax_img, label='pC/mrad/(GeV/c)')
            self.line_sp, = self.ax_sp.plot([], [], 'r-', lw=1)
            self.fig_tr, self.can_tr = self._canvas((6, 4))
            self.fig_wf, self.can_wf = self._canvas((6, 4))
            self.ax_wf = self.fig_wf.add_subplot()
            self.im_wf = self.ax_wf.imshow(np.zeros((2, 2)), aspect='auto', origin='lower',
                                           interpolation='nearest', extent=(0, 5, 0.5, 1.5))
            self.fig_wf.colorbar(self.im_wf, ax=self.ax_wf, label='pC/GeV')
            self.ax_wf.set_xlabel('Momentum [GeV/c]')
            self.ax_wf.set_ylabel('shot')
            right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            right.addWidget(self.can_tr)
            right.addWidget(self.can_wf)
            split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            split.addWidget(self.can_shot)
            split.addWidget(right)
            split.setSizes([900, 600])
            self.setCentralWidget(split)

        def _spin(self, lo, hi, value, step, decimals=3):
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setDecimals(decimals)
            w.setSingleStep(step)
            w.setValue(value)
            w.valueChanged.connect(self._controls_changed)
            return w

        def _check(self, text, value):
            w = QtWidgets.QCheckBox(text)
            w.setChecked(value)
            w.toggled.connect(self._controls_changed)
            return w

        def _build_controls(self):
            s = self.settings
            f = lambda k, d: float(s.value(k, d))
            b = lambda k, d: str(s.value(k, d)).lower() in ('true', '1')
            self.c = {
                'pmin': self._spin(0, 20, f('pmin', 0.0), 0.1),
                'pmax': self._spin(0, 20, f('pmax', 2.5), 0.1),
                'img_auto': self._check('auto', b('img_auto', True)),
                'vmin': self._spin(0, 1e6, f('vmin', 0.0), 1.0),
                'vmax': self._spin(0, 1e6, f('vmax', 100.0), 1.0),
                'log': self._check('log', b('log', True)),
                'y_auto': self._check('auto', b('y_auto', True)),
                'ymin': self._spin(1e-6, 1e6, f('ymin', 1e-2), 0.1, 4),
                'ymax': self._spin(1e-6, 1e6, f('ymax', 100.0), 1.0, 4),
                'wf_auto': self._check('auto', b('wf_auto', True)),
                'wf_vmax': self._spin(0, 1e6, f('wf_vmax', 50.0), 1.0),
                'pause': self._check('pause', False),
            }
            self.cmap = QtWidgets.QComboBox()
            self.cmap.addItems(CMAPS)
            self.cmap.setCurrentText(str(s.value('cmap', 'jet')))
            self.cmap.currentTextChanged.connect(self._controls_changed)
            self.last_n = QtWidgets.QSpinBox()
            self.last_n.setRange(0, 100000)
            self.last_n.setValue(int(s.value('last_n', 0)))
            self.last_n.setSpecialValueText('all')
            self.last_n.valueChanged.connect(self._controls_changed)
            self.trends = QtWidgets.QListWidget()
            chosen = s.value('trends', TRENDS)
            for name in (chosen if isinstance(chosen, list) else TRENDS):
                self._add_trend(name, True)
            self.trends.itemChanged.connect(self._trends_changed)

            form = QtWidgets.QFormLayout()
            row = lambda *ws: self._row(*ws)
            form.addRow('Momentum min / max [GeV/c]', row(self.c['pmin'], self.c['pmax']))
            form.addRow('Image colour', row(self.c['img_auto'], self.c['vmin'], self.c['vmax']))
            form.addRow('Colormap', self.cmap)
            form.addRow('Spectrum', row(self.c['log'], self.c['y_auto'], self.c['ymin'], self.c['ymax']))
            form.addRow('Waterfall colour max', row(self.c['wf_auto'], self.c['wf_vmax']))
            form.addRow('Waterfall shots (last N)', self.last_n)
            form.addRow('Trends', self.trends)
            form.addRow('', self.c['pause'])
            panel = QtWidgets.QWidget()
            panel.setLayout(form)
            dock = QtWidgets.QDockWidget('Controls', self)
            dock.setWidget(panel)
            dock.setMaximumWidth(340)
            self.trends.setMaximumHeight(260)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        def _row(self, *widgets):
            w = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            for x in widgets:
                lay.addWidget(x)
            return w

        def _add_trend(self, name, checked):
            if self.trends.findItems(name, QtCore.Qt.MatchFlag.MatchExactly):
                return
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
            self.trends.addItem(item)

        # ---------------------------------------------------------- messages
        def _tick(self):
            """Drain the queue; draw the newest shot; redraw trends/waterfall at most 2x/s."""
            newest = None
            while True:
                try:
                    msg = self.q.get_nowait()
                except _queue.Empty:
                    break
                except (EOFError, OSError):
                    self.close()
                    return
                kind = msg.get('kind')
                if kind == 'quit':
                    self.close()
                    return
                if kind == 'status':
                    self.status.setText(msg.get('text', ''))
                elif kind == 'grab':
                    self._draw_aggregates()
                    self.grab().save(msg['path'])
                elif kind == 'shot':
                    msg['t_received'] = time.time()
                    self._store(msg)
                    newest = msg
            if newest is not None and not self.c['pause'].isChecked():
                self.latest = newest
                self._draw_shot(newest)
            elif newest is not None:
                self.latest = newest
            if self.dirty_agg and not self.c['pause'].isChecked() and time.monotonic() - self.last_agg > 0.5:
                self._draw_aggregates()

        def _store(self, msg):
            if msg['scan'] != self.scan:            # a new scan: start the trends / waterfall afresh
                self.scan = msg['scan']
                self.rows.clear()
                self.spectra.clear()
                self.setWindowTitle(f'Live magspec: Scan{self.scan:03d}')
            self.rows[msg['shot']] = msg.get('results', {})
            if msg.get('image') is not None and len(msg.get('grid_spectrum', ())):
                self.spectra[msg['shot']] = msg['grid_spectrum']
                self.grid = msg['grid']
            for name in msg.get('results', {}):     # offer every scalar as a trend
                if name.startswith(self.prefix + ' '):
                    self._add_trend(name[len(self.prefix) + 1:], False)
            self.dirty_agg = True

        # ----------------------------------------------------------- drawing
        def _draw_shot(self, msg):
            r = msg.get('results', {})
            g = lambda k: r.get(f'{self.prefix} {k}', np.nan)
            self.ax_img.set_title(f"Scan{msg['scan']:03d} shot {msg['shot']}", fontsize=10)
            self.ax_sp.set_title(f"Q = {g('charge_pC'):.3g} pC,  peak = {g('peakMomentum_GeV/c'):.3g} GeV/c,  "
                                 f"max = {g('maxMomentum_GeV/c'):.3g} GeV/c,  eY = {g('eYAngle_mrad'):.2f} mrad",
                                 fontsize=9)
            if msg.get('image') is not None:
                img = msg['image']
                self.im.set_data(img)
                self.im.set_extent(msg['extent'])
                c = self.c
                if c['img_auto'].isChecked():
                    self.im.set_clim(0, float(np.nanmax(img)) or 1.0)
                else:
                    self.im.set_clim(c['vmin'].value(), c['vmax'].value())
                if self.applied.get('cmap') != self.cmap.currentText():
                    self.im.set_cmap(self.cmap.currentText())
                    self.im_wf.set_cmap(self.cmap.currentText())
                    self.applied['cmap'] = self.cmap.currentText()
                spec = np.where(msg['spectrum'] > 0, msg['spectrum'], np.nan) if c['log'].isChecked() \
                    else msg['spectrum']
                self.line_sp.set_data(msg['mmt'], spec)
                scale = 'log' if c['log'].isChecked() else 'linear'
                if self.applied.get('scale') != scale:
                    self.ax_sp.set_yscale(scale)
                    self.applied['scale'] = scale
                self.ax_img.set_ylim(msg['extent'][2], msg['extent'][3])
                self.ax_sp.set_xlim(c['pmin'].value(), c['pmax'].value())
                if c['y_auto'].isChecked():
                    sel = (msg['mmt'] >= c['pmin'].value()) & (msg['mmt'] <= c['pmax'].value())
                    vals = spec[sel] if sel.any() else spec
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        lo, hi = float(vals.min()), float(vals.max())
                        if c['log'].isChecked():
                            self.ax_sp.set_ylim(max(lo, hi * 1e-5) / 1.5, hi * 1.5)
                        else:
                            self.ax_sp.set_ylim(0, hi * 1.05 or 1.0)
                else:
                    self.ax_sp.set_ylim(c['ymin'].value(), c['ymax'].value())
            self.can_shot.draw()
            msg['t_drawn'] = time.time()
            if self.log_path:
                with open(self.log_path, 'a') as f:
                    f.write(f"{msg['scan']},{msg['shot']},{msg.get('t_sent', np.nan)},"
                            f"{msg['t_received']},{msg['t_drawn']}\n")

        def _freeze_layouts(self):
            # also apply the colormap / scale now, so the first shot doesn't pay for it
            cmap, scale = self.cmap.currentText(), 'log' if self.c['log'].isChecked() else 'linear'
            self.im.set_cmap(cmap)
            self.im_wf.set_cmap(cmap)
            self.ax_sp.set_yscale(scale)
            self.applied.update(cmap=cmap, scale=scale)
            for fig, canvas in ((self.fig_shot, self.can_shot), (self.fig_wf, self.can_wf)):
                canvas.draw()
                fig.set_layout_engine('none')

        def _checked_trends(self):
            return [self.trends.item(i).text() for i in range(self.trends.count())
                    if self.trends.item(i).checkState() == QtCore.Qt.CheckState.Checked]

        def _draw_aggregates(self):
            self.dirty_agg = False
            self.last_agg = time.monotonic()
            shots = sorted(self.rows)
            cols = self._checked_trends()
            if cols != self.trend_cols:             # the selection changed: new axes
                self.trend_cols = cols
                self.fig_tr.clf()
                self.ax_tr = list(np.atleast_1d(self.fig_tr.subplots(max(len(cols), 1), 1, sharex=True)))
                self.lines_tr = [a.plot([], [], '.-', ms=4, lw=0.6)[0] for a in self.ax_tr]
                for a, col in zip(self.ax_tr, cols):
                    a.set_ylabel(col, fontsize=8)
                self.ax_tr[-1].set_xlabel('shot')
                self.fig_tr.set_layout_engine('constrained')
                self.tr_frozen = False
            for a, line, col in zip(self.ax_tr, self.lines_tr, cols):
                line.set_data(shots, [self.rows[s].get(f'{self.prefix} {col}', np.nan) for s in shots])
                a.relim()
                a.autoscale_view()
            self.can_tr.draw()
            if not getattr(self, 'tr_frozen', True):
                self.fig_tr.set_layout_engine('none')
                self.tr_frozen = True
            wf = sorted(self.spectra)
            n = self.last_n.value()
            if n:
                wf = wf[-n:]
            if wf and self.grid is not None and len(self.grid):
                stack = np.vstack([self.spectra[s] for s in wf])
                self.im_wf.set_data(stack)
                self.im_wf.set_extent((self.grid[0], self.grid[-1], wf[0] - 0.5, wf[-1] + 0.5))
                vmax = (float(np.nanpercentile(stack, 99.5)) or 1.0) if self.c['wf_auto'].isChecked() \
                    else self.c['wf_vmax'].value()
                self.im_wf.set_clim(0, vmax)
                self.ax_wf.set_xlim(self.c['pmin'].value(), self.c['pmax'].value())
                self.ax_wf.set_ylim(wf[0] - 0.5, wf[-1] + 0.5)
                self.ax_wf.set_title(f'Scan{self.scan:03d}: {len(wf)} shots', fontsize=9)
            self.can_wf.draw()

        def _controls_changed(self, *_):
            if self.latest is not None and not self.c['pause'].isChecked():
                self._draw_shot(dict(self.latest, t_received=time.time()))
            self.dirty_agg = True

        def _trends_changed(self, *_):
            self.trend_cols = None                  # rebuild the trend axes
            self.dirty_agg = True

        def closeEvent(self, event):
            s = self.settings
            for k, w in self.c.items():
                if k != 'pause':
                    s.setValue(k, w.isChecked() if isinstance(w, QtWidgets.QCheckBox) else w.value())
            s.setValue('cmap', self.cmap.currentText())
            s.setValue('last_n', self.last_n.value())
            s.setValue('trends', self._checked_trends())
            s.setValue('geometry', self.saveGeometry())
            QtWidgets.QApplication.instance().quit()
            super().closeEvent(event)

    return LiveMagspecWindow


if __name__ == '__main__':
    _run(*sys.argv[1:6])
