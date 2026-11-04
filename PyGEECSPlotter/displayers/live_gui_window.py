# The live GUI window (runs in the GUI process; see live_gui.py). Generalised
# from the magspec live window: left, the latest shot (its images side by side,
# its 1-D view below); right, trends (top) and a waterfall (bottom); a dock of
# controls; a status bar. Artists are updated in place and the layouts frozen
# after the first draw, so a shot draws in well under 0.1 s.

import queue as _queue
import time

from PyGEECSPlotter.displayers.live_gui import CMAPS, colormap

MAX_IMAGES = 4


class _Missing(dict):
    """For the stats title: a result the shot doesn't have formats as NaN."""

    def __missing__(self, key):
        return float('nan')


def window_class():
    """The window class (built here so Qt is only imported in the GUI process)."""
    import numpy as np
    from PyQt6 import QtCore, QtWidgets
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    class LiveWindow(QtWidgets.QMainWindow):
        def __init__(self, q, title, options, log_path):
            super().__init__()
            self.q, self.opt, self.log_path = q, options, log_path
            self.title = title
            self.setWindowTitle(title)
            self.settings = QtCore.QSettings('BELLA', f"PyGEECSPlotter-live-{options.get('settings_name', 'live')}")
            self.prefix = options.get('trend_prefix') or ''
            self.labels = dict({'x': '', 'y': '', 'image': '', 'line': '', 'wf': ''}, **options.get('labels', {}))
            self.scan = None
            self.latest = None              # newest shot message
            self.rows = {}                  # shot -> {trend name: value}
            self.params = {}                # shot -> scan parameter value
            self.param_name = ''
            self.wf = {}                    # shot -> (x, y)
            self.dirty_agg = False
            self.last_agg = 0.0
            self.trend_key = None
            self.applied = {}               # colormap / scale last set (re-applying them is slow)
            self.n_images = 0
            self._build_plots()
            self._build_controls()
            self.status = QtWidgets.QLabel('waiting for the first shot')
            self.statusBar().addWidget(self.status)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._tick)
            self.timer.start(30)
            QtCore.QTimer.singleShot(0, self._first_draw)
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
            self._shot_axes(max(1, min(MAX_IMAGES, int(self.opt.get('n_images', 1)))))
            self.fig_tr, self.can_tr = self._canvas((6, 4))
            self.fig_wf, self.can_wf = self._canvas((6, 4))
            self.ax_wf = self.fig_wf.add_subplot()
            self.im_wf = self.ax_wf.imshow(np.zeros((2, 2)), aspect='auto', origin='lower',
                                           interpolation='nearest', extent=(0, 1, 0.5, 1.5))
            self.fig_wf.colorbar(self.im_wf, ax=self.ax_wf, label=self.labels['wf'])
            self.ax_wf.set_xlabel(self.labels['x'])
            self.ax_wf.set_ylabel('shot')
            self.ax_wf.set_title(' ', fontsize=9)
            right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            right.addWidget(self.can_tr)
            right.addWidget(self.can_wf)
            split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            split.addWidget(self.can_shot)
            split.addWidget(right)
            split.setSizes([1000, 600])
            self.setCentralWidget(split)

        def _shot_axes(self, n):
            """The latest-shot figure for ``n`` images (and the 1-D view below)."""
            self.n_images = n
            self.fig_shot.clf()
            self.fig_shot.set_layout_engine('constrained')
            grid = self.fig_shot.add_gridspec(2, n, height_ratios=[1, 1])
            self.ax_img, self.ims = [], []
            for i in range(n):
                ax = self.fig_shot.add_subplot(grid[0, i])
                im = ax.imshow(np.zeros((2, 2)), aspect='auto', origin='lower', interpolation='nearest')
                self.fig_shot.colorbar(im, ax=ax, label=self.labels['image'] if i == n - 1 else '')
                ax.set_ylabel(self.labels['y'] if i == 0 else '')
                ax.set_title('waiting for the first shot' if i == 0 else ' ', fontsize=10)
                self.ax_img.append(ax)
                self.ims.append(im)
            self.ax_line = self.fig_shot.add_subplot(grid[1, :])
            self.ax_line.set_title(' ', fontsize=9)
            self.ax_line.set_xlabel(self.labels['x'])
            self.ax_line.set_ylabel(self.labels['line'])
            self.line, = self.ax_line.plot([], [], 'r-', lw=1)
            self.applied.pop('cmap', None)
            self.applied.pop('scale', None)
            self.shot_frozen = False

        def _spin(self, lo, hi, value, step, decimals=3):
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setDecimals(decimals)
            w.setSingleStep(step)
            w.setValue(value)
            w.setMinimumWidth(110)
            w.valueChanged.connect(self._controls_changed)
            return w

        def _check(self, text, value):
            w = QtWidgets.QCheckBox(text)
            w.setChecked(value)
            w.toggled.connect(self._controls_changed)
            return w

        def _build_controls(self):
            s, o = self.settings, self.opt
            f = lambda k, d: float(s.value(k, d))
            b = lambda k, d: str(s.value(k, d)).lower() in ('true', '1')
            x0, x1 = o.get('x_range') or (0.0, 1.0)
            self.c = {
                'x_auto': self._check('auto', b('x_auto', o.get('x_range') is None)),
                'xmin': self._spin(-1e6, 1e6, f('xmin', x0), 0.1),
                'xmax': self._spin(-1e6, 1e6, f('xmax', x1), 0.1),
                'img_auto': self._check('auto (per shot)', b('img_auto', True)),
                'vmin': self._spin(-1e9, 1e9, f('vmin', 0.0), 1.0),
                'vmax': self._spin(-1e9, 1e9, f('vmax', 100.0), 1.0),
                'log': self._check('log', b('log', bool(o.get('log_line', False)))),
                'y_auto': self._check('auto y', b('y_auto', True)),
                'ymin': self._spin(-1e9, 1e9, f('ymin', 1e-2), 0.1, 4),
                'ymax': self._spin(-1e9, 1e9, f('ymax', 100.0), 1.0, 4),
                'wf_auto': self._check('auto', b('wf_auto', True)),
                'wf_vmax': self._spin(-1e9, 1e9, f('wf_vmax', 50.0), 1.0),
                'pause': self._check('pause', False),
            }
            self.cmap = QtWidgets.QComboBox()
            self.cmap.addItems(CMAPS)
            self.cmap.setCurrentText(str(s.value('cmap', o.get('cmap', 'white_blue_yellow'))))
            self.cmap.currentTextChanged.connect(self._controls_changed)
            self.last_n = QtWidgets.QSpinBox()
            self.last_n.setRange(0, 100000)
            self.last_n.setValue(int(s.value('last_n', 0)))
            self.last_n.setSpecialValueText('all')
            self.last_n.valueChanged.connect(self._controls_changed)
            self.trend_x = QtWidgets.QComboBox()
            self.trend_x.addItems(['shot', 'scan parameter'])
            self.trend_x.setCurrentText(str(s.value('trend_x', 'shot')))
            self.trend_x.currentTextChanged.connect(self._trends_changed)
            self.trend_filter = QtWidgets.QLineEdit()
            self.trend_filter.setPlaceholderText('filter')
            self.trend_filter.textChanged.connect(self._filter_trends)
            self.trends = QtWidgets.QListWidget()
            chosen = s.value('trends', o.get('trends', []))
            for name in (chosen if isinstance(chosen, list) else o.get('trends', [])):
                self._add_trend(name, True)
            self.trends.itemChanged.connect(self._trends_changed)

            form = QtWidgets.QFormLayout()
            row = self._row
            form.addRow(f"x ({self.labels['x'] or 'all plots'})", self.c['x_auto'])
            form.addRow('    min / max', row(self.c['xmin'], self.c['xmax']))
            form.addRow('Image color', self.c['img_auto'])
            form.addRow('    min / max', row(self.c['vmin'], self.c['vmax']))
            form.addRow('Colormap', self.cmap)
            form.addRow('1-D view', row(self.c['log'], self.c['y_auto']))
            form.addRow('    y min / max', row(self.c['ymin'], self.c['ymax']))
            form.addRow('Waterfall color max', row(self.c['wf_auto'], self.c['wf_vmax']))
            form.addRow('Waterfall shots (last N)', self.last_n)
            form.addRow('Trends against', self.trend_x)
            form.addRow('Trends', self.trend_filter)
            form.addRow(self.trends)
            form.addRow('', self.c['pause'])
            panel = QtWidgets.QWidget()
            panel.setLayout(form)
            dock = QtWidgets.QDockWidget('Controls', self)
            dock.setWidget(panel)
            dock.setMaximumWidth(420)
            self.trends.setMinimumHeight(160)
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
            text = self.trend_filter.text().lower()
            item.setHidden(bool(text) and text not in name.lower())

        def _filter_trends(self, text):
            text = text.lower()
            for i in range(self.trends.count()):
                item = self.trends.item(i)
                item.setHidden(bool(text) and text not in item.text().lower()
                               and item.checkState() != QtCore.Qt.CheckState.Checked)

        # ---------------------------------------------------------- messages
        def _tick(self):
            """Drain the queue; draw the newest shot; redraw trends / waterfall at most 2x/s."""
            newest = None
            while True:
                try:
                    msg = self.q.get_nowait()
                except _queue.Empty:
                    break
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
            paused = self.c['pause'].isChecked()
            if newest is not None:
                self.latest = newest
                if not paused:
                    self._draw_shot(newest)
            if self.dirty_agg and not paused and time.monotonic() - self.last_agg > 0.5:
                self._draw_aggregates()

        def _store(self, msg):
            if msg['scan'] != self.scan:            # a new scan: start trends / waterfall afresh
                self.scan = msg['scan']
                self.rows.clear()
                self.params.clear()
                self.wf.clear()
                self.setWindowTitle(f"{self.title}: Scan{self.scan:03d}")
            first = not self.rows
            row = {}
            for name, v in (msg.get('results') or {}).items():
                if self.prefix and name.startswith(self.prefix + ' '):
                    name = name[len(self.prefix) + 1:]
                row[name] = v
                self._add_trend(name, False)        # results offered as trends first
            context = msg.get('context') or {}
            row.update(context)
            if first:                               # then the .tdms scalars, once per scan
                for name in context:
                    self._add_trend(name, False)
            self.rows[msg['shot']] = row
            param = msg.get('param') or {}
            if param:
                self.params[msg['shot']] = param.get('value', np.nan)
                self.param_name = param.get('name', '')
            wf = msg.get('wf')
            if wf is not None and len(wf.get('y', ())):
                self.wf[msg['shot']] = (np.asarray(wf['x'], float), np.asarray(wf['y'], float))
            self.dirty_agg = True

        # ----------------------------------------------------------- drawing
        @staticmethod
        def _images(msg):
            if msg.get('images'):
                return msg['images'][:MAX_IMAGES]
            if msg.get('image') is not None:
                return [{'name': '', 'image': msg['image'], 'extent': msg.get('extent')}]
            return []

        def _x_limits(self, xs):
            if self.c['x_auto'].isChecked():
                xs = [x for x in xs if x is not None and len(x)]
                if not xs:
                    return None
                return float(min(np.nanmin(x) for x in xs)), float(max(np.nanmax(x) for x in xs))
            return self.c['xmin'].value(), self.c['xmax'].value()

        def _draw_shot(self, msg):
            images = self._images(msg)
            if images and len(images) != self.n_images:     # rare: a different number of panels
                self._shot_axes(len(images))
                self._first_draw()
            c = self.c
            title = f"Scan{msg['scan']:03d} shot {msg['shot']}"
            cmap = self.cmap.currentText()
            for i, (ax, im) in enumerate(zip(self.ax_img, self.ims)):
                if i >= len(images) or images[i].get('image') is None:
                    continue
                p = images[i]
                img = np.asarray(p['image'])
                im.set_data(img)
                ext = p.get('extent') or (-0.5, img.shape[1] - 0.5, -0.5, img.shape[0] - 0.5)
                im.set_extent(ext)
                if c['x_auto'].isChecked():
                    ax.set_xlim(ext[0], ext[1])
                else:                               # the x range applies to the images too
                    ax.set_xlim(c['xmin'].value(), c['xmax'].value())
                ax.set_ylim(ext[2], ext[3])
                if c['img_auto'].isChecked():
                    finite = img[np.isfinite(img)]
                    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
                    im.set_clim(min(lo, 0.0), hi if hi > lo else lo + 1.0)
                else:
                    im.set_clim(c['vmin'].value(), c['vmax'].value())
                name = p.get('name') or ''
                ax.set_title(f"{title}{': ' + name if name else ''}" if i == 0 else name, fontsize=10)
            if self.applied.get('cmap') != cmap:
                for im in self.ims + [self.im_wf]:
                    im.set_cmap(colormap(cmap))
                self.applied['cmap'] = cmap
            line = msg.get('line')
            stats = self.opt.get('stats')
            r = {(k[len(self.prefix) + 1:] if self.prefix and k.startswith(self.prefix + ' ') else k): v
                 for k, v in (msg.get('results') or {}).items()}
            if stats:
                try:
                    self.ax_line.set_title(stats.format_map(_Missing(r)), fontsize=9)
                except (ValueError, TypeError):
                    self.ax_line.set_title(' ', fontsize=9)
            elif not images:
                self.ax_line.set_title(title, fontsize=10)
            if line is not None and len(line.get('y', ())):
                x = np.asarray(line['x'], float)
                y = np.asarray(line['y'], float)
                if c['log'].isChecked():
                    y = np.where(y > 0, y, np.nan)
                self.line.set_data(x, y)
                scale = 'log' if c['log'].isChecked() else 'linear'
                if self.applied.get('scale') != scale:
                    self.ax_line.set_yscale(scale)
                    self.applied['scale'] = scale
                xl = self._x_limits([x])
                if xl:
                    self.ax_line.set_xlim(*xl)
                if c['y_auto'].isChecked():
                    sel = (x >= xl[0]) & (x <= xl[1]) if xl else np.ones(x.shape, bool)
                    vals = y[sel] if sel.any() else y
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        lo, hi = float(vals.min()), float(vals.max())
                        if c['log'].isChecked():
                            self.ax_line.set_ylim(max(lo, hi * 1e-5) / 1.5, hi * 1.5)
                        else:
                            pad = 0.05 * (hi - lo or abs(hi) or 1.0)
                            self.ax_line.set_ylim(min(lo, 0.0) if lo >= 0 else lo - pad, hi + pad)
                else:
                    self.ax_line.set_ylim(c['ymin'].value(), c['ymax'].value())
            self.can_shot.draw()
            if not self.shot_frozen:
                self.fig_shot.set_layout_engine('none')
                self.shot_frozen = True
            msg['t_drawn'] = time.time()
            if self.log_path:
                with open(self.log_path, 'a') as f:
                    f.write(f"{msg['scan']},{msg['shot']},{msg.get('t_sent', np.nan)},"
                            f"{msg['t_received']},{msg['t_drawn']}\n")

        def _first_draw(self):
            """Apply colormap / scale and lay the figures out once, then keep that
            layout (re-running constrained layout costs ~0.1 s per draw)."""
            cmap, scale = self.cmap.currentText(), 'log' if self.c['log'].isChecked() else 'linear'
            for im in self.ims + [self.im_wf]:
                im.set_cmap(colormap(cmap))
            self.ax_line.set_yscale(scale)
            self.applied.update(cmap=cmap, scale=scale)
            for fig, canvas in ((self.fig_shot, self.can_shot), (self.fig_wf, self.can_wf)):
                canvas.draw()
                fig.set_layout_engine('none')
            self.shot_frozen = True
            self._draw_aggregates()

        def _checked_trends(self):
            return [self.trends.item(i).text() for i in range(self.trends.count())
                    if self.trends.item(i).checkState() == QtCore.Qt.CheckState.Checked]

        def _draw_aggregates(self):
            self.dirty_agg = False
            self.last_agg = time.monotonic()
            shots = sorted(self.rows)
            cols = self._checked_trends()
            by_param = self.trend_x.currentText() == 'scan parameter' and bool(self.params)
            key = (tuple(cols), by_param)
            if key != self.trend_key:               # the selection changed: new axes
                self.trend_key = key
                self.fig_tr.clf()
                self.ax_tr = list(np.atleast_1d(self.fig_tr.subplots(max(len(cols), 1), 1, sharex=True)))
                style = 'o' if by_param else '.-'
                self.lines_tr = [a.plot([], [], style, ms=4, lw=0.6)[0] for a in self.ax_tr]
                for a, col in zip(self.ax_tr, cols):
                    a.text(0.01, 0.95, col, transform=a.transAxes, fontsize=8, va='top',
                           bbox={'facecolor': 'white', 'alpha': 0.7, 'lw': 0, 'pad': 1})
                    a.tick_params(labelsize=8)
                self.ax_tr[-1].set_xlabel(self.param_name if by_param else 'shot')
                self.fig_tr.set_layout_engine('constrained')
                self.tr_frozen = False
            xs = [self.params.get(s, np.nan) for s in shots] if by_param else shots
            for a, line, col in zip(self.ax_tr, self.lines_tr, cols):
                line.set_data(xs, [self.rows[s].get(col, np.nan) for s in shots])
                a.relim()
                a.autoscale_view()
            self.can_tr.draw()
            if not getattr(self, 'tr_frozen', True):
                self.fig_tr.set_layout_engine('none')
                self.tr_frozen = True
            wf = sorted(self.wf)
            n = self.last_n.value()
            if n:
                wf = wf[-n:]
            if wf:
                x0 = self.wf[wf[0]][0]
                stack = np.vstack([y if len(x) == len(x0) and np.array_equal(x, x0)
                                   else np.interp(x0, x, y, left=np.nan, right=np.nan)
                                   for x, y in (self.wf[s] for s in wf)])
                self.im_wf.set_data(stack)
                self.im_wf.set_extent((x0[0], x0[-1], wf[0] - 0.5, wf[-1] + 0.5))
                finite = stack[np.isfinite(stack)]
                vmax = (float(np.percentile(finite, 99.5)) if finite.size else 1.0) \
                    if self.c['wf_auto'].isChecked() else self.c['wf_vmax'].value()
                self.im_wf.set_clim(min(0.0, float(finite.min()) if finite.size else 0.0), vmax or 1.0)
                xl = self._x_limits([x0])
                if xl:
                    self.ax_wf.set_xlim(*xl)
                self.ax_wf.set_ylim(wf[0] - 0.5, wf[-1] + 0.5)
                self.ax_wf.set_title(f'Scan{self.scan:03d}: {len(wf)} shots', fontsize=9)
            self.can_wf.draw()

        def _controls_changed(self, *_):
            if self.latest is not None and not self.c['pause'].isChecked():
                self._draw_shot(dict(self.latest, t_received=time.time()))
            self.dirty_agg = True

        def _trends_changed(self, *_):
            self.trend_key = None                   # rebuild the trend axes
            self.dirty_agg = True

        def closeEvent(self, event):
            s = self.settings
            for k, w in self.c.items():
                if k != 'pause':
                    s.setValue(k, w.isChecked() if isinstance(w, QtWidgets.QCheckBox) else w.value())
            s.setValue('cmap', self.cmap.currentText())
            s.setValue('last_n', self.last_n.value())
            s.setValue('trend_x', self.trend_x.currentText())
            s.setValue('trends', self._checked_trends())
            s.setValue('geometry', self.saveGeometry())
            QtWidgets.QApplication.instance().quit()
            super().closeEvent(event)

    return LiveWindow
