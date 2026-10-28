# Live views for LiveShotWatcher: matplotlib windows showing the latest
# analysed shot (the analyzer's display_data) and scan displayers redrawn as
# the scan runs.
#
# Everything here runs on the watcher's controller thread, which must be the
# main thread for the windows to show (GUI toolkits require it). With a
# non-interactive backend (Agg) the figures are still drawn, just not shown.

import os
import threading
import traceback

import matplotlib
import matplotlib.pyplot as plt

from PyGEECSPlotter.scan_data_analyzers.scan_data_analysis import _aux_kwarg


def read_scan_info(top_dir, scan):
    """``{key: value}`` from ``scans/ScanNNN/ScanInfoScanNNN.ini`` (values
    unquoted; ``{}`` if the file isn't there). GEECS writes it when the scan
    starts, with e.g. ``Scan Parameter``, ``Start``, ``End``, ``Step size``
    and ``Shots per step``."""
    label = f'Scan{int(scan):03d}'
    path = os.path.join(os.fspath(top_dir), 'scans', label, f'ScanInfo{label}.ini')
    info = {}
    try:
        with open(path) as f:
            for line in f:
                key, sep, value = line.partition('=')
                if sep:
                    info[key.strip()] = value.strip().strip('"')
    except OSError:
        pass
    return info


def _interactive_backend():
    backend = matplotlib.get_backend().lower()
    try:
        from matplotlib.backends import backend_registry, BackendFilter
        non_interactive = backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE)
    except ImportError:                 # matplotlib < 3.9
        non_interactive = ['agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
    return backend not in non_interactive and 'inline' not in backend


class LiveView:
    """
    The live windows of one watcher: the latest shot (``show_shots``) and one
    window per scan displayer. Redrawn by ``update_shot`` / ``update_scan``;
    a window the user closes stays closed until ``new_scan``.
    """

    def __init__(self, show_shots=False, displayers=(), interval_s=1.0, log=print):
        self.show_shots = show_shots
        self.displayers = list(displayers)
        self.interval_s = interval_s
        self.log = log
        self.figs = {}              # key -> figure ('shot', or displayer index)
        self.closed = set()         # keys closed by the user this scan
        self.warned = set()
        self.label = ''
        self.last_draw = float('-inf')
        self.drawn = None           # what was drawn last (to skip unchanged redraws)
        self.shot_drawn = None      # shot number in the shot window
        self.scan_drawn_at = {}     # displayer index -> time it was last redrawn
        self.show = _interactive_backend() and threading.current_thread() is threading.main_thread()
        if self.active and not self.show:
            self.log("Live views: no interactive matplotlib backend, or not on the main "
                     "thread; figures are drawn but not shown.")

    @property
    def active(self):
        return bool(self.show_shots or self.displayers)

    def figures(self):
        return [f for f in self.figs.values() if plt.fignum_exists(f.number)]

    def new_scan(self, label):
        self.label = label
        self.closed.clear()
        self.drawn = None
        self.shot_drawn = None
        self.scan_drawn_at.clear()

    # ------------------------------------------------------------------
    def _figure(self, key, title, figsize=(6, 5), clear=True):
        """The live figure for ``key``, cleared unless ``clear`` is False (None if
        the user closed it)."""
        if key in self.closed:
            return None
        fig = self.figs.get(key)
        if fig is not None and not plt.fignum_exists(fig.number):
            self.closed.add(key)        # closed by the user
            return None
        if fig is None:
            fig = plt.figure(figsize=figsize, constrained_layout=True)
            self.figs[key] = fig
            if self.show:
                fig.show()
        if clear:
            fig.clf()
        self._title(fig, title)
        return fig

    def _adopt(self, key, fig, new_fig, title):
        """A displayer that drew its own figure: make that the live window."""
        if new_fig is None or new_fig is fig:
            return
        plt.close(fig)
        self.figs[key] = new_fig
        self._title(new_fig, title)
        if self.show:
            new_fig.show()

    @staticmethod
    def _title(fig, title):
        try:
            fig.canvas.manager.set_window_title(title)
        except AttributeError:
            pass

    def _failed(self, fig, key, what):
        message = traceback.format_exc().strip().splitlines()[-1]
        fig.clf()
        fig.text(0.5, 0.5, f'{what}:\n{message}', ha='center', va='center', wrap=True)
        if key not in self.warned:
            self.warned.add(key)
            self.log(f"Live view {what} failed:\n{traceback.format_exc()}")

    def _refresh(self, fig):
        if self.show:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    # ------------------------------------------------------------------
    def update_shot(self, analyzer, shot, data, results, aux):
        """The analyzer's ``display_data`` of one shot."""
        title = f'{self.label} shot {shot}'
        # an analyzer with ``live_update = True`` updates its own figure in
        # place (much faster than drawing it afresh for every shot)
        in_place = getattr(analyzer, 'live_update', False)
        fig = self._figure('shot', f'Live: {title}', clear=not in_place)
        if fig is None:
            return
        ax = None if in_place else fig.add_subplot()
        try:
            if data is None:
                raise ValueError('the analyzer returned no data to display')
            out = analyzer.display_data(data, return_dict=results, title=title, fig=fig, ax=ax,
                                        **_aux_kwarg(analyzer.display_data, aux))
            new_fig = out[0] if isinstance(out, tuple) else None
            self._adopt('shot', fig, new_fig, f'Live: {title}')
            fig = self.figs['shot']
        except Exception:
            self._failed(fig, 'shot', f'display_data of {type(analyzer).__name__}')
        self._refresh(fig)

    def _due(self, i, displayer, now, force):
        """A displayer with ``live_interval_s`` is redrawn at most that often."""
        return force or now - self.scan_drawn_at.get(i, float('-inf')) >= getattr(
            displayer, 'live_interval_s', 0.0)

    def any_due(self, now, force=False):
        return any(self._due(i, d, now, force) for i, d in enumerate(self.displayers))

    def update_scan(self, scan, n_shots, now=0.0, force=False):
        """Each displayer on ``scan`` (the running scan, as far as it has got)
        that is due (see ``live_interval_s``)."""
        for i, displayer in enumerate(self.displayers):
            if not self._due(i, displayer, now, force):
                continue
            self.scan_drawn_at[i] = now
            title = f'Live: {self.label} {type(displayer).__name__} ({n_shots} shots)'
            # a displayer with ``live_update = True`` updates its figure in place
            in_place = getattr(displayer, 'live_update', False)
            fig = self._figure(i, title, displayer.display_dict.get('figsize', (6, 5))
                               if hasattr(displayer, 'display_dict') else (6, 5), clear=not in_place)
            if fig is None:
                continue
            ax = None if in_place else fig.add_subplot()
            try:
                new_fig, _ = scan.display_scan(displayer, fig=fig, ax=ax)
                self._adopt(i, fig, new_fig, title)
                fig = self.figs[i]
            except KeyError as err:
                # a column not there yet (e.g. no shot analysed so far)
                fig.clf()
                fig.text(0.5, 0.5, f'waiting for data:\n{err}', ha='center', va='center', wrap=True)
            except Exception:
                self._failed(fig, i, type(displayer).__name__)
            self._refresh(fig)

    def wait(self, seconds, event=None):
        """Wait up to ``seconds`` (less if ``event`` is set) while keeping the
        windows responsive."""
        import time
        figs = self.figures() if self.show else []
        if not figs:
            if event is not None:
                event.wait(seconds)
            else:
                time.sleep(seconds)
            return
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end and not (event is not None and event.is_set()):
            figs[0].canvas.start_event_loop(min(0.02, max(0.0, t_end - time.monotonic())))

    def close_others(self):
        """Close every pyplot figure except the live windows."""
        keep = {f.number for f in self.figs.values()}
        for num in plt.get_fignums():
            if num not in keep:
                plt.close(num)
