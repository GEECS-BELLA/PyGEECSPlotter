# Live analysis: analyse each shot while the scan is still running.
#
# LiveShotWatcher waits for a scan to start, analyses each shot as soon as its
# diagnostic file(s) and its row in the scan's .tdms are complete, appends the
# results to analysis/ScanNNN/online/<name>_live.txt as it goes, and when the
# scan ends (its sfile appears) analyses any shots it couldn't, merges the
# results into the sfile and runs the displayers. Then it waits for the next
# scan. Optionally, live windows (live_view.py) show the latest shot and scan
# displayers redrawn as the scan runs, and the master control's multicast
# messages (live_trigger.py) say which shot to look for and when a scan ends.
#
# Threading: one controller loop (the thread that calls start()) owns all the
# bookkeeping, so there are no shared-state races; a thread pool only runs
# load + analyse (+ the shot's write_analyzed files). Results are collected in
# shot order. pyplot (displayers) stays on the controller thread.

import concurrent.futures
import threading
import os
import re
import time
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyGEECSPlotter.scan_data_analyzers.scan_data_analysis import ScanDataAnalyzer
from PyGEECSPlotter.scan_data_analyzers.live_analysis import LiveJob, online_output_dir
from PyGEECSPlotter.scan_data_analyzers.live_view import LiveView, read_scan_info
from PyGEECSPlotter.scan_data_analyzers.live_trigger import TriggerListener
from PyGEECSPlotter.scan_data_analyzers.sfile_frame import merge_results, rename_result_columns
from PyGEECSPlotter.utils.io_utils import file_lock
from PyGEECSPlotter.utils.tdms_utils import read_scan_tdms, scan_tdms_path, alias_map_for_scan
import PyGEECSPlotter.utils.navigation_utils as nav_utils

_SCAN_DIR_RE = re.compile(r'Scan(\d{3,})$')


def _inputs(analyzer):
    """``[(diagnostic, file_ext), ...]`` the analyzer reads per shot."""
    inputs = getattr(analyzer, 'inputs', None)
    if inputs:
        return list(inputs)
    if getattr(analyzer, 'diagnostic', None) and getattr(analyzer, 'file_ext', None):
        return [(analyzer.diagnostic, analyzer.file_ext)]
    return []


_SMB_KEYS = ('DirectoryCacheLifetime', 'FileNotFoundCacheLifetime', 'FileInfoCacheLifetime')


def smb_cache_warning(path):
    """A warning (str) if ``path`` is on a network drive and Windows' SMB client
    caches listings / file info there (new files then show up only every ~10 s,
    and a file being written can look complete), else None."""
    if os.name != 'nt' or not path:
        return None
    path = os.path.abspath(os.fspath(path))
    remote = path.startswith('\\\\')
    if not remote:
        import ctypes
        try:
            remote = ctypes.windll.kernel32.GetDriveTypeW(os.path.splitdrive(path)[0] + '\\') == 4
        except (OSError, AttributeError):
            remote = False
    if not remote:
        return None
    import winreg
    bad = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r'SYSTEM\CurrentControlSet\Services\LanmanWorkstation\Parameters') as key:
            for name in _SMB_KEYS:
                try:
                    value = winreg.QueryValueEx(key, name)[0]
                except OSError:
                    value = None            # not set: Windows' default (10 s, or 5 s for not-found)
                if value != 0:
                    bad.append(f'{name}={value if value is not None else "default"}')
    except OSError:
        return None
    if not bad:
        return None
    return (f"{path} is on a network drive and this PC caches its listings ({', '.join(bad)}): new "
            f"shots will show up in bursts every ~10 s. Set {', '.join(_SMB_KEYS)} to 0 (DWORD) under "
            r"HKLM\SYSTEM\CurrentControlSet\Services\LanmanWorkstation\Parameters (admin) on the PC "
            f"that runs the watcher.")


class _ScanState:
    """Bookkeeping for the scan being watched (owned by the controller loop)."""

    def __init__(self, top_dir, scan):
        self.top_dir = top_dir
        self.scan = scan
        self.label = f'Scan{scan:03d}'
        self.scan_dir = os.path.join(top_dir, 'scans', self.label)
        self.tdms = scan_tdms_path(top_dir, scan)
        self.sfile = nav_utils.get_sfilename_from_top_dir(top_dir, scan)
        self.info = os.path.join(top_dir, 'analysis', f's{scan}info.txt')
        self.files = {}          # shot -> {diagnostic: path}
        self.sizes = {}          # path -> (size, time first seen at that size)
        self.complete = {}       # shot -> {diagnostic: path}, files done writing
        self.first_ready = {}    # shot -> time its files were complete
        self.partial_since = {}  # shot -> time its present files settled, inputs missing
        self.partial = set()     # shots analysed with an input missing
        self.submitted = {}      # shot -> future of its analysis
        self.writing = {}        # shot -> (future of its file writes, result row)
        self.done = set()        # shots with results (this run or a previous one)
        self.deferred = set()    # waited too long for their .tdms context
        self.failed = set()
        self.pending = []        # result rows not yet written to the live file
        self.results = []        # every result row of the scan, shown or done (for the live views)
        self.latest = None       # (shot, data, results, aux) of the newest shot collected
        self.scan_info = read_scan_info(top_dir, scan)
        self.live_scan = None    # ScanDataAnalyzer of the running scan, for the live views
        self.triggered = {}      # shot -> time its Path>> message arrived
        self.expected = {}       # triggered shot -> {diagnostic: path} not seen yet
        self.ended_at = None     # time ScanEnd arrived
        self.warned_silent = False
        self.gui_sent = set()    # shots sent to the live GUI from the .tdms alone (no job)
        self.context = None      # last good .tdms frame
        self.context_size = None # .tdms size when it was read
        self.context_rows = {}   # shot -> row position in context (its last row)
        self.next_read = 0.0     # earliest time to re-read the .tdms
        self.warned_no_shots = False
        self.required = []       # required_columns as named in the .tdms frame
        self.sfile_size = None   # (size, time first seen at that size)
        self.newer_since = None  # time a newer scan was first seen started


class LiveShotWatcher:
    r"""
    Analyse each shot while the scan runs; finish up when the scan ends.

    Parameters
    ----------
    experiment_dir : str or Path, optional
        Folder holding the year folders; the day is today's (re-checked).
    job : LiveJob or analyzer, optional
        The one analyzer (single- or multi-diagnostic) to run, with its
        displayers and write options (see ``LiveJob``). None: no analysis,
        only the live views (e.g. ``live_displayers`` of ``.tdms`` scalars).
    top_dir : str or Path, optional
        Watch this day folder instead (fixed).
    poll_s : float
        Seconds between polls.
    max_workers : int, optional
        Analysis threads (1 if the analyzer is not thread-safe).
    context_timeout_s : float
        How long a shot whose files are complete may wait for its row (and
        ``required_columns``) in the ``.tdms``. After that it is analysed at
        the end of the scan, with the sfile as context.
    required_columns : list of str, optional
        Columns the analyzer needs from the shot's context, e.g. a
        filter-wheel setting. Default: the analyzer's ``required_columns``
        attribute, if any. A shot waits until they are filled in.
    start_scan : int, optional
        First scan to analyse (on the first day watched). Default: the scan
        running now, else the next to start.
    write : bool
        False: never write the sfile / ScanData files (dry run).
    file_settle_s : float
        A shot's file counts as written once its size has not changed for
        this long.
    settle_s : float
        The scan has ended when its ``sNinfo.txt`` exists (GEECS writes it
        after the sfile) and the sfile has stopped changing; if there is no
        info file, when the sfile has not changed for ``settle_s`` seconds.
    abandon_grace_s : float
        A scan with no sfile is given up (aborted) once a newer scan has been
        running this long: its live results are kept, nothing is merged.
    analyze_partial : bool
        Multi-diagnostic jobs: analyse a shot that is missing one of the
        inputs (its data is None; the analyzer must handle that) once a
        later shot has all of them, or after ``missing_input_timeout_s``.
        False (default): such shots are skipped, as by ``analyze_scan``.
    missing_input_timeout_s : float
        See ``analyze_partial``.
    show_shots : bool
        Show each shot live, as soon as it is analysed (before its files are
        written). By default in the live GUI, a Qt window in its own process
        (``PyGEECSPlotter.displayers.live_gui``) fed by the analyzer's
        ``live_payload``: the latest shot's images and 1-D view, trends of
        its results and ``.tdms`` scalars (against shot or scan parameter)
        and a waterfall. Without a job, the ``.tdms`` scalars alone (trends).
    live_view : str
        ``'qt'`` (default): ``show_shots`` uses the live GUI. ``'pyplot'``:
        the analyzer's ``display_data`` in a pyplot window redrawn by the
        watcher's own loop (slower; it shares the loop with the analysis).
    gui_options : dict, optional
        Options of the live GUI window (see ``live_gui``), e.g. ``labels``,
        ``trends``, ``x_range``, ``log_line``.
    end_process : bool
        Run the job's displayers (and its ``write_displayed`` per-shot
        figures) after each scan in a separate process (``live_end.py``,
        output in ``analysis/ScanNNN/online/end_of_scan.log``), so the
        watcher is ready for the next scan within about a second. Default:
        here (a job that doesn't pickle is also run here).
    live_displayers : list of ScanDisplayer
        Redrawn in their own windows as the scan runs, on every ``.tdms`` row
        so far (device scalars) plus the live results (named as they will be
        in the sfile). Mid-scan the bins come from the ScanInfo file's shots
        per step. The ``.tdms`` columns carry their aliases if an earlier
        sfile gave them, else device names (logged at scan start).
    gui_interval_s : float
        Redraw the live windows at most this often.
    on_shot : callable, optional
        ``on_shot(scan, shot, data, results, aux)``, called on the watcher's
        loop as soon as a shot's analysis is done, before its files are
        written (e.g. to send it to a display in another process). Errors
        are logged, not raised.
    trigger : str or (str, int), optional
        The experiment's master-control multicast group, e.g.
        ``'234.5.6.17:58456'`` (each experiment has its own). Its ``Path>>``
        message names each shot ~0.3-2 s before the shot's files exist: the
        watcher then checks exactly those files and analyses the shot once
        its ``.tdms`` row is in and its files have kept their size for
        ``trigger_settle_s``. ``ScanEnd`` ends the scan (finished when the
        sfile is written; aborted if none comes within ``settle_s``).
        Shots whose message is lost are still found by the folder polling.
    trigger_root : str, optional
        The experiment's root path as master control sends it, when
        ``experiment_dir`` is its local mapping. Both are in the
        experiment's GEECS database, e.g.
        ``trigger_root=r'\\192.168.13.6\pdna2\data\ControlRoom'`` for
        ``experiment_dir=r'N:\data\ControlRoom'``. Not needed when
        ``experiment_dir`` is given as that path itself.
    trigger_settle_s : float
        File-settle time of a triggered shot (``file_settle_s`` otherwise).

    The live windows need an interactive matplotlib backend and ``start()``
    called from the main thread (a script, not a notebook cell).

    Per-shot results go to ``analysis/ScanNNN/online/<name>_live.txt`` as
    they come in (a restarted watcher skips the shots already there);
    displayer figures to ``analysis/ScanNNN/online/<name>/``.
    """

    def __init__(self, experiment_dir=None, job=None, top_dir=None, poll_s=0.1, max_workers=None,
                 context_timeout_s=30.0, required_columns=None, start_scan=None, write=True,
                 file_settle_s=0.3, settle_s=30.0, abandon_grace_s=60.0, analyze_partial=False,
                 missing_input_timeout_s=10.0, show_shots=False, live_displayers=(),
                 gui_interval_s=1.0, on_shot=None, trigger=None, trigger_root=None,
                 trigger_settle_s=0.1, live_view='qt', gui_options=None, end_process=False):
        if experiment_dir is None and top_dir is None:
            raise ValueError("Give experiment_dir (watch today's folder) or top_dir.")
        if job is None and not live_displayers and not (show_shots and live_view == 'qt'):
            raise ValueError("Give the job (a LiveJob or an analyzer) to run, or live_displayers.")
        if live_view not in ('qt', 'pyplot'):
            raise ValueError("live_view is 'qt' or 'pyplot'.")
        self.experiment_dir = os.fspath(experiment_dir) if experiment_dir is not None else None
        self.fixed_top_dir = os.fspath(top_dir) if top_dir is not None else None
        self.job = job if job is None or isinstance(job, LiveJob) else LiveJob(job)
        self.analyzer = self.job.analyzer if self.job else None
        self.inputs = _inputs(self.analyzer) if self.job else []
        self.name = self.job.name if self.job else 'live plots'
        self.poll_s = poll_s
        self.max_workers = max_workers if self.job and self.job.parallel else 1
        self.context_timeout_s = context_timeout_s
        self.required_columns = list(required_columns if required_columns is not None
                                     else getattr(self.analyzer, 'required_columns', None) or [])
        self.start_scan = start_scan
        self.write = write
        self.file_settle_s = file_settle_s
        self.settle_s = settle_s
        self.abandon_grace_s = abandon_grace_s
        self.analyze_partial = analyze_partial
        self.missing_input_timeout_s = missing_input_timeout_s
        self.view = LiveView(show_shots and live_view == 'pyplot' and self.job is not None, live_displayers,
                             gui_interval_s, log=self._log)
        self.use_gui = show_shots and live_view == 'qt'
        self.gui_options = dict(gui_options or {})
        self.gui_process = None      # LiveGui (the Qt process), once started
        self.end_process = end_process
        self.end_procs = []          # end-of-scan processes (live_end.py) still running
        self.on_shot = on_shot
        self._wake = threading.Event()  # set when an analysis or write finishes, or a trigger arrives
        self.trigger = trigger
        self.trigger_root = trigger_root
        self.trigger_settle_s = trigger_settle_s
        self.listener = None         # TriggerListener, once started
        self._held = []              # trigger events of a scan after the one being watched
        self.state = None            # _ScanState of the scan being watched
        self.aliases = {}            # .tdms raw name -> sfile alias (from an earlier sfile)
        self.last_scan = {}          # top_dir -> last scan handled
        self.status = ''
        self._running = False
        self._pool = None

    # ------------------------------------------------------------------
    @property
    def top_dir(self):
        return self.fixed_top_dir or nav_utils.get_todays_top_dir(self.experiment_dir)

    def _log(self, message):
        self.status = message
        print(time.strftime('[%H:%M:%S] ') + message)

    def live_results_path(self, top_dir, scan):
        return os.path.join(online_output_dir(top_dir, scan), f'{self.job.name}_live.txt')

    def progress(self):
        """Counts for the scan being watched (or None)."""
        st = self.state
        if st is None:
            return None
        running = sum(1 for f in st.submitted.values() if not f.done())
        extra = {}
        if self.listener is not None:
            age = None if self.listener.last_packet is None else time.monotonic() - self.listener.last_packet
            extra = {'triggered': len(st.triggered), 'packets': self.listener.packets,
                     'last packet s': None if age is None else round(age, 1)}
        return {'scan': st.scan, **extra, 'files': len(st.files), 'tdms shots': len(st.context_rows),
                'done': len(st.done),
                'arriving': len(set(st.files) - set(st.complete) - st.done),
                'in progress': running, 'writing': len(st.writing),
                'deferred': len(st.deferred), 'failed': len(st.failed),
                'waiting': len(set(st.complete) - set(st.submitted) - set(st.writing) - st.done
                               - st.deferred - st.failed)}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def _scan_numbers(self, top_dir):
        try:
            with os.scandir(os.path.join(top_dir, 'scans')) as it:
                return sorted(int(m.group(1)) for e in it
                              if e.is_dir() and (m := _SCAN_DIR_RE.fullmatch(e.name)))
        except FileNotFoundError:
            return []

    def _next_scan(self, top_dir):
        """The next scan to watch: started (its .tdms exists), not handled yet."""
        scans = self._scan_numbers(top_dir)
        if top_dir not in self.last_scan:
            # first look at this day: start_scan (first day only), else the
            # newest started scan if it is still running, else the next one
            if self.start_scan is not None and not self.last_scan:
                self.last_scan[top_dir] = self.start_scan - 1
            else:
                started = [n for n in scans if os.path.exists(scan_tdms_path(top_dir, n))]
                newest = max(started, default=0)
                running = newest and not os.path.exists(
                    nav_utils.get_sfilename_from_top_dir(top_dir, newest))
                self.last_scan[top_dir] = newest - 1 if running else newest
        for n in scans:
            if n > self.last_scan[top_dir] and os.path.exists(scan_tdms_path(top_dir, n)):
                return n
        return None

    def _begin(self, top_dir, scan):
        st = _ScanState(top_dir, scan)
        path = self.live_results_path(top_dir, scan) if self.job else None
        if path and os.path.exists(path):
            try:
                earlier = pd.read_csv(path, sep='\t', float_precision='round_trip')
                st.done = set(earlier['Shotnumber'].astype(int))
                st.results = earlier.to_dict('records')
            except Exception:
                pass
        self.view.new_scan(st.label)
        self.aliases = alias_map_for_scan(top_dir, scan)
        if not self.aliases:
            self._log(f"{st.label}: no earlier sfile found for the column aliases; "
                      f".tdms columns keep their device names")
        st.required = [self.aliases.get(c, c) for c in self.required_columns]
        self.state = st
        self._log(f"{st.label}: started; "
                  + ("analysing shots as they arrive" if self.job else "live plots only")
                  + (f" ({len(st.done)} already done)" if st.done else ''))
        if self.job is not None and self.job.use_bg:
            self._helper_scan(st)              # resolves st.bg now, not on the first shot
            missing = [d for d, _ in self.inputs
                       if not (st.bg.get(d) if isinstance(st.bg, dict) else st.bg)]
            if missing:
                self._log(f"{st.label}: use_bg, but no background file in "
                          f"{os.path.join(top_dir, 'analysis')} for {', '.join(missing)}")

    def _find_files(self, st):
        """New shot files of each input, ``ScanNNN_<diag>_<shot><ext>``."""
        for diag, ext in self.inputs:
            folder = os.path.join(st.scan_dir, diag)
            pattern = re.compile(rf'{re.escape(st.label)}_{re.escape(diag)}_(\d+){re.escape(ext)}$')
            try:
                with os.scandir(folder) as it:
                    for e in it:
                        m = pattern.match(e.name)
                        if m:
                            st.files.setdefault(int(m.group(1)), {})[diag] = e.path
            except FileNotFoundError:
                continue

    def _settled(self, st, path, now, settle_s):
        """The file exists and has kept its (non-zero) size for ``settle_s``."""
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        if st.sizes.get(path, (None,))[0] != size:
            st.sizes[path] = (size, now)
        return size > 0 and now - st.sizes[path][1] >= settle_s

    def _update_complete(self, st, now):
        """Shots whose every input file exists and kept its size for ``file_settle_s``
        (and, with ``analyze_partial``, shots given up waiting for a missing input)."""
        n_inputs = len(self.inputs)
        newest_full = max((s for s, p in st.files.items() if len(p) == n_inputs), default=0)
        for shot, paths in st.files.items():
            if shot in st.complete or shot in st.done:
                continue
            settle_s = self.trigger_settle_s if shot in st.triggered else self.file_settle_s
            settled = all([self._settled(st, path, now, settle_s) for path in paths.values()])
            if not settled:
                continue
            if len(paths) < n_inputs:
                if not self.analyze_partial:
                    continue
                since = st.partial_since.setdefault(shot, now)
                if shot >= newest_full and now - since < self.missing_input_timeout_s:
                    continue            # the missing file may still come
                st.partial.add(shot)
                self._log(f"{st.label}: shot {shot} has no "
                          f"{', '.join(d for d, _ in self.inputs if d not in paths)}; "
                          f"analysing it without")
            st.complete[shot] = dict(paths)
            st.first_ready[shot] = now

    def _read_context(self, st, now):
        """Re-read the .tdms if it has grown since the last good read. Reading
        it costs a few ms per shot, so the next read waits at least twice as
        long as the last one took (long scans: the loop isn't kept busy
        reading). A failed read (locked, mid-segment) keeps the last good
        frame; the next poll tries again."""
        if now < st.next_read:
            return
        try:
            size = os.path.getsize(st.tdms)
        except OSError:
            return
        if st.context is not None and size == st.context_size:
            return
        t0 = time.monotonic()
        df = read_scan_tdms(st.tdms, aliases=self.aliases)
        t1 = time.monotonic()
        st.next_read = t1 + 2 * (t1 - t0)
        if df is None:
            return
        st.context, st.context_size = df, size
        shots = df['Shotnumber'] if 'Shotnumber' in df.columns else []
        st.context_rows = {int(n): i for i, n in enumerate(shots) if n == n}
        if len(df) and not st.context_rows and not st.warned_no_shots:
            st.warned_no_shots = True
            self._log(f"{st.label}: the .tdms has {len(df)} rows but no shot numbers "
                      f"(no 'Shotnumber' or '<device> shot #' channel); shots will wait "
                      f"for the sfile")

    def _context_row(self, st, shot):
        """The shot's .tdms row as a dict, or None if it isn't there (complete) yet."""
        i = st.context_rows.get(shot)
        if i is None:
            return None
        row = st.context.iloc[i].to_dict()
        for col in st.required:
            value = row.get(col)
            if value is None or (isinstance(value, float) and value != value):
                return None
        return row

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def _helper_scan(self, st):
        """A ScanDataAnalyzer for this scan with no data loaded, used to run
        ``_process_row_and_write`` (file loading, bg, per-shot writes)."""
        if getattr(st, 'helper', None) is None:
            st.helper = ScanDataAnalyzer(top_dir=st.top_dir, scan=st.scan)
            st.bg = self.job.bg(st.helper)
        return st.helper

    def _submit_ready(self, st, now):
        """Submit shots whose files are complete and whose context is in the .tdms."""
        waiting = [s for s in sorted(st.complete) if s not in st.submitted and s not in st.writing
                   and s not in st.done and s not in st.deferred and s not in st.failed]
        if not waiting:
            return
        self._read_context(st, now)
        helper = self._helper_scan(st)
        for shot in waiting:
            row = self._context_row(st, shot)
            if row is None:
                if now - st.first_ready[shot] > self.context_timeout_s:
                    st.deferred.add(shot)
                    self._log(f"{st.label}: shot {shot} has no complete .tdms row after "
                              f"{self.context_timeout_s:g} s; it will be analysed at the end")
                continue
            row.update({'scan': st.scan, 'Shotnumber': shot})
            row.setdefault('temp Bin number', row.get('Bin #'))
            for diag, ext in self.inputs:
                path = st.complete[shot].get(diag)
                # a missing input gets its expected path, as analyze_scan does
                row[f'{diag} file_list'] = path or os.path.join(
                    st.scan_dir, diag, f'{st.label}_{diag}_{shot:03d}{ext}')
                row[f'{diag} file_exists'] = int(path is not None)
            st.submitted[shot] = self._submit(helper._process_row, self.analyzer, st.bg, pd.Series(row))

    def _submit(self, fn, *args):
        """Run ``fn(*args)`` in the pool; the loop is woken when it finishes."""
        future = self._pool.submit(fn, *args)
        future.add_done_callback(lambda f: self._wake.set())
        return future

    def _collect(self, st, wait=False):
        """Finished shots, in two stages.

        1. Analysis done (any order): the shot is shown at once (``on_shot``,
           the live windows) and its files are written in the pool.
        2. Files written (in shot order): its row is appended to the live
           results file and the shot is ``done``. A row stays in
           ``st.pending`` until that file write succeeds (retried next poll).

        So a watcher stopped between the two redoes the shot on restart.
        ``wait``: wait for every analysis and write in flight."""
        helper = self._helper_scan(st) if st.submitted else None
        for shot in sorted(st.submitted):
            future = st.submitted[shot]
            if not wait and not future.done():
                continue
            del st.submitted[shot]
            try:
                context, data, results, aux = future.result()
            except Exception:
                st.failed.add(shot)
                self._log(f"{st.label}: shot {shot} failed (retried at the end of the scan):\n"
                          f"{traceback.format_exc()}")
                continue
            row = {'scan': st.scan, 'Shotnumber': shot, **results}
            st.results.append(row)
            if st.latest is None or shot >= st.latest[0]:
                st.latest = (shot, data, results, aux)
            if self.on_shot is not None:
                try:
                    self.on_shot(st.scan, shot, data, results, aux)
                except Exception:
                    self._log(f"{st.label}: on_shot failed for shot {shot}:\n{traceback.format_exc()}")
            if self.gui_process is not None:
                self._send_shot(st, shot, context, data, results, aux)
                self._send_status(st)
            args = (helper._write_row, self.analyzer, context, data, aux,
                    self.job.write_analyzed, self.job.write_lineouts)
            if self._pool is not None:
                st.writing[shot] = (self._submit(*args), row)
            else:                       # closing: write here
                done = concurrent.futures.Future()
                try:
                    args[0](*args[1:])
                    done.set_result(None)
                except Exception as err:
                    done.set_exception(err)
                st.writing[shot] = (done, row)
        for shot in sorted(st.writing):
            future, row = st.writing[shot]
            if not wait and not future.done():
                break                   # keep shot order in the live results file
            del st.writing[shot]
            try:
                future.result()
                st.pending.append(row)
                st.done.add(shot)
            except Exception:
                st.failed.add(shot)
                self._log(f"{st.label}: writing shot {shot}'s files failed (retried at the end of "
                          f"the scan):\n{traceback.format_exc()}")
        if st.pending:
            try:
                self._append_live(st, pd.DataFrame(st.pending))
                st.pending = []
            except Exception as err:
                self._log(f"{st.label}: could not write the live results "
                          f"({len(st.pending)} shots kept, retrying): {err!r}")

    # ------------------------------------------------------------------
    # Master-control triggers
    # ------------------------------------------------------------------
    def _trigger_top(self, top):
        """The watcher's own name for a message's day folder, or None if it
        isn't a day of the watched experiment."""
        norm = lambda p: os.path.normcase(os.path.normpath(os.path.abspath(p)))
        if self.fixed_top_dir:
            return self.fixed_top_dir if norm(top) == norm(self.fixed_top_dir) else None
        parts = os.path.normpath(top).split(os.sep)
        if len(parts) < 4 or norm(os.sep.join(parts[:-3])) != norm(self.experiment_dir):
            return None
        return os.path.join(self.experiment_dir, *parts[-3:])

    def _start_trigger(self):
        """Start listening (once). A listener that can't start is logged and
        the watcher carries on with folder polling."""
        if self.trigger is None or self.listener is not None:
            return
        root_map = None
        if self.trigger_root:
            local = self.experiment_dir or os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.normpath(self.fixed_top_dir))))
            root_map = (self.trigger_root, local)
        try:
            self.listener = TriggerListener(self.trigger, lambda top: self._trigger_top(top) is not None,
                                            root_map, self._wake).start()
            self._log(f"Listening for master-control messages on {self.listener.group}:"
                      f"{self.listener.port}" + (f" ({root_map[0]} -> {root_map[1]})" if root_map else ''))
        except OSError as err:
            self._log(f"Can't listen on {self.trigger} ({err!r}); finding shots by folder polling only")
            self.trigger = None

    def _handle_triggers(self):
        """Apply the master-control events: a Path>> message begins its scan if
        need be and marks its shot triggered (its files are then checked by
        name); ScanEnd marks the scan ended. Messages for a later scan wait
        until the current one is finished."""
        events, self._held = self._held + self.listener.drain(), []
        for ev in events:
            st = self.state
            if ev.kind == 'end':
                if st is not None and st.ended_at is None:
                    st.ended_at = ev.t
                    self._log(f"{st.label}: ScanEnd received; finishing when the sfile is written")
                continue
            if ev.kind != 'shot':
                continue
            top = self._trigger_top(ev.top)
            if top is None:
                continue
            if st is None:
                if top in self.last_scan and ev.scan <= self.last_scan[top]:
                    continue                    # a scan already handled
                if self.start_scan is not None and not self.last_scan and ev.scan < self.start_scan:
                    continue
                self.last_scan.setdefault(top, ev.scan - 1)
                self._begin(top, ev.scan)
                st = self.state
            elif os.path.normcase(top) != os.path.normcase(st.top_dir) or ev.scan != st.scan:
                if (os.path.normcase(top), ev.scan) > (os.path.normcase(st.top_dir), st.scan):
                    self._held.append(ev)       # the next scan: after this one is finished
                continue
            if ev.shot in st.done or ev.shot in st.triggered:
                continue
            st.triggered[ev.shot] = ev.t
            st.expected[ev.shot] = {d: os.path.join(st.scan_dir, d, f'{st.label}_{d}_{ev.shot:03d}{e}')
                                    for d, e in self.inputs}

    def _check_triggered(self, st):
        """The expected files of triggered shots, checked by name (no listing)."""
        for shot in list(st.expected):
            found = st.files.setdefault(shot, {})
            for diag, path in list(st.expected[shot].items()):
                if os.path.exists(path):
                    found[diag] = path
                    del st.expected[shot][diag]
            if not st.expected[shot]:
                del st.expected[shot]
            if not found:
                del st.files[shot]

    def _trigger_pending(self):
        """A triggered shot is waiting for its files or row (poll fast)."""
        st = self.state
        if st is None or not st.triggered:
            return False
        busy = st.done | st.failed | st.deferred
        return any(s not in busy and s not in st.submitted and s not in st.writing for s in st.triggered)

    # ------------------------------------------------------------------
    # Live GUI (Qt, in its own process)
    # ------------------------------------------------------------------
    def _start_gui(self):
        """Start the live GUI process (once). If it can't start, that is logged
        and the watcher carries on without it."""
        if not self.use_gui or self.gui_process is not None:
            return
        from PyGEECSPlotter.displayers.live_gui import start_gui
        options = {'n_images': len(self.inputs) if len(self.inputs) > 1 else 1}
        options.update(self.gui_options)
        log_path = options.pop('log_path', None)     # per drawn shot: scan, shot, t_sent, t_received, t_drawn
        try:
            self.gui_process = start_gui(self.name, title=f'Live {self.name}', options=options, log_path=log_path)
        except Exception as err:
            self._log(f"The live GUI didn't start ({err!r}); carrying on without it")
            self.use_gui = False

    @staticmethod
    def _numeric_context(context):
        """The shot's .tdms scalars (numbers only; not shot counters or file columns)."""
        out = {}
        for k, v in context.items():
            if (isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                    and k not in ('scan', 'Shotnumber', 'Bin #', 'temp Bin number')
                    and not k.endswith((' shot #', ' file_exists'))):
                out[k] = float(v)
        return out

    def _param(self, st, context):
        raw = st.scan_info.get('Scan Parameter', '')
        name = self.aliases.get(raw, raw)
        if name in context:
            try:
                return {'name': name, 'value': float(context[name])}
            except (TypeError, ValueError):
                pass
        return None

    def _send_shot(self, st, shot, context, data, results, aux):
        """One shot to the live GUI: the analyzer's live_payload, plus its .tdms
        scalars and scan parameter value."""
        try:
            msg = (self.analyzer.live_payload(st.scan, shot, data, results, aux) if self.job
                   else {'kind': 'shot', 'scan': st.scan, 'shot': shot})
            msg['context'] = self._numeric_context(context)
            msg['param'] = self._param(st, context)
            msg['t_sent'] = time.time()
            self.gui_process.send(msg)
        except Exception:
            if not getattr(self, '_warned_payload', False):
                self._warned_payload = True
                self._log(f"{st.label}: live payload of shot {shot} failed (logged once):\n"
                          f"{traceback.format_exc()}")

    def _send_status(self, st, text=None):
        if self.gui_process is None:
            return
        if text is None:
            p = self.progress() or {}
            text = (f"{st.label}: {p.get('done', 0)} done, {p.get('in progress', 0)} analysing, "
                    f"{p.get('writing', 0)} writing, {p.get('waiting', 0) + p.get('arriving', 0)} waiting, "
                    f"{p.get('failed', 0)} failed"
                    + (f", {p['triggered']} triggered" if 'triggered' in p else '')
                    + (f"; GUI dropped {self.gui_process.dropped}" if self.gui_process.dropped else ''))
        self.gui_process.send({'kind': 'status', 'text': text})

    def _send_tdms_rows(self, st):
        """No job: each new .tdms row to the live GUI (trends of the device scalars)."""
        if st.context is None:
            return
        for shot, i in sorted(st.context_rows.items()):
            if shot not in st.gui_sent:
                st.gui_sent.add(shot)
                self._send_shot(st, shot, st.context.iloc[i].to_dict(), None, None, None)

    # ------------------------------------------------------------------
    # Live views
    # ------------------------------------------------------------------
    def _live_scan(self, st):
        """The running scan as a ScanDataAnalyzer for the live displayers: every
        .tdms row so far, with the live results merged in (named as they will be
        in the sfile), bins from the ScanInfo file. None before the first row."""
        df = st.context
        if df is None or 'Shotnumber' not in df.columns:
            return None
        data = df[df['Shotnumber'].notna()].copy()
        if data.empty:
            return None
        data['Shotnumber'] = data['Shotnumber'].astype(int)
        data['scan'] = st.scan
        data = data.drop_duplicates('Shotnumber', keep='last').reset_index(drop=True)
        if st.results:
            label = self.job.analysis_label or ''
            results = pd.DataFrame(st.results).drop_duplicates(['scan', 'Shotnumber'], keep='last')
            data = merge_results(data, rename_result_columns(results, self.job.column_prefix, label))
        if 'Bin #' in data.columns and data['Bin #'].notna().all():
            data['temp Bin number'] = data['Bin #']
        else:
            try:
                per_step = int(float(st.scan_info.get('Shots per step', 0)))
            except ValueError:
                per_step = 0
            data['temp Bin number'] = (data['Shotnumber'] - 1) // per_step + 1 if per_step > 0 else 1
        if st.live_scan is None:
            st.live_scan = ScanDataAnalyzer(top_dir=st.top_dir, scan=st.scan)
            st.live_scan.sfilename = None     # not written yet: titles show the scan label
        scan = st.live_scan
        scan.data = data
        raw = st.scan_info.get('Scan Parameter', '')
        param = self.aliases.get(raw, raw)
        scan.scan_parameter = param if param in data.columns else 'Shotnumber'
        if self.inputs:
            self.analyzer.register_with_scan(scan, remove_missing_files=False)
        # a displayer with its own analyzer (e.g. a reader of the per-shot files
        # the job writes) finds its files the same way; shots without them yet
        # are left out
        for displayer in self.view.displayers:
            other = getattr(displayer, 'analyzer', None)
            if other is not None and other is not self.analyzer and hasattr(other, 'register_with_scan'):
                other.register_with_scan(scan)
        return scan

    def _update_views(self, st, now, force=False):
        """Redraw the live windows if something new came in (at most every
        ``gui_interval_s``, unless ``force``)."""
        view = self.view
        if not view.active or (not force and now - view.last_draw < view.interval_s):
            return
        latest = st.latest[0] if st.latest else None
        key = (latest, len(st.context_rows), len(st.done))
        if key == view.drawn:
            return
        view.drawn, view.last_draw = key, now
        if view.show_shots and st.latest and (force or latest != view.shot_drawn):
            view.shot_drawn = latest
            view.update_shot(self.analyzer, *st.latest)
        if view.displayers and view.any_due(now, force):
            scan = self._live_scan(st)
            if scan is not None:
                view.update_scan(scan, len(scan.data), now, force)

    def _append_live(self, st, df):
        path = self.live_results_path(st.top_dir, st.scan)
        with file_lock(path):
            if os.path.exists(path):
                df = pd.concat([pd.read_csv(path, sep='\t', float_precision='round_trip'), df], ignore_index=True)
            df.to_csv(path, sep='\t', index=False)

    # ------------------------------------------------------------------
    # End of scan
    # ------------------------------------------------------------------
    def _scan_ended(self, st, now):
        """The sfile has stopped changing and ``sNinfo.txt`` exists (GEECS writes it
        after the sfile), or, with no info file, the sfile is unchanged for ``settle_s``."""
        try:
            size = os.path.getsize(st.sfile)
        except OSError:
            return False
        if st.sfile_size is None or st.sfile_size[0] != size:
            st.sfile_size = (size, now)
            return False
        return size > 0 and (os.path.exists(st.info) or now - st.sfile_size[1] >= self.settle_s)

    def _launch_end(self, st, results, out):
        """The job's displayers (and write_displayed) in a separate process
        (live_end.py). False if the job can't be handed over (e.g. it doesn't
        pickle): the caller then runs them here."""
        import pickle
        import subprocess
        import sys
        from PyGEECSPlotter.displayers.live_gui import package_env
        spec = {'job': self.job, 'sfile': st.sfile, 'results': results,
                'merged': self.write and self.job.write_sfile, 'output_dir': out}
        folder = online_output_dir(st.top_dir, st.scan)
        path = os.path.join(folder, 'end_of_scan.pkl')
        try:
            with open(path, 'wb') as f:
                pickle.dump(spec, f)
        except Exception as err:
            try:
                os.remove(path)                 # the partly written spec
            except OSError:
                pass
            if not getattr(self, '_warned_pickle', False):
                self._warned_pickle = True
                self._log(f"The end-of-scan work can't go to a separate process ({err!r}); "
                          f"running it here (logged once)")
            return False
        self.end_procs = [p for p in self.end_procs if p.poll() is None]
        log = open(os.path.join(folder, 'end_of_scan.log'), 'w')
        self.end_procs.append(subprocess.Popen(
            [sys.executable, '-m', 'PyGEECSPlotter.scan_data_analyzers.live_end', path],
            stdout=log, stderr=subprocess.STDOUT, env=package_env()))
        log.close()
        self._log(f"{st.label}: scan views being drawn in a separate process (end_of_scan.log)")
        return True

    def _abandoned(self, st, now):
        """No sfile, and a newer scan has been running for ``abandon_grace_s``,
        or ScanEnd came ``settle_s`` ago."""
        if st.sfile_size is not None:
            return False
        if st.ended_at is not None and now - st.ended_at >= self.settle_s:
            return True
        if st.newer_since is None:
            if any(n > st.scan and os.path.exists(scan_tdms_path(st.top_dir, n))
                   for n in self._scan_numbers(st.top_dir)):
                st.newer_since = now
            return False
        return now - st.newer_since >= self.abandon_grace_s

    def _finish(self, st):
        """Analyse the shots left over, merge everything into the sfile, run the displayers."""
        self._collect(st, wait=True)
        self._update_views(st, time.monotonic(), force=True)
        if self.job is None:
            self._log(f"{st.label}: finished (live plots only)")
            self._send_status(st, f'{st.label} finished; waiting for the next scan')
            return

        scan = ScanDataAnalyzer(sfilename=st.sfile)
        scan.load_scan_data()
        if scan.aborted:
            self._log(f"{st.label}: ended with no shots")
            return
        job = self.job
        label = job.analysis_label or ''
        parts = []
        with scan.filtered():
            if self.analyze_partial and len(self.inputs) > 1:
                self.analyzer.register_with_scan(scan, remove_missing_files=False)
                exists = [f'{d} file_exists' for d, _ in self.inputs]
                scan.filter_rows(lambda d: (d[exists] > 0).any(axis=1).to_numpy(),
                                 name='live: some input exists')
            else:
                self.analyzer.register_with_scan(scan)
            path = self.live_results_path(st.top_dir, st.scan)
            live = [pd.read_csv(path, sep='\t', float_precision='round_trip')] if os.path.exists(path) else []
            if st.pending:              # results the live file could not take
                live.append(pd.DataFrame(st.pending))
            if live:
                live = pd.concat(live, ignore_index=True).drop_duplicates(['scan', 'Shotnumber'],
                                                                          keep='last')
                live = live[live['Shotnumber'].isin(st.done)]
                scan.data = merge_results(scan.data, rename_result_columns(live, job.column_prefix, label))
                parts.append(live)
            done = st.done
            todo = scan.active_data['Shotnumber'].astype(int)
            todo = set(todo[~todo.isin(done)])
            if todo:
                self._log(f"{st.label}: analysing {len(todo)} remaining shots with the sfile")
                scan.filter_rows(lambda d: d['Shotnumber'].astype(int).isin(todo).to_numpy(),
                                 name='live: remaining')
                rest = scan.analyze_scan(
                    self.analyzer, bg=job.bg(scan), analysis_label=label, extra_info_str=label,
                    write_analyzed=job.write_analyzed, write_lineouts=job.write_lineouts,
                    parallel=job.parallel, max_workers=self.max_workers,
                    column_prefix=job.column_prefix)
                if rest is not None:
                    parts.append(rest)
        results = pd.concat(parts, ignore_index=True) if parts else None
        if results is not None and self.write:
            job.write(scan, results.drop_duplicates(['scan', 'Shotnumber'], keep='last'))
        out = online_output_dir(st.top_dir, st.scan, job.name)
        if not (self.end_process and self._launch_end(st, results, out)):
            job.display(scan, out)
            if job.write_displayed and results is not None:
                from PyGEECSPlotter.scan_data_analyzers.live_end import end_of_scan
                end_of_scan({'job': LiveJob(job.analyzer, write_displayed=True, column_prefix=job.column_prefix,
                                            analysis_label=job.analysis_label, use_bg=job.use_bg),
                             'sfile': st.sfile, 'results': results,
                             'merged': self.write and job.write_sfile, 'output_dir': out})
        self.view.close_others()
        n = 0 if results is None else results['Shotnumber'].nunique()
        self._log(f"{st.label}: finished, {n} shots ({len(st.deferred)} deferred, "
                  f"{len(st.failed)} failed live"
                  + (f", {len(st.partial)} with an input missing" if st.partial else '') + ")")
        self._send_status(st, f'{st.label} finished ({n} shots); waiting for the next scan')

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    def poll(self):
        """One step of the controller loop. Returns the scan finished in this step, if any."""
        if self._pool is None:
            self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self._start_gui()
        self._start_trigger()
        if self.listener is not None:
            self._handle_triggers()
        st = self.state
        if st is None:
            top_dir = self.top_dir
            scan = self._next_scan(top_dir)
            if scan is None:
                return None
            self._begin(top_dir, scan)
            st = self.state
        now = time.monotonic()
        ended = self._scan_ended(st, now)   # checked before the sweep, so no late file is missed
        self._find_files(st)
        if st.expected:
            self._check_triggered(st)
        self._update_complete(st, now)
        self._submit_ready(st, now)
        self._collect(st)
        gui_tdms = self.gui_process is not None and self.job is None
        if self.view.displayers or (ended and self.view.active) or gui_tdms:
            self._read_context(st, now)
        if gui_tdms:
            self._send_tdms_rows(st)
        self._update_views(st, now)
        if not ended:
            if self.listener is not None and not st.warned_silent and not st.triggered \
                    and len(st.context_rows) >= 3:
                st.warned_silent = True
                lst = self.listener
                other = (f"; {lst.rejected} Path>> messages were for another folder, e.g. "
                         f"{lst.last_rejected} (see trigger_root)" if lst.rejected else '')
                self._log(f"{st.label}: no Path>> message for this scan on {self.trigger} "
                          f"({lst.packets} packets received{other}); is it this experiment's "
                          f"group? Carrying on with folder polling")
            if self._abandoned(st, now):
                self._collect(st, wait=True)
                why = 'ScanEnd came' if st.ended_at is not None else 'a newer scan has started'
                self._log(f"{st.label}: no sfile and {why}; giving up on it"
                          + (f" ({len(st.done)} shots in "
                             f"{self.live_results_path(st.top_dir, st.scan)}, nothing merged)"
                             if self.job else ''))
                self.last_scan[st.top_dir] = st.scan
                self.state = None
            return None
        try:
            self._finish(st)
        except Exception:
            self._log(f"{st.label}: finishing failed:\n{traceback.format_exc()}")
        self.last_scan[st.top_dir] = st.scan
        self.state = None
        return st.scan

    def check_ready(self):
        """Fail fast, before watching: the analyzer's ``check_ready()`` (e.g. its
        default backgrounds or calibrations; raises with the missing files).
        Also logs a warning if the SMB cache will delay new files."""
        if self.job is not None and hasattr(self.analyzer, 'check_ready'):
            self.analyzer.check_ready()
        warning = smb_cache_warning(self.fixed_top_dir or self.experiment_dir)
        if warning:
            self._log(warning)

    def start(self):
        """Run until ``stop()`` or Ctrl+C (after ``check_ready()``, which raises
        if the analyzer can't analyse shots)."""
        self.check_ready()
        self._running = True
        self._log(f"Watching {os.path.join(self.top_dir, 'scans')} for scans to start "
                  f"({self.name}). Ctrl+C to stop.")
        errors = 0
        try:
            while self._running:
                try:
                    self.poll()
                    errors = 0
                except Exception:
                    # e.g. the network share dropped out: report, back off, carry on
                    errors += 1
                    self._log(f"poll failed ({errors} in a row):\n{traceback.format_exc()}")
                wait = min(self.poll_s * 2 ** min(errors, 5), 30.0)
                if not errors and self._trigger_pending():
                    wait = min(wait, 0.05)      # a triggered shot's files / row are due
                self.view.wait(wait, self._wake)
                self._wake.clear()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def stop(self):
        self._running = False

    def close(self):
        """Cancel queued shots and stop the threads; results written so far are kept."""
        self._running = False
        if self._pool is not None:
            st = self.state
            if st is not None:
                # drop the shots not started yet (a restarted watcher does them);
                # finish, show and write the ones running
                for shot in [s for s, f in st.submitted.items() if f.cancel()]:
                    del st.submitted[shot]
                self._collect(st, wait=True)
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        if self.gui_process is not None:
            self.gui_process.stop()
            self.gui_process = None
        self._log("Stopped.")
