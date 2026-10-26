# Live analysis: analyse each shot while the scan is still running.
#
# LiveShotWatcher waits for a scan to start, analyses each shot as soon as its
# diagnostic file(s) and its row in the scan's .tdms are complete, appends the
# results to online_analysis/ScanNNN/<name>_live.txt as it goes, and when the
# scan ends (its sfile appears) analyses any shots it couldn't, merges the
# results into the sfile and runs the displayers. Then it waits for the next
# scan.
#
# Threading: one controller loop (the thread that calls start()) owns all the
# bookkeeping, so there are no shared-state races; a thread pool only runs
# load + analyse (+ the shot's write_analyzed files). Results are collected in
# shot order. pyplot (displayers) stays on the controller thread.

import concurrent.futures
import os
import re
import time
import traceback

import pandas as pd
import matplotlib.pyplot as plt

from PyGEECSPlotter.scan_data_analyzers.scan_data_analysis import ScanDataAnalyzer
from PyGEECSPlotter.scan_data_analyzers.live_analysis import LiveJob, online_output_dir
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
        self.submitted = {}      # shot -> future
        self.done = set()        # shots with results (this run or a previous one)
        self.deferred = set()    # waited too long for their .tdms context
        self.failed = set()
        self.pending = []        # result rows not yet written to the live file
        self.context = None      # last good .tdms frame
        self.context_size = None # .tdms size when it was read
        self.context_rows = {}   # shot -> row position in context (its last row)
        self.next_read = 0.0     # earliest time to re-read the .tdms
        self.warned_no_shots = False
        self.required = []       # required_columns as named in the .tdms frame
        self.sfile_size = None   # (size, time first seen at that size)
        self.newer_since = None  # time a newer scan was first seen started


class LiveShotWatcher:
    """
    Analyse each shot while the scan runs; finish up when the scan ends.

    Parameters
    ----------
    experiment_dir : str or Path, optional
        Folder holding the year folders; the day is today's (re-checked).
    job : LiveJob or analyzer
        The one analyzer (single- or multi-diagnostic) to run, with its
        displayers and write options (see ``LiveJob``).
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

    Per-shot results go to ``online_analysis/ScanNNN/<name>_live.txt`` as
    they come in (a restarted watcher skips the shots already there);
    displayer figures to ``online_analysis/ScanNNN/<name>/``.
    """

    def __init__(self, experiment_dir=None, job=None, top_dir=None, poll_s=1.0, max_workers=None,
                 context_timeout_s=30.0, required_columns=None, start_scan=None, write=True,
                 file_settle_s=1.0, settle_s=30.0, abandon_grace_s=60.0, analyze_partial=False,
                 missing_input_timeout_s=10.0):
        if experiment_dir is None and top_dir is None:
            raise ValueError("Give experiment_dir (watch today's folder) or top_dir.")
        if job is None:
            raise ValueError("Give the job (a LiveJob or an analyzer) to run.")
        self.experiment_dir = os.fspath(experiment_dir) if experiment_dir is not None else None
        self.fixed_top_dir = os.fspath(top_dir) if top_dir is not None else None
        self.job = job if isinstance(job, LiveJob) else LiveJob(job)
        self.analyzer = self.job.analyzer
        self.inputs = _inputs(self.analyzer)
        self.poll_s = poll_s
        self.max_workers = max_workers if self.job.parallel else 1
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
        return {'scan': st.scan, 'files': len(st.files), 'tdms shots': len(st.context_rows),
                'done': len(st.done),
                'arriving': len(set(st.files) - set(st.complete) - st.done),
                'in progress': running, 'deferred': len(st.deferred), 'failed': len(st.failed),
                'waiting': len(set(st.complete) - set(st.submitted) - st.done - st.deferred
                               - st.failed)}

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
        path = self.live_results_path(top_dir, scan)
        if os.path.exists(path):
            try:
                st.done = set(pd.read_csv(path, sep='\t')['Shotnumber'].astype(int))
            except Exception:
                pass
        self.aliases = alias_map_for_scan(top_dir, scan)
        if not self.aliases:
            self._log(f"{st.label}: no earlier sfile found for the column aliases; "
                      f".tdms columns keep their device names")
        st.required = [self.aliases.get(c, c) for c in self.required_columns]
        self.state = st
        self._log(f"{st.label}: started; analysing shots as they arrive"
                  + (f" ({len(st.done)} already done)" if st.done else ''))

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

    def _settled(self, st, path, now):
        """The file exists and has kept its (non-zero) size for ``file_settle_s``."""
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        if st.sizes.get(path, (None,))[0] != size:
            st.sizes[path] = (size, now)
        return size > 0 and now - st.sizes[path][1] >= self.file_settle_s

    def _update_complete(self, st, now):
        """Shots whose every input file exists and kept its size for ``file_settle_s``
        (and, with ``analyze_partial``, shots given up waiting for a missing input)."""
        n_inputs = len(self.inputs)
        newest_full = max((s for s, p in st.files.items() if len(p) == n_inputs), default=0)
        for shot, paths in st.files.items():
            if shot in st.complete or shot in st.done:
                continue
            settled = all([self._settled(st, path, now) for path in paths.values()])
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
        waiting = [s for s in sorted(st.complete) if s not in st.submitted and s not in st.done
                   and s not in st.deferred and s not in st.failed]
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
            st.submitted[shot] = self._pool.submit(
                helper._process_row_and_write, self.analyzer, st.bg, pd.Series(row),
                self.job.write_analyzed, self.job.write_lineouts)

    def _collect(self, st, wait=False):
        """Results of finished shots, in shot order, appended to the live results file.

        A shot with results is ``done``; its row stays in ``st.pending`` until
        the file write succeeds (retried on the next poll)."""
        for shot in sorted(st.submitted):
            future = st.submitted[shot]
            if not wait and not future.done():
                break                   # keep shot order: stop at the first unfinished
            del st.submitted[shot]
            try:
                _, _, results, _ = future.result()
                st.pending.append({'scan': st.scan, 'Shotnumber': shot, **results})
                st.done.add(shot)
            except Exception:
                st.failed.add(shot)
                self._log(f"{st.label}: shot {shot} failed (retried at the end of the scan):\n"
                          f"{traceback.format_exc()}")
        if st.pending:
            try:
                self._append_live(st, pd.DataFrame(st.pending))
                st.pending = []
            except Exception as err:
                self._log(f"{st.label}: could not write the live results "
                          f"({len(st.pending)} shots kept, retrying): {err!r}")

    def _append_live(self, st, df):
        path = self.live_results_path(st.top_dir, st.scan)
        with file_lock(path):
            if os.path.exists(path):
                df = pd.concat([pd.read_csv(path, sep='\t'), df], ignore_index=True)
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

    def _abandoned(self, st, now):
        """No sfile, and a newer scan has been running for ``abandon_grace_s``."""
        if st.sfile_size is not None:
            return False
        if st.newer_since is None:
            if any(n > st.scan and os.path.exists(scan_tdms_path(st.top_dir, n))
                   for n in self._scan_numbers(st.top_dir)):
                st.newer_since = now
            return False
        return now - st.newer_since >= self.abandon_grace_s

    def _finish(self, st):
        """Analyse the shots left over, merge everything into the sfile, run the displayers."""
        self._collect(st, wait=True)

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
            live = [pd.read_csv(path, sep='\t')] if os.path.exists(path) else []
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
        job.display(scan, online_output_dir(st.top_dir, st.scan, job.name))
        plt.close('all')
        n = 0 if results is None else results['Shotnumber'].nunique()
        self._log(f"{st.label}: finished, {n} shots ({len(st.deferred)} deferred, "
                  f"{len(st.failed)} failed live"
                  + (f", {len(st.partial)} with an input missing" if st.partial else '') + ")")

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    def poll(self):
        """One step of the controller loop. Returns the scan finished in this step, if any."""
        if self._pool is None:
            self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
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
        self._update_complete(st, now)
        self._submit_ready(st, now)
        self._collect(st)
        if not ended:
            if self._abandoned(st, now):
                self._collect(st, wait=True)
                self._log(f"{st.label}: no sfile and a newer scan has started; giving up on it "
                          f"({len(st.done)} shots in {self.live_results_path(st.top_dir, st.scan)}, "
                          f"nothing merged)")
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

    def start(self):
        """Run until ``stop()`` or Ctrl+C."""
        self._running = True
        self._log(f"Watching {os.path.join(self.top_dir, 'scans')} for scans to start "
                  f"({self.job.name}). Ctrl+C to stop.")
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
                time.sleep(min(self.poll_s * 2 ** min(errors, 5), 30.0))
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
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
            st = self.state
            if st is not None:
                # keep the shots that finished while shutting down, so a
                # restarted watcher doesn't redo them; cancelled ones it will
                for shot in [s for s, f in st.submitted.items() if f.cancelled()]:
                    del st.submitted[shot]
                self._collect(st, wait=True)
        self._log("Stopped.")
