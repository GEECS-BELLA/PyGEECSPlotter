# Online analysis: run analyzers (and displayers) on each scan as it finishes.
#
# ScanWatcher polls a day's analysis folder for new sfiles. When a scan's
# sfile appears and stops changing, every LiveJob runs on it: analyze_scan
# (in threads unless the analyzer is not thread-safe), the results merged
# into the sfile (and optionally scans/ScanNNN/ScanDataScanNNN.txt) under a
# file lock, then the job's displayers, saved to
# <top_dir>/online_analysis/ScanNNN/<diagnostic>/.
#
# LiveShotWatcher (live_shot_analysis.py) analyses shots while a scan is still
# running instead.

import os
import re
import time
import traceback

import matplotlib.pyplot as plt

from PyGEECSPlotter.scan_data_analyzers.scan_data_analysis import ScanDataAnalyzer
import PyGEECSPlotter.utils.navigation_utils as nav_utils

ONLINE_DIR = 'online_analysis'
_SFILE_RE = re.compile(r's(\d+)\.txt$')


def online_output_dir(top_dir, scan, name=None, make_dir=True):
    """``<top_dir>/online_analysis/ScanNNN[/name]``, created if ``make_dir``."""
    out = os.path.join(os.fspath(top_dir), ONLINE_DIR, f'Scan{int(scan):03d}')
    if name:
        out = os.path.join(out, name)
    if make_dir:
        os.makedirs(out, exist_ok=True)
    return out


def _flag(value, analyzer, key, default):
    """``value`` if set, else ``analyzer.analyzer_dict[key]``, else ``default``."""
    if value is not None:
        return value
    return getattr(analyzer, 'analyzer_dict', {}).get(key, default)


class LiveJob:
    """
    One analyzer to run on each scan, plus what to do with its results.

    Every option left as None falls back to the analyzer's ``analyzer_dict``
    flag of the old watcher (in brackets), so existing analyzers behave as
    before:

    analyzer : DiagnosticAnalyzer, ColumnMathAnalyzer or MultiDiagnosticAnalyzer
    displayers : list of ScanDisplayer
        Run after the analysis, on the scan with the new columns; each figure
        is saved to ``online_analysis/ScanNNN/<name>/`` and closed.
    write_sfile : bool [``add_columns_to_masterlog``, default True]
        Merge the results into ``analysis/sN.txt``.
    update_scan_data_file : bool [``update_scan_data_file``, default False]
        Also merge them into ``scans/ScanNNN/ScanDataScanNNN.txt``.
    use_bg : bool [``use_bg``, default False]
        Use the day's ``*<diagnostic>_averaged<ext>`` background.
    write_analyzed, write_lineouts : bool [same keys, default False]
        Per-shot files, written to ``analysis/ScanNNN`` as by ``analyze_scan``.
    analysis_label : str [``analysis_label``]
        Appended to the result column names.
    column_prefix : str
        Prefix of the result column names. Default: the analyzer's
        ``diagnostic`` (none for column maths), as the old watcher wrote.
    parallel : bool
        Default: the analyzer's ``thread_safe``.
    name : str
        Output folder name. Default: ``output_diagnostic``, ``diagnostic``,
        ``friendly_name`` or the class name, whichever is set first.
    """

    def __init__(self, analyzer, displayers=(), write_sfile=None, update_scan_data_file=None,
                 use_bg=None, write_analyzed=None, write_lineouts=None, analysis_label=None,
                 column_prefix=None, parallel=None, name=None):
        a = analyzer
        self.analyzer = a
        self.displayers = list(displayers)
        self.write_sfile = _flag(write_sfile, a, 'add_columns_to_masterlog', True)
        self.update_scan_data_file = _flag(update_scan_data_file, a, 'update_scan_data_file', False)
        self.use_bg = _flag(use_bg, a, 'use_bg', False)
        self.write_analyzed = _flag(write_analyzed, a, 'write_analyzed', False)
        self.write_lineouts = _flag(write_lineouts, a, 'write_lineouts', False)
        self.analysis_label = _flag(analysis_label, a, 'analysis_label', None)
        self.column_prefix = column_prefix if column_prefix is not None else getattr(a, 'diagnostic', None)
        self.parallel = getattr(a, 'thread_safe', True) if parallel is None else parallel
        self.name = name or next(
            (n for n in (getattr(a, 'output_diagnostic', None), getattr(a, 'diagnostic', None),
                         getattr(a, 'friendly_name', None)) if n),
            type(a).__name__)

    def __repr__(self):
        return (f"LiveJob({self.name}: {type(self.analyzer).__name__}, "
                f"{len(self.displayers)} displayers)")

    def bg(self, scan):
        """Background path for this scan, or None."""
        a = self.analyzer
        if not self.use_bg or getattr(a, 'diagnostic', None) is None:
            return None
        return scan.get_bg_file_path(a.diagnostic, a.file_ext or '.png')

    def run(self, scan, output_dir=None, write=True):
        """
        Analyse ``scan`` (a loaded ScanDataAnalyzer), write the results
        (if ``write`` and the job asks for it) and run the displayers.
        Returns the results frame (or None).
        """
        a = self.analyzer
        a.register_with_scan(scan)
        results = scan.analyze_scan(
            a, bg=self.bg(scan), write_columns_to_sfile=False,
            analysis_label=self.analysis_label or '', write_analyzed=self.write_analyzed,
            write_lineouts=self.write_lineouts, parallel=self.parallel,
            column_prefix=self.column_prefix,
            extra_info_str=self.analysis_label or '')
        if write and results is not None:
            self.write(scan, results)
        self.display(scan, output_dir)
        return results

    def write(self, scan, results):
        """Merge ``results`` into the sfile and / or ScanData file, as configured."""
        if self.write_sfile:
            scan.merge_data_frame_to_sfile(results, self.column_prefix, overwrite_columns=True,
                                           analysis_label=self.analysis_label)
        if self.update_scan_data_file:
            target = os.path.join(scan.top_dir, 'scans', f'Scan{int(scan.scan):03d}',
                                  f'ScanDataScan{int(scan.scan):03d}.txt')
            if os.path.exists(target):
                scan.merge_data_frame_to_sfile(results, self.column_prefix, overwrite_columns=True,
                                               analysis_label=self.analysis_label,
                                               target_filename=target)

    def display(self, scan, output_dir=None):
        """Run each displayer on ``scan``, saving to ``output_dir``; errors are reported, not raised."""
        for displayer in self.displayers:
            try:
                fig, _ = scan.display_scan(displayer, save=True, output_dir=output_dir)
                plt.close(fig)
            except Exception:
                print(f"Displayer {type(displayer).__name__} failed on {scan.scan_label}:")
                print(traceback.format_exc())
            finally:
                for extra in getattr(displayer, 'figures', []) or []:
                    plt.close(extra[-1] if isinstance(extra, tuple) else extra)


def as_jobs(jobs):
    """A list of LiveJob, wrapping bare analyzers in a default LiveJob."""
    if isinstance(jobs, LiveJob) or hasattr(jobs, 'analyze_data'):
        jobs = [jobs]
    return [j if isinstance(j, LiveJob) else LiveJob(j) for j in jobs]


class ScanWatcher:
    """
    Watch a day's ``analysis`` folder and run ``jobs`` on every scan as it
    finishes (when its ``sN.txt`` appears and has stopped changing).

    Parameters
    ----------
    experiment_dir : str or Path, optional
        Folder holding the ``Y2026`` year folders. The day watched is today's,
        re-checked every poll, so a run past midnight moves to the new day.
    jobs : list of LiveJob (or analyzers)
        Run in order on each scan; put cheap ones first.
    top_dir : str or Path, optional
        Watch this day folder instead (fixed).
    poll_s : float
        Seconds between polls of the analysis folder.
    start_scan : int, optional
        First scan to analyse. Default: only scans finishing after ``start()``.
    search_replace_filename, column_math_filename : optional
        Passed to ``load_scan_data``.
    write : bool
        False: analyse and plot but never write to the sfile / ScanData files
        (for testing on real data).

    Uses polling rather than file-system events, which are unreliable on
    network shares. ``reanalyze(scans)`` runs the jobs on scans already taken.
    """

    def __init__(self, experiment_dir=None, jobs=(), top_dir=None, poll_s=2.0, start_scan=None,
                 search_replace_filename=None, column_math_filename=None, write=True):
        if experiment_dir is None and top_dir is None:
            raise ValueError("Give experiment_dir (watch today's folder) or top_dir.")
        self.experiment_dir = os.fspath(experiment_dir) if experiment_dir is not None else None
        self.fixed_top_dir = os.fspath(top_dir) if top_dir is not None else None
        self.jobs = as_jobs(jobs)
        self.poll_s = poll_s
        self.start_scan = start_scan
        self.search_replace_filename = search_replace_filename
        self.column_math_filename = column_math_filename
        self.write = write
        self.status = ''
        self._seen_days = set()         # top dirs whose earlier scans were skipped
        self.done = set()               # (top_dir, scan) analysed or skipped
        self._sizes = {}                # sfile -> size at the last poll
        self._running = False

    # ------------------------------------------------------------------
    @property
    def top_dir(self):
        return self.fixed_top_dir or nav_utils.get_todays_top_dir(self.experiment_dir)

    @property
    def analysis_dir(self):
        return os.path.join(self.top_dir, 'analysis')

    def _log(self, message):
        self.status = message
        print(time.strftime('[%H:%M:%S] ') + message)

    @staticmethod
    def _sfiles(top_dir):
        """``{scan: path}`` for the sfiles in ``top_dir/analysis``."""
        out = {}
        try:
            with os.scandir(os.path.join(top_dir, 'analysis')) as it:
                for entry in it:
                    m = _SFILE_RE.fullmatch(entry.name)
                    if m and entry.is_file():
                        out[int(m.group(1))] = entry.path
        except FileNotFoundError:
            pass
        return out

    def poll(self):
        """
        One look at the analysis folder: run the jobs on each new sfile that
        has stopped changing since the last poll. Returns the scans analysed.
        """
        top_dir = self.top_dir
        sfiles = self._sfiles(top_dir)
        if top_dir not in self._seen_days:
            # Scans already finished when watching starts (or before
            # start_scan) are skipped.
            self._seen_days.add(top_dir)
            self.done.update((top_dir, n) for n in sfiles
                             if self.start_scan is None or n < self.start_scan)
        ran = []
        for scan, path in sorted(sfiles.items()):
            if (top_dir, scan) in self.done:
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if self._sizes.get(path) != size:      # new, or still being written
                self._sizes[path] = size
                continue
            self.done.add((top_dir, scan))
            self.run_scan(scan, top_dir=top_dir)
            ran.append(scan)
        return ran

    def start(self):
        """Poll until ``stop()`` or Ctrl+C."""
        self._running = True
        self._log(f"Watching {self.analysis_dir} for new scans "
                  f"({len(self.jobs)} jobs). Ctrl+C to stop.")
        try:
            while self._running:
                self.poll()
                time.sleep(self.poll_s)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self._log("Stopped watching.")

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    def load_scan(self, scan, top_dir=None):
        """A loaded ScanDataAnalyzer for ``scan`` of ``top_dir`` (default: the watched day)."""
        top_dir = top_dir or self.top_dir
        s = ScanDataAnalyzer(sfilename=nav_utils.get_sfilename_from_top_dir(top_dir, scan))
        s.load_scan_data(search_replace_filename=self.search_replace_filename,
                         column_math_filename=self.column_math_filename)
        return s

    def run_scan(self, scan, top_dir=None):
        """Run every job on one scan. A failing job is reported and the next one runs."""
        top_dir = top_dir or self.top_dir
        label = f"Scan{int(scan):03d}"
        self._log(f"{label}: analysing ({len(self.jobs)} jobs)")
        try:
            s = self.load_scan(scan, top_dir)
        except Exception:
            self._log(f"{label}: could not load the sfile")
            print(traceback.format_exc())
            return None
        if s.aborted:
            self._log(f"{label}: no shots, skipped")
            return s
        for job in self.jobs:
            t0 = time.monotonic()
            try:
                with s.filtered():          # each job's missing-file rules apply to it alone
                    results = job.run(s, output_dir=online_output_dir(top_dir, scan, job.name),
                                      write=self.write)
                n = 0 if results is None else len(results)
                self._log(f"{label}: {job.name}: {n} shots ({time.monotonic() - t0:.1f} s)")
            except Exception:
                self._log(f"{label}: {job.name} failed")
                print(traceback.format_exc())
        plt.close('all')
        return s

    def reanalyze(self, scans=None, top_dir=None):
        """Run the jobs on ``scans`` (default: every sfile of the day), in order."""
        top_dir = top_dir or self.top_dir
        available = self._sfiles(top_dir)
        scans = sorted(available) if scans is None else [int(n) for n in scans]
        for scan in scans:
            if scan not in available:
                print(f"No sfile for Scan{scan:03d}; skipped.")
                continue
            self.run_scan(scan, top_dir=top_dir)


# ----------------------------------------------------------------------
# Old names (the pre-0.1 watcher), kept so existing scripts still run.
# ----------------------------------------------------------------------
class DirectoryWatcherSFile(ScanWatcher):
    """Old name: ``DirectoryWatcherSFile(experiment_dir, analyzers)``; use ``ScanWatcher``."""

    def __init__(self, experiment_dir, analyzers, search_replace_filename=None,
                 column_math_filename=None):
        super().__init__(experiment_dir=experiment_dir, jobs=analyzers,
                         search_replace_filename=search_replace_filename,
                         column_math_filename=column_math_filename)


class DirectoryWatcherSFileTester(ScanWatcher):
    """Old name: watch a fixed ``top_dir``; use ``ScanWatcher(top_dir=...)``."""

    def __init__(self, top_dir, analyzers, search_replace_filename=None,
                 column_math_filename=None):
        super().__init__(top_dir=top_dir, jobs=analyzers,
                         search_replace_filename=search_replace_filename,
                         column_math_filename=column_math_filename)


class SFileReanalyzer(ScanWatcher):
    """Old name: re-analyse a day's scans; use ``ScanWatcher(top_dir=...).reanalyze()``."""

    def __init__(self, top_dir, analyzers, search_replace_filename=None,
                 column_math_filename=None):
        super().__init__(top_dir=top_dir, jobs=analyzers,
                         search_replace_filename=search_replace_filename,
                         column_math_filename=column_math_filename)

    def analyze_all_scans(self, start_scan=0, end_scan=1000):
        scans = [n for n in self._sfiles(self.top_dir) if start_scan <= n <= end_scan]
        self.reanalyze(scans)
