# Scan classification for online analysis: hashtags in the scan info text
# (ScanInfoScanNNN.ini, e.g. "focus scan #focus #bg") choose extra jobs for
# that scan, or change the default ones. An experiment's tags live in a Python
# config file that builds one ScanClassifier and hands it to ScanWatcher.
#
# A job is anything with a ``name`` and ``run(scan, output_dir=None,
# write=True)``: a LiveJob, DisplayJob, AveragedBackground or PlaceholderJob
# below, or your own class.

import copy
import os
import re
import warnings

import numpy as np
import pandas as pd

from PyGEECSPlotter.analyzers.data_reader import DataReader

_TAG_RE = re.compile(r'#(\w+)')


def scan_tags(text):
    """Lower-case tags (``#word``) in ``text``, sorted: ``'Run 3 #Focus #bg'`` -> ``['bg', 'focus']``."""
    return sorted({t.lower() for t in _TAG_RE.findall(text or '')})


def read_scan_tags(scan):
    """Tags of ``scan`` (a ScanDataAnalyzer), from its ScanInfo .ini. A
    missing or unreadable file gives no tags, with a warning."""
    try:
        return scan_tags(scan.get_scan_info_text())
    except Exception as e:
        warnings.warn(f"{scan.scan_label}: could not read the scan info ({e}); no tags.",
                      stacklevel=2)
        return []


class Tag:
    """
    What one hashtag does.

    name : str
        The tag, without ``#`` (matched case-insensitively).
    jobs : list of jobs
        Run on tagged scans, after the default jobs.
    modify : callable, optional
        ``modify(jobs) -> jobs``: change the default jobs for tagged scans,
        e.g. ``lambda jobs: [tweak_job(j, analyzer_dict={'use_bg': False})
        for j in jobs]``. Must return new objects rather than change the
        ones it is given (see ``tweak_job``).
    replace_default : bool
        True: tagged scans run only this tag's jobs (and other tags'), not
        the default ones.
    """

    def __init__(self, name, jobs=(), modify=None, replace_default=False):
        self.name = name.lstrip('#').lower()
        self.jobs = list(jobs)
        self.modify = modify
        self.replace_default = replace_default

    def __repr__(self):
        return f"Tag(#{self.name}: {len(self.jobs)} jobs)"


class ScanClassifier:
    """
    The online analysis of one experiment: default jobs for every scan,
    what each scan tag adds or changes, and the day trackers.

    jobs : list of jobs (or analyzers)
        Run on every scan, in order.
    tags : list of Tag
        Applied in this order to scans that carry them.
    day : list of day trackers
        Updated after each scan (see ``displayers.day``).
    config_file : str, optional
        The config file this classifier came from; its path and SHA-256 go
        into every result's provenance record.
    """

    def __init__(self, jobs=(), tags=(), day=(), config_file=None):
        from PyGEECSPlotter.scan_data_analyzers.live_analysis import as_jobs
        self.jobs = as_jobs(jobs)
        self.tags = {t.name: t for t in tags}
        self.day = list(day)
        self.config_file = os.fspath(config_file) if config_file is not None else None

    def provenance(self):
        """``{'config': path, 'config_sha256': hash}`` (empty without a config file)."""
        if self.config_file is None:
            return {}
        from PyGEECSPlotter.utils.provenance import file_hash
        return {'config': os.path.abspath(self.config_file),
                'config_sha256': file_hash(self.config_file)}

    def __repr__(self):
        tags = ' '.join(f'#{n}' for n in self.tags)
        return f"ScanClassifier({len(self.jobs)} jobs; tags {tags or '-'}; {len(self.day)} day trackers)"

    def jobs_for(self, scan, tags=None):
        """
        ``(jobs, tags)`` for ``scan``: the default jobs (unless a tag replaces
        them), changed by each tag's ``modify``, then each tag's own jobs.
        ``tags`` defaults to the scan's (``read_scan_tags``).
        """
        tags = read_scan_tags(scan) if tags is None else tags
        known = [self.tags[t] for t in self.tags if t in tags]     # config order
        unknown = [t for t in tags if t not in self.tags]
        if unknown:
            print(f"{scan.scan_label}: tags not in the config, ignored: "
                  + ' '.join(f'#{t}' for t in unknown))
        jobs = [] if any(t.replace_default for t in known) else list(self.jobs)
        for tag in known:
            if tag.modify is not None:
                jobs = list(tag.modify(jobs))
        for tag in known:
            jobs += tag.jobs
        return jobs, [t.name for t in known]


def tweak_job(job, analyzer_dict=None, **attributes):
    """
    A copy of a LiveJob with ``analyzer_dict`` entries updated (on a copy of
    its analyzer) and / or LiveJob attributes replaced, e.g.
    ``tweak_job(job, analyzer_dict={'pipeline': [...]}, use_bg=False)``.
    The original job and analyzer are left unchanged. Displayers built
    around the original analyzer keep using it.
    """
    new = copy.copy(job)
    if analyzer_dict:
        a = copy.copy(job.analyzer)
        a.analyzer_dict = {**job.analyzer.analyzer_dict, **analyzer_dict}
        new.analyzer = a
    for key, value in attributes.items():
        if not hasattr(new, key):
            raise AttributeError(f"{type(job).__name__} has no attribute {key!r}")
        setattr(new, key, value)
    return new


class AveragedBackground:
    """
    Background job (e.g. for ``#bg``): the scan's mean raw data of each
    diagnostic, written as ``analysis/ScanNNN<diag>_averaged<ext>``, the
    name ``use_bg`` / ``get_bg_file_path`` looks for, so later scans use it.

    diagnostics : list, optional
        Diagnostic names, or ``(diagnostic, file_ext)`` pairs. Default: every
        diagnostic folder of the scan (``scans/ScanNNN/<diag>``) whose shot
        files have one of ``file_exts``.
    file_exts : extensions averaged when ``diagnostics`` is not given.
    overwrite : bool
        False (default): a file of that name already there (e.g. GEECS's
        own average of a background scan) is kept, and that diagnostic
        skipped.
    max_workers : threads loading the diagnostics in parallel.

    Files are read with ``DiagnosticAnalyzer``'s default ``load_data``.
    Images are saved as plain 16-bit PNGs of rounded counts (read back
    unchanged by ``read_imaq_image``); tables as tab-separated text with
    the input's columns (and header row, if it had one), numeric columns
    averaged. Shots whose data doesn't match the first shot's shape are
    left out. With ``write=False`` the files go to the run's ``output_dir``.
    """

    def __init__(self, diagnostics=None, file_exts=('.png', '.txt'), name='background',
                 overwrite=False, max_workers=8):
        self.diagnostics = None if diagnostics is None else [
            tuple(d) if isinstance(d, (tuple, list)) else (d, None) for d in diagnostics]
        self.file_exts = tuple(e.lower() for e in file_exts)
        self.name = name
        self.overwrite = overwrite
        self.max_workers = max_workers

    def __repr__(self):
        which = 'all ' + '/'.join(self.file_exts) if self.diagnostics is None else \
            ', '.join(d for d, _ in self.diagnostics)
        return f"AveragedBackground({which})"

    def path(self, scan, diagnostic, file_ext, output_dir=None, write=True):
        folder = os.path.join(scan.top_dir, 'analysis') if write else output_dir
        return os.path.join(folder, f'Scan{int(scan.scan):03d}{diagnostic}_averaged{file_ext}')

    def find_diagnostics(self, scan):
        """``[(diagnostic, ext)]``: the scan's diagnostic folders and their shot-file
        extension (the commonest among ``ScanNNN_<diag>_*`` files), if in ``file_exts``."""
        scan_dir = os.path.join(scan.top_dir, 'scans', scan.scan_label)
        found = []
        try:
            folders = sorted(e.name for e in os.scandir(scan_dir) if e.is_dir())
        except FileNotFoundError:
            return found
        for diag in folders:
            prefix = f'{scan.scan_label}_{diag}_'
            counts = {}
            with os.scandir(os.path.join(scan_dir, diag)) as it:
                for e in it:
                    if e.name.startswith(prefix):
                        ext = os.path.splitext(e.name)[1].lower()
                        counts[ext] = counts.get(ext, 0) + 1
            if counts:
                ext = max(counts, key=counts.get)
                if ext in self.file_exts:
                    found.append((diag, ext))
        return found

    def _diagnostics(self, scan):
        if self.diagnostics is None:
            return self.find_diagnostics(scan)
        known = dict(self.find_diagnostics(scan)) if any(e is None for _, e in self.diagnostics) else {}
        return [(d, e or known.get(d)) for d, e in self.diagnostics if (e or known.get(d))]

    @staticmethod
    def _mean(files):
        """Mean of the files' data: ``(mean, first_frame_or_None, n_used, n_skipped)``."""
        from PyGEECSPlotter.analyzers.diagnostic_analyzer import load_shot_file
        total, first, n, skipped = None, None, 0, 0
        for f in files:
            try:
                d = load_shot_file(f)
            except Exception:
                skipped += 1
                continue
            if d is None:
                continue
            if hasattr(d, 'columns'):
                if first is None:
                    first = d
                elif list(d.columns) != list(first.columns) or d.shape != first.shape:
                    skipped += 1
                    continue
                values = d.apply(pd.to_numeric, errors='coerce').to_numpy(float)
            else:
                values = np.asarray(d, dtype=float)
            if total is not None and values.shape != total.shape:
                skipped += 1
                continue
            total = values if total is None else total + values
            n += 1
        return (None if total is None else total / n), first, n, skipped

    def _write(self, path, mean, first):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if first is None:
            import imageio
            imageio.imwrite(path, np.clip(np.round(mean), 0, 65535).astype(np.uint16))
        else:
            table = pd.DataFrame(mean, columns=first.columns)
            for c in first.columns:          # non-numeric columns: keep the first shot's
                if not pd.api.types.is_numeric_dtype(first[c]):
                    table[c] = first[c].values
            header = not all(isinstance(c, (int, np.integer)) for c in first.columns)
            table.to_csv(path, sep='\t', index=False, header=header)

    def run(self, scan, output_dir=None, write=True):
        import concurrent.futures
        todo, kept = [], []
        for diagnostic, file_ext in self._diagnostics(scan):
            path = self.path(scan, diagnostic, file_ext, output_dir, write)
            if os.path.exists(path) and not self.overwrite:
                kept.append(os.path.basename(path))
                continue
            reader = DataReader(diagnostic=diagnostic, file_ext=file_ext)
            with scan.filtered():                    # registering changes the scan: serial
                reader.register_with_scan(scan, remove_missing_files=True)
                files = list(scan.active_data[f'{diagnostic} file_list'])
            if files:
                todo.append((diagnostic, path, files))
        if kept:
            print(f"{scan.scan_label}: kept {len(kept)} existing averaged files "
                  f"(overwrite=True to replace): {', '.join(kept)}")
        written = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._mean, files): (diag, path) for diag, path, files in todo}
            for future in concurrent.futures.as_completed(futures):
                diag, path = futures[future]
                try:
                    mean, first, n, skipped = future.result()
                    if mean is None:
                        print(f"{scan.scan_label}: {diag}: no readable files; no background")
                        continue
                    self._write(path, mean, first)
                    written.append(path)
                    print(f"Background ({n} shots{f', {skipped} skipped' if skipped else ''}) "
                          f"written to {path}")
                except Exception as e:
                    print(f"{scan.scan_label}: {diag}: background failed ({type(e).__name__}: {e})")
        print(f"{scan.scan_label}: {len(written)} background files written")
        return written


class DisplayJob:
    """
    Run displayers on the scan as it is, without analysing again: e.g. a
    tag's extra figures from columns the default jobs have just added.

    displayers : list of ScanDisplayer
    register : list of analyzers / readers whose file lists the displayers
        need besides their own ``analyzer`` (which is registered anyway).
    name : output folder name, ``analysis/ScanNNN/online/<name>/``.
    """

    def __init__(self, displayers, register=(), name='figures'):
        self.displayers = list(displayers)
        self.register = list(register)
        self.name = name

    def __repr__(self):
        return f"DisplayJob({self.name}: {len(self.displayers)} displayers)"

    def run(self, scan, output_dir=None, write=True):
        from PyGEECSPlotter.scan_data_analyzers.live_analysis import run_displayers
        for reader in self.register:
            reader.register_with_scan(scan)
        run_displayers(scan, self.displayers, output_dir)
        return None


class PlaceholderJob:
    """A job that only reports that it ran (stand-in for one not written yet)."""

    def __init__(self, name, message=None):
        self.name = name
        self.message = message or 'placeholder, nothing done yet'

    def run(self, scan, output_dir=None, write=True):
        print(f"{scan.scan_label}: {self.name}: {self.message}")
        return None
