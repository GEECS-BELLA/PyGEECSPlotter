# Scan classification for online analysis: hashtags in the scan info text
# (ScanInfoScanNNN.ini, e.g. "focus scan #focus #bg") choose extra jobs for
# that scan, or change the default ones. An experiment's tags live in a Python
# config file that builds one ScanClassifier and hands it to ScanWatcher.
#
# A job is anything with a ``name`` and ``run(scan, output_dir=None,
# write=True)``: a LiveJob, AveragedBackground or PlaceholderJob below, or
# your own class.

import copy
import os
import re
import warnings

import numpy as np

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
    """

    def __init__(self, jobs=(), tags=(), day=()):
        from PyGEECSPlotter.scan_data_analyzers.live_analysis import as_jobs
        self.jobs = as_jobs(jobs)
        self.tags = {t.name: t for t in tags}
        self.day = list(day)

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
    Background job (e.g. for ``#bg``): the scan's mean raw image of each
    camera, written as ``analysis/ScanNNN<diag>_averaged.png``, the name
    ``use_bg`` / ``get_bg_file_path`` looks for, so later scans use it.

    cameras : list of (diagnostic, file_ext)
    The image is saved as a plain 16-bit PNG of rounded counts, read back
    unchanged by ``read_imaq_image``. With ``write=False`` it goes to the
    run's ``output_dir`` instead of the analysis folder.
    """

    def __init__(self, cameras, name='background'):
        self.cameras = [(d, e) for d, e in cameras]
        self.name = name

    def __repr__(self):
        return f"AveragedBackground({', '.join(d for d, _ in self.cameras)})"

    def path(self, scan, diagnostic, file_ext, output_dir=None, write=True):
        folder = os.path.join(scan.top_dir, 'analysis') if write else output_dir
        return os.path.join(folder, f'Scan{int(scan.scan):03d}{diagnostic}_averaged{file_ext}')

    def run(self, scan, output_dir=None, write=True):
        import imageio
        written = []
        for diagnostic, file_ext in self.cameras:
            reader = DataReader(diagnostic=diagnostic, file_ext=file_ext)
            with scan.filtered():
                reader.register_with_scan(scan)
                if scan.n_active == 0:
                    print(f"{scan.scan_label}: no {diagnostic} images; no background written")
                    continue
                mean, _ = scan.mean_std_diagnostic(reader)
            if mean is None:
                continue
            path = self.path(scan, diagnostic, file_ext, output_dir, write)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            imageio.imwrite(path, np.clip(np.round(mean), 0, 65535).astype(np.uint16))
            print(f"Background written to {path}")
            written.append(path)
        return written


class PlaceholderJob:
    """A job that only reports that it ran (stand-in for one not written yet)."""

    def __init__(self, name, message=None):
        self.name = name
        self.message = message or 'placeholder, nothing done yet'

    def run(self, scan, output_dir=None, write=True):
        print(f"{scan.scan_label}: {self.name}: {self.message}")
        return None
