"""Provenance of analysis results: what produced the numbers in an sfile.

``analysis_provenance(analyzer, ...)`` gathers, as a JSON-ready dict:
- when / where: UTC time, host, user;
- the analyzer: class, ``analyzer_dict`` and its public settings (plain-type
  attributes, so constructor arguments such as a column name, a background
  file or a scale factor are kept, not just ``analyzer_dict``);
- the code: version, git commit and "modified" flag of PyGEECSPlotter and of
  the analyzer's own package (when it is a git checkout), since the same
  settings give different numbers once the code that reads them changes;
- the scan and anything ``extra`` (e.g. the watcher config and scan tags).

``merge_data_frame_to_sfile(provenance=...)`` writes it next to the summary
file as ``Scan003_<diag>[_<label>]_provenance.json``.
"""

import datetime
import getpass
import hashlib
import importlib
import os
import socket
import subprocess
import sys

import numpy as np

_PLAIN = (str, int, float, bool, type(None))
_GIT_CACHE = {}


def _plain(value, depth=0):
    """``value`` as JSON-ready data; arrays and objects become a short description."""
    if isinstance(value, _PLAIN):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return f'<ndarray {value.shape} {value.dtype}>'
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if depth < 4 and isinstance(value, (list, tuple)):
        return [_plain(v, depth + 1) for v in value]
    if depth < 4 and isinstance(value, dict):
        return {str(k): _plain(v, depth + 1) for k, v in value.items()}
    return f'<{type(value).__name__}>'


def analyzer_settings(analyzer):
    """Public plain-type attributes of ``analyzer`` (``analyzer_dict`` kept separately)."""
    out = {}
    for key, value in sorted(vars(analyzer).items()):
        if key.startswith('_') or key == 'analyzer_dict' or callable(value):
            continue
        out[key] = _plain(value)
    return out


def _git_info(path):
    """``{'git_commit', 'git_modified'}`` of the checkout holding ``path``, or None."""
    folder = os.path.dirname(os.path.abspath(path))
    if folder in _GIT_CACHE:
        return _GIT_CACHE[folder]
    info = None
    try:
        def git(*args):
            return subprocess.run(['git', '-C', folder, *args], capture_output=True, text=True,
                                  timeout=10, check=True).stdout.strip()
        commit = git('rev-parse', 'HEAD')
        root = git('rev-parse', '--show-toplevel')
        modified = bool(git('status', '--porcelain', '--untracked-files=no'))
        info = {'git_commit': commit, 'git_modified': modified, 'git_root': root}
    except Exception:
        pass
    _GIT_CACHE[folder] = info
    return info


def package_code_info(module_name):
    """Version and git state of the top-level package of ``module_name``."""
    top = module_name.split('.')[0]
    try:
        module = sys.modules.get(top) or importlib.import_module(top)
    except Exception:
        return {'package': top}
    path = getattr(module, '__file__', None)
    info = {'package': top, 'version': getattr(module, '__version__', None),
            'path': os.path.dirname(path) if path else None}
    git = _git_info(path) if path else None
    info.update(git or {'git_commit': None, 'git_modified': None})
    return info


def file_hash(path):
    """SHA-256 of a file (e.g. a watcher config), or None if unreadable."""
    try:
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def analysis_provenance(analyzer, scan=None, extra=None):
    """
    Dict describing what produced a set of results (``merge_data_frame_to_sfile``
    adds the columns written and the shot count when it saves it).

    analyzer : the analyzer whose results these are
    scan : ScanDataAnalyzer, optional (scan number and sfile)
    extra : dict, optional (e.g. ``{'config': ..., 'tags': [...]}``)
    """
    cls = type(analyzer)
    packages = ['PyGEECSPlotter']
    if cls.__module__.split('.')[0] not in packages:
        packages.append(cls.__module__)
    out = {
        'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'host': socket.gethostname(),
        'user': getpass.getuser(),
        'analyzer': f'{cls.__module__}.{cls.__qualname__}',
        'analyzer_dict': _plain(getattr(analyzer, 'analyzer_dict', {})),
        'analyzer_settings': analyzer_settings(analyzer),
        'code': [package_code_info(p) for p in packages],
    }
    if scan is not None:
        out['scan'] = int(scan.scan) if scan.scan is not None else None
        out['sfile'] = scan.sfilename
    if extra:
        out.update(_plain(extra))
    return out
