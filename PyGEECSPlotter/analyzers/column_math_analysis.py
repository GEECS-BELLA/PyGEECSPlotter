import os
import threading
import warnings

import numpy as np
import pandas as pd

from PyGEECSPlotter.analyzers.diagnostic_analyzer import DiagnosticAnalyzer


class ColumnMathAnalyzer(DiagnosticAnalyzer):
    """
    Base class for analyzers that compute new columns from existing scan data
    without loading any diagnostic file.

    Subclass this and implement
    ``analyze_data(self, data, *, bg=None, context=None)``. The ``context``
    dict is the full scan-data row (column_name -> value). Use it to read
    existing columns and return new ones in ``results``.

    Returns ``(None, results, {})`` — no per-shot image data, no auxiliary
    outputs.
    """

    def __init__(self, friendly_name=None):
        super().__init__(
            diagnostic=None,
            file_ext=None,
            analyzer_dict={
                "add_columns_to_masterlog": True,
                "write_analyzed": False,
                "analyze_raw_data": False,
                "update_scan_data_file": True,
            },
            display_dict={},
            output_diagnostic=None,
            output_file_ext=None,
        )
        self.friendly_name = friendly_name if friendly_name is not None else self.__class__.__name__

    def load_data(self, filename):
        return None

    def analyze_data(self, data, bg=None, context=None):
        """Override in subclasses to compute new columns from ``context``."""
        return None, {}, {}


class ControlsAnalyzer(ColumnMathAnalyzer):
    """
    New sfile columns from search / replace and column-math controls files,
    as an analyzer: ``analyze_scan`` returns them as ``add_columns_df`` and
    ``write_columns_to_sfile=True`` (or a watcher ``LiveJob``) writes them.

    column_math, search_replace : optional
        Anything ``load_scan_data`` takes: a path, a ``(path, parse_from)``
        tuple, an inline list of dicts, or a list of these (applied in
        order; search / replace first, then column math).
    parse_from : {'auto', 'python', 'labview'}
        Format of files given without one.
    name : str
        Names the summary / provenance files (``Scan003_<name>_Summary.txt``)
        and the job; result columns get no prefix, they keep the names the
        controls give them.

    Each shot's row gets the search / replace renames, then the column math;
    the results are the columns that adds: renamed columns under their new
    names, and every formula whose inputs exist. Formulas are evaluated one
    shot at a time, so ``min`` / ``max`` act on that shot alone. Files
    edited while a watcher runs are re-read for the next shot.
    """

    def __init__(self, column_math=None, search_replace=None, parse_from='auto', name='column_math'):
        super().__init__(friendly_name=name)
        if column_math is None and search_replace is None:
            raise ValueError("Give column_math and / or search_replace.")
        self.column_math = column_math
        self.search_replace = search_replace
        self.parse_from = parse_from
        self.summary_name = name
        self._lock = threading.Lock()
        self._mtimes = None
        self.reload()

    def _files(self):
        from PyGEECSPlotter.scan_data_analyzers.sfile_frame import iter_control_sources
        return [label for spec in (self.search_replace, self.column_math)
                for label, parse_from, _ in iter_control_sources(spec, self.parse_from)
                if parse_from is not None]

    def _file_mtimes(self):
        return tuple(os.path.getmtime(f) if os.path.exists(f) else None for f in self._files())

    def reload(self):
        """(Re)read the controls; also records each file's SHA-256 (``control_files``)."""
        from PyGEECSPlotter.scan_data_analyzers.sfile_frame import load_controls
        from PyGEECSPlotter.utils.provenance import file_hash
        self.pairs = load_controls(self.search_replace, 'search_replace', self.parse_from)[0] \
            if self.search_replace is not None else []
        self.clusters = load_controls(self.column_math, 'column_math', self.parse_from)[0] \
            if self.column_math is not None else []
        self.control_files = {f: file_hash(f) for f in self._files()}
        self._mtimes = self._file_mtimes()
        reductions = [c['column_name'] for c in self.clusters
                      if any(f'{fn}(' in c['formula'] for fn in ('min', 'max'))]
        if reductions:
            warnings.warn(f"{self.summary_name}: {reductions} use min / max, which act on one "
                          f"shot at a time here (not the whole scan).", stacklevel=2)

    def _current(self):
        mtimes = self._file_mtimes()
        if mtimes != self._mtimes:
            with self._lock:
                if mtimes != self._mtimes:
                    self.reload()
        return self.pairs, self.clusters

    def analyze_data(self, data, bg=None, context=None):
        from PyGEECSPlotter.scan_data_analyzers.sfile_frame import (apply_column_math,
                                                                    apply_search_replace)
        pairs, clusters = self._current()
        row = pd.DataFrame([context or {}])
        out = apply_search_replace(row, pairs) if pairs else row
        names = [c for c in out.columns if c not in row.columns]
        out = apply_column_math(out, clusters)
        names += [c['column_name'] for c in clusters
                  if c['column_name'] in out.columns and c['column_name'] not in names
                  and all(v['column_name'] in out.columns for v in c['variables'])]
        results = {}
        for name in names:
            value = out[name].iloc[0]
            results[name] = value.item() if isinstance(value, np.generic) else value
        return None, results, {}
