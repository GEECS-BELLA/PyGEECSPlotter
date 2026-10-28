# PyGEECS base functions
# Author: Alex Picksley
# Version 0.4
# Created: 2023-02-26
# Last Modified: 2025-07-30

import numpy as np
import os, sys
import matplotlib.pyplot as plt
import re
import glob
import json
import datetime
import inspect
import concurrent.futures
import warnings
import pandas as pd
from contextlib import contextmanager
from pathlib import Path
from tqdm import tqdm

from PyGEECSPlotter.utils.navigation_utils import (
    alias_column_names, get_analysis_dir, get_parameter_alias, get_scan_parameter,
    get_sfilename_from_top_dir,
    get_top_dir, get_top_dir_from_sfilename, open_directory_in_explorer,
)
from PyGEECSPlotter.utils.io_utils import write_controls_from_python, file_lock
from PyGEECSPlotter.scan_data_analyzers.binning import compute_bin_numbers
from PyGEECSPlotter.scan_data_analyzers.filters import Filter, combined_mask, summary
from PyGEECSPlotter.scan_data_analyzers.sfile_frame import (
    apply_column_math, apply_search_replace, drop_incomplete_rows, merge_results,
    parse_column_math_from_lv_controls, parse_search_replace_from_lv_controls, read_controls,
    rename_result_columns, search_replace_text)
from PyGEECSPlotter.analyzers.multi_diagnostic_analyzer import MultiDiagnosticAnalyzer

import PyGEECSPlotter.utils.plotting as gplt
colors = gplt.configure_plotting()



def _as_path_str(path):
    """``str`` for a str / Path / os.PathLike, None for None."""
    return None if path is None else os.fspath(path)


def _aux_kwarg(method, aux):
    """``{'aux': aux}`` if ``method`` accepts an ``aux`` argument, else ``{}``.

    ``analyze_scan`` hands each shot's ``aux`` to ``display_data`` and
    ``write_analyzed_data`` only when the analyzer asks for it, so analyzers
    written before this was added keep working unchanged. Passing ``aux``
    per call (rather than an analyzer caching the last shot's ``aux``) keeps
    display and write correct when shots are analysed in parallel.
    """
    try:
        params = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return {}
    return {'aux': aux} if 'aux' in params else {}


class FilterSummary(str):
    """``scan.filters``: a string that prints as a readable list in the REPL."""

    def __repr__(self):
        return str(self)


class ScanDataAnalyzer:
    """
    One GEECS scan: its sfile as a DataFrame, shot selection (filters),
    bins, per-shot analysis and scan-level displays.

    Construct from one of:
      1. ``sfilename``                                (``.../yy_mmdd/analysis/sN.txt``)
      2. ``top_dir`` + ``scan``                       (``top_dir`` is the day folder)
      3. ``experiment_dir`` + ``year``, ``month``, ``day``, ``scan``

    ``year``, ``month``, ``day`` and ``scan`` are always stored as ``int``
    (strings such as ``'05'`` are converted; invalid values raise). Paths
    may be ``str`` or ``pathlib.Path`` and are stored as ``str``.
    """

    def __init__(self,
                 sfilename=None,
                 top_dir=None,
                 experiment_dir=None,
                 year=None,
                 month=None,
                 day=None,
                 scan=None,
                 ):
        self._init_common()
        self.sfilename = _as_path_str(sfilename)
        self.top_dir = _as_path_str(top_dir)
        self.experiment_dir = _as_path_str(experiment_dir)
        self.year = year
        self.month = month
        self.day = day
        self.scan = scan

        if self.sfilename:
            self._init_from_sfilename()
        elif self.top_dir and self.scan is not None:
            self._init_from_top_dir_and_scan()
        elif self.experiment_dir and None not in (self.year, self.month, self.day, self.scan):
            self._init_from_experiment_path()
        else:
            raise ValueError("Invalid initialization arguments. Use one of:\n"
                             "1. sfilename\n"
                             "2. top_dir and scan\n"
                             "3. experiment_dir, year, month, day, scan")
        self.scan_title = self.scan_data_title()

    def _init_common(self):
        """State every scan object has, before any data is loaded (also used by subclasses)."""
        self.sfilename = None
        self.top_dir = None
        self.experiment_dir = None
        self._year = self._month = self._day = self._scan = None
        self.scan_parameter = 'Shotnumber'
        self.analysis_dir = None
        self.data_columns = None
        self.data = None
        self.aborted = False
        self.print_data = False
        self._filters = {}
        self._order = None           # (column, ascending, per_bin) set by order_by
        self.verbose = True          # print a line for each user filter call
        self.last_merged_columns = []

    def _init_from_sfilename(self):
        top_dir, year, month, day = get_top_dir_from_sfilename(self.sfilename)
        self.top_dir = top_dir
        self.year, self.month, self.day = year, month, day
        match = re.search(r's(\d+)\.txt$', os.path.basename(self.sfilename))
        if self.scan is None and match:
            self.scan = int(match.group(1))

    def _init_from_top_dir_and_scan(self):
        self.sfilename = get_sfilename_from_top_dir(self.top_dir, self.scan)
        top_dir, year, month, day = get_top_dir_from_sfilename(self.sfilename)
        self.top_dir = top_dir
        self.year, self.month, self.day = year, month, day

    def _init_from_experiment_path(self):
        self.top_dir = get_top_dir(self.experiment_dir, self.year, self.month, self.day)
        self.sfilename = get_sfilename_from_top_dir(self.top_dir, self.scan)

    # identity: always int (or None until known) -------------------------
    _RANGES = {'year': (2000, 2999), 'month': (1, 12), 'day': (1, 31), 'scan': (0, None)}

    @classmethod
    def _to_int(cls, value, name):
        if value is None:
            return None
        try:
            as_float = float(value)
            if as_float != int(as_float):
                raise ValueError
            out = int(as_float)
        except (TypeError, ValueError):
            raise TypeError(f"'{name}' must be a whole number, got {type(value).__name__} {value!r}")
        if name == 'year' and 0 <= out < 100:
            out += 2000                            # '26' -> 2026
        low, high = cls._RANGES.get(name, (None, None))
        if (low is not None and out < low) or (high is not None and out > high):
            raise ValueError(f"'{name}' = {out} is out of range "
                             f"[{low}, {'...' if high is None else high}]")
        return out

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, value):
        self._year = self._to_int(value, 'year')

    @property
    def month(self):
        return self._month

    @month.setter
    def month(self, value):
        self._month = self._to_int(value, 'month')

    @property
    def day(self):
        return self._day

    @day.setter
    def day(self, value):
        self._day = self._to_int(value, 'day')

    @property
    def scan(self):
        return self._scan

    @scan.setter
    def scan(self, value):
        self._scan = self._to_int(value, 'scan')

    @property
    def date(self):
        """The experiment date as ``datetime.date``, or None if not known."""
        if None in (self.year, self.month, self.day):
            return None
        return datetime.date(self.year, self.month, self.day)

    @property
    def scan_label(self):
        """``'Scan023'`` (or ``'Scan???'`` if the scan number isn't known)."""
        return f'Scan{self.scan:03d}' if self.scan is not None else 'Scan???'

    def __repr__(self):
        return (f"{type(self).__name__}(sfilename={self.sfilename}, scan={self.scan}, "
                f"date={self.date}, experiment_dir={self.experiment_dir})")

    # -------------------------------------------------------------------
    # Shot selection: named filter rules, re-evaluated against self.data
    # (see scan_data_analyzers/filters.py). Nothing here modifies self.data.
    # -------------------------------------------------------------------
    def _init_mask(self):
        """Clear every filter rule, including the missing-file rules."""
        self._filters = {}

    @property
    def _mask(self):
        """Boolean ndarray over self.data: the rows every filter keeps."""
        if self.data is None:
            return None
        return combined_mask(self._filters, self.data)

    @property
    def active_data(self):
        """
        The selected rows of ``self.data`` (every filter applied).

        The index is ``self.data``'s, so a row keeps its id across filters;
        use ``.iloc`` for positions within the selection. Rows are in sfile
        order unless ``order_by`` has set an order.
        """
        active = self.data[self._mask]
        if self._order is not None:
            active = self._sort(active, *self._order)
        return active

    @property
    def n_active(self):
        """Number of selected shots."""
        return int(self._mask.sum())

    @property
    def filters(self):
        """Printable summary of the filters: each rule and how many shots remain."""
        return FilterSummary(summary(self._filters, self.data))

    def _add_filter(self, rule, quiet=False):
        self._filters[rule.name] = rule
        if self.verbose and not quiet:
            print(f'{self.n_active} / {len(self.data)} shots after {rule.describe()}')

    def filter(self, column, low=None, high=None, exclusive=False, name=None):
        """
        Keep shots with ``low <= column <= high`` (bounds inclusive; either
        may be None for a one-sided cut). NaNs are dropped. ``exclusive=True``
        keeps the shots outside the range instead.

        Rules combine with AND. ``name`` (default: the column) identifies the
        rule for ``remove_filter``; adding a rule with an existing name
        replaces it.
        """
        self._check_column(column)
        self._add_filter(Filter(name or column, 'range', column=column, low=low, high=high,
                                exclusive=exclusive))

    def filter_values(self, column, values, exclusive=False, name=None):
        """Keep shots whose ``column`` value is in ``values`` (or not, with ``exclusive=True``)."""
        self._check_column(column)
        self._add_filter(Filter(name or f'{column} values', 'values', column=column,
                                values=np.asarray(values), exclusive=exclusive))

    def filter_rows(self, rule, name):
        """
        Keep shots where ``rule`` is True. ``rule`` is either a function
        ``rule(data) -> bool Series / array`` (re-evaluated whenever the data
        changes) or a fixed boolean array over ``self.data``.
        """
        if callable(rule):
            func = rule
        else:
            fixed = pd.Series(np.asarray(rule, dtype=bool), index=self.data.index)
            func = lambda data: fixed.reindex(data.index, fill_value=False).to_numpy()  # noqa: E731
        self._add_filter(Filter(name, 'rows', func=func))

    def remove_filter(self, name):
        """Remove one filter rule by name."""
        if name not in self._filters:
            raise KeyError(f"No filter named {name!r}. Filters: {list(self._filters)}")
        del self._filters[name]
        if self.verbose:
            print(f'{self.n_active} / {len(self.data)} shots after removing {name!r}')

    def clear_filters(self, keep_file_filters=True):
        """Remove every filter (by default keeping the missing-file rules)."""
        self._filters = {k: f for k, f in self._filters.items() if keep_file_filters and f.file_rule}
        if self.verbose:
            print(f'Filters cleared. {self.n_active} / {len(self.data)} shots included.')

    @contextmanager
    def temporary_filter(self, column, low=None, high=None, exclusive=False):
        """
        ``with scan.temporary_filter(col, lo, hi): ...`` applies a range
        filter inside the block and restores the previous filters on exit,
        even on error.
        """
        saved = dict(self._filters)
        try:
            self._check_column(column)
            self._add_filter(Filter(f'temporary {column}', 'range', column=column, low=low,
                                    high=high, exclusive=exclusive), quiet=True)
            yield self
        finally:
            self._filters = saved

    @contextmanager
    def filtered(self):
        """
        ``with scan.filtered(): scan.filter(...); ...`` restores the filters
        as they were on entry when the block exits, whatever was added or
        removed inside.
        """
        saved = dict(self._filters)
        try:
            yield self
        finally:
            self._filters = saved

    def save_mask(self):
        """Snapshot of the current filters, for ``restore_mask`` (prefer ``filtered()``)."""
        return dict(self._filters)

    def restore_mask(self, saved):
        """Restore filters saved by ``save_mask``."""
        self._filters = dict(saved)

    def _unique_name(self, base):
        """``base``, or ``base #2``, ``#3``... if a filter already has that name."""
        name, i = base, 1
        while name in self._filters:
            i += 1
            name = f'{base} #{i}'
        return name

    def _check_column(self, column):
        if column not in self.data.columns:
            raise KeyError(f"Column {column!r} not in the scan data.")

    # bins ---------------------------------------------------------------
    def bins(self):
        """Sorted bin numbers (``temp Bin number``, as int) present among the selected shots."""
        values = self.active_data['temp Bin number'].to_numpy()
        values = values[np.isfinite(values)] if values.dtype.kind == 'f' else values
        return np.unique(values).astype(int)

    def rows_in_bin(self, b):
        """Selected shots in bin ``b`` (pass as ``rows=`` to ``_iter_shots`` etc.)."""
        active = self.active_data
        return active[active['temp Bin number'] == b]

    def iter_bins(self, bins=None):
        """Yield ``(bin, rows)`` for each bin (default: every bin among the selected shots)."""
        for b in (self.bins() if bins is None else bins):
            yield b, self.rows_in_bin(b)

    # ordering -----------------------------------------------------------
    @staticmethod
    def _sort(rows, column, ascending=True, per_bin=False):
        """``rows`` sorted by ``column`` (NaNs last), grouped by bin first if ``per_bin``."""
        if column not in rows.columns:
            raise KeyError(f"Column {column!r} not in the scan data.")
        by = ['temp Bin number', column] if per_bin else [column]
        asc = [True, ascending] if per_bin else [ascending]
        return rows.sort_values(by, ascending=asc, na_position='last', kind='stable')

    def sorted_rows(self, column, ascending=True, per_bin=False):
        """
        Selected shots sorted by ``column`` (NaNs last). ``per_bin=True``
        sorts within each bin, bins in order. Doesn't change the scan.
        """
        return self._sort(self.data[self._mask], column, ascending, per_bin)

    def top(self, column, n=1, ascending=False, per_bin=False):
        """
        The ``n`` selected shots with the largest ``column`` values
        (``ascending=True``: the smallest), or the top ``n`` of each bin with
        ``per_bin=True``. Shots where ``column`` is NaN are never picked.
        Pass the result as ``rows=``, or index ``.iloc`` / ``['Shotnumber']``.
        """
        rows = self.sorted_rows(column, ascending=ascending, per_bin=per_bin)
        rows = rows[rows[column].notna()]
        if per_bin:
            return rows.groupby('temp Bin number', sort=True).head(n)
        return rows.head(n)

    def order_by(self, column=None, ascending=True, per_bin=False):
        """
        Set the order of ``active_data`` (and so of every displayer that
        walks the shots in order, e.g. waterfalls and ``SampledImages``)
        until cleared with ``order_by(None)``. Most of the time ``top`` /
        ``sorted_rows`` are what you want; this is for when every view
        should follow one order.
        """
        if column is None:
            self._order = None
            if self.verbose:
                print('Shots back in sfile order.')
            return
        self._check_column(column)
        self._order = (column, ascending, per_bin)
        if self.verbose:
            print(f"Shots ordered by {column} ({'ascending' if ascending else 'descending'}"
                  f"{', within each bin' if per_bin else ''}).")

    # quick look-ups ----------------------------------------------------
    def find_columns(self, *terms, regex=False, show=True):
        """
        Column names containing every one of ``terms`` (case-insensitive;
        ``regex=True`` treats them as regular expressions). Prints them
        with a count of non-NaN values among the selected shots, and
        returns the list.
        """
        flags = re.IGNORECASE
        cols = [c for c in self.data.columns
                if all((re.search(t, c, flags) if regex else t.lower() in c.lower()) for t in terms)]
        if show:
            active = self.data[self._mask]
            width = max((len(c) for c in cols), default=0)
            for c in cols:
                print(f'  {c:<{width}}  ({int(active[c].notna().sum())} / {len(active)} values)')
            if not cols:
                print(f'  no columns contain {" and ".join(map(repr, terms))}')
        return cols

    def describe_columns(self, columns):
        """
        Count, NaN count, mean, std, min, median and max of numeric
        ``columns`` (a list, or a search term as for ``find_columns``) over
        the selected shots, as a DataFrame.
        """
        if isinstance(columns, str):
            columns = self.find_columns(columns, show=False)
        active = self.data[self._mask]
        numeric = [c for c in columns if pd.api.types.is_numeric_dtype(active[c])]
        desc = active[numeric].agg(['count', 'mean', 'std', 'min', 'median', 'max']).T
        desc.insert(1, 'nan', len(active) - desc['count'])
        return desc

    def shot(self, shotnumber):
        """One shot's row (from all the data, filtered or not) as a dict."""
        match = self.data[self.data['Shotnumber'] == int(shotnumber)]
        if match.empty:
            raise KeyError(f"No shot {shotnumber} in {self.scan_label}.")
        return match.iloc[0].to_dict()

    def shot_files(self, shotnumber):
        """``{diagnostic: path}`` of one shot's registered diagnostic files."""
        row = self.shot(shotnumber)
        return {k[:-len(' file_list')]: v for k, v in row.items()
                if k.endswith(' file_list') and isinstance(v, str) and v}

    def summary(self, show=True):
        """Date, scan, scan parameter, shots, bins, filters and diagnostics, as text."""
        lines = [f'{self.scan_data_title()}  ({self.scan_label})']
        if self.data is None:
            lines.append('  no data loaded (load_scan_data)')
        else:
            active = self.data[self._mask]
            lines.append(f'  shots      : {len(active)} selected of {len(self.data)}')
            param = self.scan_parameter
            if param in active.columns and len(active) and pd.api.types.is_numeric_dtype(active[param]):
                lines.append(f'  parameter  : {param}  ({active[param].min():g} to {active[param].max():g})')
            else:
                lines.append(f'  parameter  : {param}')
            if 'temp Bin number' in active.columns:
                lines.append(f'  bins       : {len(self.bins())}')
            diags = [c[:-len(' file_list')] for c in self.data.columns if c.endswith(' file_list')]
            lines.append(f"  diagnostics: {', '.join(diags) if diags else 'none registered'}")
            if self._order is not None:
                col, asc, per_bin = self._order
                lines.append(f"  ordered by : {col} ({'ascending' if asc else 'descending'}"
                             f"{', per bin' if per_bin else ''})")
            lines.append('  filters    :')
            lines += ['    ' + line for line in summary(self._filters, self.data).splitlines()]
        text = '\n'.join(lines)
        if show:
            print(text)
            return None
        return text

    def filter_outliers(self, column, n_sigma=3.0, per_bin=True, name=None):
        """
        Drop shots whose ``column`` is more than ``n_sigma`` robust standard
        deviations (1.4826 x median absolute deviation) from the median of
        their bin (``per_bin=True``) or of the whole scan. A named rule
        (default ``'outliers: <column>'``), computed over all shots, so it
        doesn't depend on the other filters. NaN values are dropped.
        """
        self._check_column(column)

        def within_n_sigma(data):
            values = data[column]
            groups = data['temp Bin number'] if per_bin else pd.Series(0, index=data.index)
            median = values.groupby(groups).transform('median')
            deviation = (values - median).abs()
            mad = deviation.groupby(groups).transform('median') * 1.4826
            return (deviation <= n_sigma * mad).to_numpy()

        within_n_sigma.__name__ = (f'{column} within {n_sigma:g} sigma of its '
                                   f"{'bin' if per_bin else 'scan'} median")
        self._add_filter(Filter(name or f'outliers: {column}', 'rows', func=within_n_sigma))

    def scan_data_title(self, append_text=None):
        """``'2024-01-31 - s23.txt'`` plus `` - append_text`` if given."""
        parts = [str(self.date) if self.date else None,
                 os.path.basename(self.sfilename) if self.sfilename else self.scan_label,
                 append_text]
        return ' - '.join(p for p in parts if p)

    def load_scan_data(self, 
        search_replace_filename=None, 
        column_math_filename=None, 
        analyzer=None,
        parse_from='python', 
        remove_missing_diagnostic_files=True,
        ):

        sfile_data = pd.read_csv(self.sfilename, sep='\t')
        sfile_data = self._drop_incomplete_rows(sfile_data)
        if len(sfile_data) == 0:
            self.data = sfile_data
            self.scan_parameter = 'Shotnumber'
            self.aborted = True

        else:
            scan_parameter, scan = get_scan_parameter(self.top_dir, sfile_data)
            if self.scan is not None and int(scan) not in (0, self.scan):
                warnings.warn(f"{os.path.basename(self.sfilename)} records scan {int(scan)}, "
                              f"not {self.scan}; using {int(scan)}.", stacklevel=2)
            self.scan = scan
            self.scan_parameter = scan_parameter

            self.data = sfile_data.copy()
            self._init_mask()
            names = alias_column_names(self.data.columns)
            shared = sorted({get_parameter_alias(c) for c, n in names.items()
                             if n != get_parameter_alias(c)})
            if shared:
                warnings.warn(f"{os.path.basename(self.sfilename)}: aliases used by more than "
                              f"one column ({shared}); those columns keep their device names.",
                              stacklevel=2)
            self.data = self.data.rename(columns=names)

            self.data['temp Bin number'] = self.data['Bin #']

            if search_replace_filename is not None:
                self.add_search_replace_column_names(search_replace_filename, parse_from=parse_from)
                self.add_search_replace_scan_parameter(search_replace_filename, parse_from=parse_from)
            if column_math_filename is not None:
                self.add_column_math(column_math_filename, parse_from=parse_from)
            
            if analyzer is not None:
                analyzer.register_with_scan(self, remove_missing_diagnostic_files)
            self.set_analysis_dir()

    @staticmethod
    def _drop_incomplete_rows(sfile_data):
        """Rows without scan / Shotnumber removed (see ``sfile_frame.drop_incomplete_rows``)."""
        return drop_incomplete_rows(sfile_data)

    def get_data_columns(self):
        if self.data is not None:
            return self.data.columns.tolist()
        else:
            return None

    def set_analysis_dir(self, scan=None, make_dir=False):
        if scan is None:
            scan = self.scan
        self.analysis_dir = get_analysis_dir(self.top_dir, scan, make_dir=make_dir)

    def get_scan_info_text(self, print_data=False):
        with open(os.path.join(self.top_dir, 'scans', 'Scan%03d' %self.scan, 'ScanInfoScan%03d.ini' %self.scan)) as f:
            lines = f.readlines()
        scan_info_text = lines[2].split('"')[1]
        
        if print_data:
            print('Scan Information  : %s' %scan_info_text)
        return scan_info_text

        
    def add_search_replace_column_names(self, search_replace_filename, parse_from='labview'):
        """Rename columns with a search / replace controls file (``sfile_frame.apply_search_replace``)."""
        pairs = read_controls(search_replace_filename, 'search_replace', parse_from)
        self.data = apply_search_replace(self.data, pairs)

    def add_search_replace_scan_parameter(self, search_replace_filename, parse_from='labview'):
        """Apply the same search / replace to the scan parameter's name."""
        pairs = read_controls(search_replace_filename, 'search_replace', parse_from)
        self.scan_parameter = search_replace_text(self.scan_parameter, pairs)

    def add_column_math(self, column_math_filename, parse_from='labview'):
        """Add computed columns from a column-math controls file (``sfile_frame.apply_column_math``)."""
        clusters = read_controls(column_math_filename, 'column_math', parse_from)
        self.data = apply_column_math(self.data, clusters, verbose=self.print_data)

    def merge_column_math_to_sfile(self, column_math_filename, parse_from='labview'):
        if not self.aborted:
            math_clusters_list = read_controls(column_math_filename, 'column_math', parse_from)
            cols_to_add = ['scan', 'Shotnumber']
            for math_cluster in math_clusters_list:
                cols_to_add.append( math_cluster['column_name'] )
            valid_columns = [col for col in cols_to_add if col in self.data.columns]
            add_columns_df = self.data[valid_columns]
            
            self.merge_data_frame_to_sfile(add_columns_df, 
                                    diagnostic=None,
                                    overwrite_columns=True, 
                                    analysis_label=None, 
                                    )
            return add_columns_df
        else:
            return None
                    

    def add_file_list_to_scan_data(self, diagnostic, file_ext, remove_missing_files=True):
        if diagnostic is None or file_ext is None:
            print('No diagnostic selected')
            return

        # Determine diagnostic directory and type
        scan_diag_dir = os.path.join(self.top_dir, 'scans', 'Scan%03d' % self.scan, diagnostic)
        analysis_diag_dir = os.path.join(self.top_dir, 'analysis', 'Scan%03d' % self.scan, diagnostic)

        if os.path.exists(scan_diag_dir):
            analysis_diagnostic = False
        elif os.path.exists(analysis_diag_dir):
            analysis_diagnostic = True
        else:
            print(f'No {diagnostic} directory found for Scan{self.scan:03d}')
            self.data['%s file_list' % diagnostic] = ''
            self.data['%s file_exists' % diagnostic] = 0
            if remove_missing_files:
                self._add_file_filter(diagnostic)
            return

        # Build all filenames at once using vectorised string formatting
        base_dir = 'analysis' if analysis_diagnostic else 'scans'

        self.data['%s file_list' % diagnostic] = (
            self.data.apply(
                lambda row: os.path.join(
                    self.top_dir, base_dir,
                    'Scan%03d' % row['scan'],
                    diagnostic,
                    'Scan%03d_%s_%03d%s' % (row['scan'], diagnostic, row['Shotnumber'], file_ext)
                ),
                axis=1
            )
        )

        # Collect all unique directories that appear in the file list,
        # scan each once with os.scandir, and build a set of known existing files.
        unique_dirs = self.data['%s file_list' % diagnostic].apply(os.path.dirname).unique()

        existing_files = set()
        for d in unique_dirs:
            if os.path.exists(d):
                existing_files.update(entry.path for entry in os.scandir(d))

        self.data['%s file_exists' % diagnostic] = (
            self.data['%s file_list' % diagnostic].isin(existing_files).astype(int)
        )

        if self.print_data:
            n_missing = (self.data['%s file_exists' % diagnostic] == 0).sum()
            missing_files = self.data.loc[
                self.data['%s file_exists' % diagnostic] == 0,
                '%s file_list' % diagnostic
            ]
            for f in missing_files:
                print('Did not find: %s' % os.path.basename(f))
            print('Files found: %d / %d' % (len(self.data) - n_missing, len(self.data)))

        if remove_missing_files:
            self._add_file_filter(diagnostic)

    def _add_file_filter(self, diagnostic):
        """Named rule dropping shots whose ``diagnostic`` file is missing."""
        column = '%s file_exists' % diagnostic
        self._filters[f'files: {diagnostic}'] = Filter(
            f'files: {diagnostic}', 'range', column=column, low=1, file_rule=True)
        if self.print_data:
            n_missing = int((self.data[column] == 0).sum())
            print('Masked out %d lines from scan_data for missing files' % n_missing)

    def filter_scan_data(self, filter_parameter, lower_bound, upper_bound,
                        filter_exclusive=False, print_data=False):
        """
        Legacy range filter: keeps ``lower_bound < column < upper_bound``
        (bounds **excluded**), or the rows outside with ``filter_exclusive``.
        Prefer ``filter`` (bounds included, one-sided cuts, named rules).
        As before, repeated calls stack (AND), even on the same column.
        """
        self._check_column(filter_parameter)
        lo, hi, col = lower_bound, upper_bound, filter_parameter

        def strict_range(data):
            values = data[col]
            if filter_exclusive:
                return ((values < lo) | (values > hi)).to_numpy()
            return ((values > lo) & (values < hi)).to_numpy()

        strict_range.__name__ = (f'NOT {lo:g} < {col} < {hi:g}' if filter_exclusive
                                 else f'{lo:g} < {col} < {hi:g}')
        self._add_filter(Filter(self._unique_name(col), 'rows', func=strict_range),
                         quiet=not print_data)

    def filter_scan_data_by_array(self, filter_parameter, values, filter_exclusive=False):
        """Legacy name for ``filter_values`` (repeated calls stack, as before)."""
        self.filter_values(filter_parameter, values, exclusive=filter_exclusive,
                           name=self._unique_name(f'{filter_parameter} values'))

    def reset_filters(self):
        """Legacy name for ``clear_filters()`` (missing-file rules are kept)."""
        self.clear_filters()

    def rebin(self, method='unique', scan_parameter=None, **kwargs):
        """
        Re-compute ``temp Bin number`` from a column of ``self.data``.

        Bins drive ``compute_bin_summary`` and ``aggregate_per_bin``. By
        default ``temp Bin number`` is initialized from the sfile's
        ``Bin #`` column at load time. Call ``rebin`` to redefine bins
        based on any column — typically when you want to bin by a
        measured/post-analyzed value rather than the DAQ's commanded
        bin index.

        The mask (filters) is unaffected.

        Parameters
        ----------
        method : str or callable, optional
            One of ``'unique'``, ``'rounding'``, ``'zscore'``,
            ``'kmeans'``, ``'edges'``, ``'quantile'``, ``'width'``, or
            a callable ``f(values, **kwargs) -> ndarray`` that returns
            bin numbers. Default ``'unique'``.
        scan_parameter : str, optional
            Column name in ``self.data`` to bin on. Defaults to
            ``self.scan_parameter``.
        **kwargs
            Forwarded to the binning method:
              - ``rounding_factor`` for ``method='rounding'``
              - ``z_threshold``     for ``method='zscore'``
              - ``n_bins``          for ``method='kmeans'`` /
                                    ``'quantile'`` / ``'width'``
              - ``bin_edges``       for ``method='edges'``
              - ``bin_width``       for ``method='width'`` (number or
                                    ``np.histogram_bin_edges`` keyword)

        Returns
        -------
        ndarray of int
            The new bin numbers (same length as ``self.data``).
        """
        if scan_parameter is None:
            scan_parameter = self.scan_parameter

        if scan_parameter not in self.data.columns:
            raise KeyError(
                f"Column {scan_parameter!r} not in self.data. "
                f"Use load_scan_data first, or analyze_scan to add a measured column."
            )

        values = self.data[scan_parameter].values
        bin_numbers = compute_bin_numbers(values, method=method, **kwargs)
        self.data['temp Bin number'] = bin_numbers

        n_bins = len(np.unique(bin_numbers[bin_numbers > 0]))
        n_unbinned = int((bin_numbers == 0).sum())
        method_name = method if isinstance(method, str) else getattr(method, '__name__', 'callable')
        print(
            "Re-binned by '%s' (method=%s): %d bins, %d unbinned (NaN) shots."
            % (scan_parameter, method_name, n_bins, n_unbinned)
        )
        return bin_numbers

    def reset_bins(self):
        """
        Reset ``temp Bin number`` to the sfile's original ``Bin #`` column.
        """
        if 'Bin #' not in self.data.columns:
            raise KeyError("'Bin #' column not found in self.data — cannot reset bins.")
        self.data['temp Bin number'] = self.data['Bin #']
        n_bins = self.data['temp Bin number'].nunique()
        print("Bins reset to sfile 'Bin #' (%d bins)." % n_bins)

    def get_bg_file_path(self, diagnostic, file_ext='.png', which_scan='last'):
        """
        Retrieves the background file path for a specific diagnostic from the analysis directory.

        This function searches for the background file based on the specified scan. It can retrieve
        the first, last, a specific scan's background file, or a file matching a given string.
        The file is expected to have the naming pattern '*<diagnostic>_averaged<file_ext>' within the 'analysis' directory.

        Parameters:
        - which_scan (str or int, optional): Specifies which scan to retrieve:
            - 'first': Retrieves the first matching file.
            - 'last': Retrieves the last matching file.
            - int: Retrieves the file for the specific scan number.
            - str: Searches for a file that includes the string in its name.
            Defaults to 'last'.

        Returns:
        - bg_file_path (str or int): The path to the background file, or 0 if no file is found or an error occurs.
        """
        try:
            # Retrieve all matching files
            bg_file_paths = glob.glob(os.path.join(self.top_dir, 'analysis', '*%s_averaged%s' % (diagnostic, file_ext)))
            
            if not bg_file_paths:
                print(f"No background files found for diagnostic '{diagnostic}'.")
                return None

            if which_scan == 'first':
                bg_file_path = bg_file_paths[0]
            elif which_scan == 'last':
                bg_file_path = bg_file_paths[-1]
            elif isinstance(which_scan, int):
                scan_paths = glob.glob(os.path.join(self.top_dir, 'analysis', '*Scan%03d%s_averaged%s' % (which_scan, diagnostic, file_ext)))       
                if len(scan_paths) == 1:
                    bg_file_path = scan_paths[-1]
                else:
                    print(f"No background file found for Scan {which_scan} with diagnostic '{diagnostic}'.")
                    return None
            elif isinstance(which_scan, str):
                # Find the file that contains the specified string
                matching_files = [path for path in bg_file_paths if which_scan in os.path.basename(path)]
                if matching_files:
                    bg_file_path = matching_files[0]  # You could return the first match or handle multiple matches differently
                else:
                    print(f"No background file found matching '{which_scan}' for diagnostic '{diagnostic}'.")
                    return None
            else:
                print("Invalid value for 'which_scan'. Must be 'first', 'last', an integer, or a string.")
                return None

            if self.print_data:
                print(f"Bg File Path             : {bg_file_path}")
            
            return bg_file_path
        
        except IndexError:
            print(f"No background file found for diagnostic '{diagnostic}'.")
            return None
        except Exception as e:
            print(f"An error occurred while retrieving the background file path: {e}")
            return None

    def _process_row(self, analyzer, bg, row):
        """
        Run one shot's pipeline: build its context, load its diagnostic
        file(s), resolve its background, and call ``analyzer.analyze_data``.

        This is the per-row body shared by ``_iter_shots`` (serial) and
        ``_iter_shots_parallel`` (threaded) —
        kept in exactly one place so the multi-vs-single-diagnostic
        branching and background resolution can't drift between the two
        paths.

        Parameters
        ----------
        analyzer : DiagnosticAnalyzer
        bg : optional
            Background spec passed through ``_resolve_bg_for_row`` /
            ``_resolve_bg_for_multi``.
        row : Series
            One row of ``active_data`` (as yielded by ``DataFrame.iterrows()``).

        Returns
        -------
        context, data, results, aux : as documented on ``_iter_shots``.
        """
        context = row.to_dict()
        is_multi = isinstance(analyzer, MultiDiagnosticAnalyzer)

        if is_multi:
            paths = {name: context.get(f'{name} file_list') for name, _ in analyzer.inputs}
            data = analyzer.load_data(paths)
            bg_i = self._resolve_bg_for_multi(analyzer, bg, context)
        elif analyzer.diagnostic is not None:
            filename = context.get(f'{analyzer.diagnostic} file_list')
            data = analyzer.load_data(filename)
            bg_i = self._resolve_bg_for_row(analyzer, bg, context)
        else:
            data = None
            bg_i = self._resolve_bg_for_row(analyzer, bg, context)

        data, results, aux = analyzer.analyze_data(data, bg=bg_i, context=context)

        return context, data, results, aux

    def _process_row_and_write(self, analyzer, bg, row, write_analyzed=False, write_lineouts=False):
        """
        ``_process_row``, then (if ``write_analyzed``) the shot's analysed
        files, in the calling thread. Used by the worker threads of
        ``analyze_scan`` and of ``LiveShotWatcher``.
        """
        context, data, results, aux = self._process_row(analyzer, bg, row)
        self._write_row(analyzer, context, data, aux, write_analyzed, write_lineouts)
        return context, data, results, aux

    def _write_row(self, analyzer, context, data, aux, write_analyzed=False, write_lineouts=False):
        """The shot's analysed files (if ``write_analyzed``), from ``_process_row``'s
        output. Separate so a live watcher can show a shot before writing it."""
        if write_analyzed and data is not None:
            # Each shot's own scan dir (an sfile can span several scans).
            scan, shot_num = int(context['scan']), int(context['Shotnumber'])
            shot_dir = self._shot_analysis_dir(scan)
            analyzer.write_analyzed_data(data, shot_dir, scan, shot_num, context=context,
                                         **_aux_kwarg(analyzer.write_analyzed_data, aux))
            if write_lineouts:
                analyzer.write_analyzed_lineouts(aux, shot_dir, scan, shot_num)

    def _selected_rows(self, rows=None):
        """
        ``rows`` restricted to the selected shots (``active_data`` if None).
        Rows the filters exclude are dropped with a warning, so a stale
        subset can't bring back shots that have been filtered out.
        """
        if rows is None:
            return self.active_data
        keep = rows.index.isin(self.active_data.index)
        if not keep.all():
            warnings.warn(f"{int((~keep).sum())} of {len(rows)} requested rows are excluded by "
                          f"the current filters and were skipped.", stacklevel=3)
            rows = rows[keep]
        return rows

    def _iter_shots(self, analyzer, bg=None, show_progress=True, rows=None):
        """
        Yield ``(context, data, results, aux)`` for each shot.

        Centralizes the per-shot iteration shell: iterate the rows
        (``active_data`` by default) and run ``_process_row`` on each.

        Used by ``analyze_scan`` and any per-shot aggregation
        (``mean_std_diagnostic``, ``aggregate_per_bin``, custom workflows).

        Parameters
        ----------
        analyzer : DiagnosticAnalyzer
        bg : optional
            Background spec passed through ``_resolve_bg_for_row``.
        show_progress : bool, optional
            Wrap iteration in a tqdm progress bar.
        rows : DataFrame, optional
            Subset of rows to iterate, typically a slice / sample of
            ``active_data`` or ``rows_in_bin(b)``. Defaults to the full
            ``active_data``; rows outside the current selection are skipped.

        Yields
        ------
        context : dict
            Row as a dict (includes ``scan``, ``Shotnumber``, etc.).
        data : Any
            Analyzed shot data, or ``None`` if no diagnostic / missing file.
        results : dict
            Scalar outputs.
        aux : dict
            Auxiliary outputs (lineouts, etc.).
        """
        rows = self._selected_rows(rows)
        it = rows.iterrows()
        if show_progress:
            it = tqdm(it, total=rows.shape[0])

        for _, row in it:
            yield self._process_row(analyzer, bg, row)

    def _iter_shots_parallel(self, analyzer, bg=None, rows=None, max_workers=None,
                             show_progress=True, process=None):
        """
        Thread-parallel counterpart of ``_iter_shots``.

        Submits ``self._process_row(analyzer, bg, row)`` for every row up
        front to a ``ThreadPoolExecutor``, then yields the results in the
        same order ``_iter_shots`` would (i.e. ``rows``' original order —
        the row order of ``active_data``/``rows``, *not* completion order),
        so downstream output (``add_columns_df`` row order, per-shot
        display/write side effects) is identical to the serial path.

        Parameters, yields — same as ``_iter_shots``, plus:

        max_workers : int, optional
            Forwarded to ``ThreadPoolExecutor``.
        process : callable, optional
            ``process(row) -> (context, data, results, aux)`` run in the
            worker thread instead of ``_process_row`` (e.g. to also write
            the shot's analysed files there).
        """
        if process is None:
            def process(row):
                return self._process_row(analyzer, bg, row)
        rows = self._selected_rows(rows)
        # list(...) freezes row order up front; a DataFrame's row order can't
        # change under us mid-analysis since nothing here mutates self.data.
        row_list = [row for _, row in rows.iterrows()]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process, row) for row in row_list]

            # Yield in submission order as each result becomes ready, so the
            # caller's main-thread display/write work overlaps with the
            # workers still analysing later shots. The bar tracks shots
            # consumed by the caller (i.e. including its side effects).
            # .result() re-raises any exception from that shot's thread.
            progress = tqdm(total=len(futures)) if show_progress else None
            try:
                for future in futures:
                    result = future.result()
                    yield result
                    if progress is not None:
                        progress.update(1)
            finally:
                if progress is not None:
                    progress.close()

    def analyze_scan(self, analyzer,
        bg=None,
        display_data=False,
        write_columns_to_sfile=False,
        overwrite_columns=True,
        analysis_label='',
        extra_info_str='',
        write_analyzed=False,
        write_lineouts=False,
        write_displayed=False,
        close_displayed=True,
        keep_unanalysed=True,
        parallel=True,
        max_workers=None,
        column_prefix=None,
        ):
        """
        Run the per-shot pipeline over active shots and handle side effects.

        Iteration is delegated to ``_iter_shots``. This method applies the
        optional per-shot side effects (display, write_analyzed,
        write_lineouts, write_displayed), accumulates scalar results, merges
        them into ``self.data`` so they're immediately plottable, and
        optionally writes them back to the sfile (plus a standalone summary
        file, so the columns can be re-merged later without rerunning the
        analysis).

        Parameters
        ----------
        analyzer : DiagnosticAnalyzer
            Provides ``load_data``, ``analyze_data``, and optionally
            ``display_data`` / ``write_analyzed_data`` /
            ``write_analyzed_lineouts`` / ``write_displayed_data``.
            If ``display_data`` or ``write_analyzed_data`` accepts an ``aux``
            argument, it receives that shot's ``aux``.
        bg : optional
            Background spec; see ``_resolve_bg_for_row``.
        write_columns_to_sfile : bool, optional
            If True, overwrite ``self.sfilename`` with the new columns merged
            in, and write a standalone summary file
            (``Scan{scan:03d}_{diagnostic}_{extra_info_str}Summary.txt``)
            under the scan's analysis dir. The columns are merged into
            ``self.data`` either way, so this only controls whether anything
            on disk is touched.
        extra_info_str : str, optional
            Extra tag inserted into the summary filename (and appended to the
            merged sfile column names, same as ``analysis_label``).
        parallel : bool, optional
            True (default): load and analyse shots in a thread pool (and
            write their analysed files there); display / write_displayed and
            the result merge still run one shot at a time on the main thread,
            in row order, so the output is the same as serial. False: one
            shot at a time. Analyzers with ``thread_safe = False`` (e.g.
            ``WavefrontAnalyzer``) need ``parallel=False``.
        max_workers : int, optional
            Thread-pool size (default: Python's, ``min(32, cpu_count + 4)``).
        keep_unanalysed : bool, optional
            True (default): re-running an analysis on a subset of shots only
            replaces those shots' values; the other shots keep theirs, in
            memory and in the sfile. False: each result column is replaced
            whole, so shots not analysed this time become NaN.
        column_prefix : str, optional
            Prefix for the result column names (``'<prefix> <result>'``),
            in memory and in the sfile. Default: the analyzer's
            ``output_diagnostic``, else its ``diagnostic``.

        Returns
        -------
        add_columns_df : DataFrame or None
            The raw (unrenamed) per-shot results, one row per shot. May
            contain non-scalar columns (e.g. an image analyzer's
            ``imshow_extent`` array) that are dropped before merging into
            ``self.data`` — see ``last_merged_columns`` for the columns that
            actually landed on ``active_data``.
        """
        self.last_merged_columns = []
        rows = []
        analysis_dir = None
        self._warn_if_write_displayed_is_a_noop(display_data, write_displayed)

        if parallel:
            if not getattr(analyzer, 'thread_safe', True):
                raise ValueError(
                    f"{type(analyzer).__name__} is not thread-safe (thread_safe = False); "
                    f"use analyze_scan(..., parallel=False).")
            if display_data:
                warnings.warn(
                    "display_data figures are drawn one shot at a time on the main thread "
                    "(pyplot is not thread-safe), so they dominate the runtime; "
                    "parallel=False gives the same result.", stacklevel=2)

            def process(row):
                return self._process_row_and_write(analyzer, bg, row, write_analyzed,
                                                   write_lineouts)

            shots = self._iter_shots_parallel(analyzer, bg=bg, max_workers=max_workers,
                                              process=process)
        else:
            shots = self._iter_shots(analyzer, bg=bg)

        for context, data, results, aux in shots:
            rows.append({'scan': int(context['scan']), 'Shotnumber': int(context['Shotnumber']), **results})
            if parallel and write_analyzed and analysis_dir is None and data is not None:
                # written in the worker; still record the first shot's dir
                # as the default for the scan-level files
                analysis_dir = self._shot_analysis_dir(int(context['scan']))
            analysis_dir = self._apply_shot_side_effects(
                analyzer, context, data, results, aux, analysis_dir,
                display_data=display_data, write_analyzed=write_analyzed and not parallel,
                write_lineouts=write_lineouts and not parallel, write_displayed=write_displayed,
                close_displayed=close_displayed,
            )

        return self._finish_analyze_scan(
            rows, analyzer, analysis_dir, write_columns_to_sfile=write_columns_to_sfile,
            analysis_label=analysis_label, extra_info_str=extra_info_str,
            overwrite_columns=overwrite_columns, keep_unanalysed=keep_unanalysed,
            column_prefix=column_prefix,
        )

    @staticmethod
    def _warn_if_write_displayed_is_a_noop(display_data, write_displayed):
        if write_displayed and not display_data:
            warnings.warn(
                "analyze_scan(write_displayed=True) has no effect without display_data=True: "
                "the figure written is the one display_data draws. Pass display_data=True "
                "to save per-shot figures.", stacklevel=3)

    def _apply_shot_side_effects(self, analyzer, context, data, results, aux, analysis_dir,
                                 display_data, write_analyzed, write_lineouts,
                                 write_displayed, close_displayed):
        """
        Run one shot's display/write side effects, exactly as ``analyze_scan``
        did inline before this was extracted so it could also be called, in
        the same order, from the parallel path of ``analyze_scan``.

        Returns the (possibly newly-resolved) ``analysis_dir``, so the caller
        can thread it through to the next shot and to ``_finish_analyze_scan``
        — it is the analysis dir of the first shot written, used as the
        default for scan-level files. Per-shot files always go to the dir of
        that shot's own ``context['scan']``.
        """
        if data is None:
            return analysis_dir

        scan, shot_num = int(context['scan']), int(context['Shotnumber'])

        fig = None
        if display_data:
            filename = context.get(f'{analyzer.diagnostic} file_list', '')
            fig, _ = analyzer.display_data(
                data, return_dict=results, title=os.path.basename(filename),
                **_aux_kwarg(analyzer.display_data, aux)
            )

        # An sfile can span several scans: each shot's files go to its own
        # scan's analysis dir; the first one resolved is the default returned
        # for the scan-level summary/controls files.
        if write_analyzed or (display_data and write_displayed):
            shot_dir = self._shot_analysis_dir(scan)
            if analysis_dir is None:
                analysis_dir = shot_dir

        if write_analyzed:
            analyzer.write_analyzed_data(data, shot_dir, scan, shot_num, context=context,
                                         **_aux_kwarg(analyzer.write_analyzed_data, aux))
            if write_lineouts:
                analyzer.write_analyzed_lineouts(aux, shot_dir, scan, shot_num)
        if display_data and write_displayed:
            analyzer.write_displayed_data(fig, shot_dir, scan, shot_num)

        if close_displayed and fig is not None:
            plt.close(fig)

        return analysis_dir

    def _finish_analyze_scan(self, rows, analyzer, analysis_dir, write_columns_to_sfile,
                             analysis_label, extra_info_str, overwrite_columns,
                             keep_unanalysed=True, column_prefix=None):
        """
        Post-loop tail of ``analyze_scan``: build ``add_columns_df``, merge
        the renamed columns into ``self.data``, and optionally write them
        back to the sfile plus the standalone controls/summary files.

        Shared by the serial ``analyze_scan`` and
        parallel ``analyze_scan`` paths so this logic exists in
        exactly one place. ``analysis_dir`` is whatever the shot loop already
        resolved (``None`` if neither ``write_analyzed`` nor
        ``write_displayed`` needed it); this method resolves it itself,
        lazily, if still ``None`` and it's needed for the controls/summary
        files — the same lazy-once behaviour ``analyze_scan`` always had.
        """
        add_columns_df = pd.DataFrame(rows) if rows else None

        if add_columns_df is not None and len(self.data) > 0:
            diag_str = column_prefix if column_prefix is not None else (
                analyzer.output_diagnostic if analyzer.output_diagnostic is not None
                else analyzer.diagnostic)

            # Merge into self.data unconditionally, so the new columns are on
            # active_data and plottable immediately — whether or not anything
            # is written to disk.
            renamed_df = self._rename_add_columns(add_columns_df, diag_str, extra_info_str)
            self.last_merged_columns = [
                c for c in renamed_df.columns if c not in ('scan', 'Shotnumber')
            ]
            self.data = merge_results(self.data, renamed_df, keep_unanalysed=keep_unanalysed)

            if write_columns_to_sfile:
                if analysis_dir is None:
                    analysis_dir = self.get_scan_data_analysis_dir(make_dir=True)
                controls_path = os.path.join(
                    analysis_dir, '%s analyzer_controls %s.txt' % (diag_str, analysis_label)
                )
                write_controls_from_python(controls_path, analyzer.analyzer_dict)

                self.merge_data_frame_to_sfile(
                    add_columns_df,
                    diag_str,
                    overwrite_columns=overwrite_columns,
                    analysis_label=analysis_label,
                    keep_unanalysed=keep_unanalysed,
                )

                summary_path = os.path.join(
                    analysis_dir,
                    f"Scan{int(self.scan):03d}_{diag_str}_{extra_info_str}Summary.txt",
                )
                add_columns_df.to_csv(summary_path, index=False, sep='\t')

        return add_columns_df

    @staticmethod
    def _rename_add_columns(add_columns_df, diagnostic, analysis_label):
        """Numeric result columns named as in the sfile (see ``sfile_frame.rename_result_columns``)."""
        return rename_result_columns(add_columns_df, diagnostic, analysis_label)

    def _shot_analysis_dir(self, scan):
        """Analysis dir (created if needed) for one shot's own scan number."""
        return get_analysis_dir(self.top_dir, int(scan), make_dir=True)

    def get_scan_data_analysis_dir( self, make_dir=True ):
        return get_analysis_dir(self.top_dir, self.scan, make_dir=True)

    def display_scan(self, displayer, save=False, export=False, suffix='', fig=None, ax=None,
                     output_dir=None, **kwargs):
        """
        Render a scan-level figure using a ``ScanDisplayer``.

        Parameters
        ----------
        displayer : ScanDisplayer
            Display object implementing ``display(scan, *, fig, ax)``.
        save : bool, optional
            If True, save the resulting figure via ``displayer.save``.
        export : bool, optional
            If True, write the arrays behind the plot via ``displayer.export``
            (an ``.npz`` alongside the saved figure).
        suffix : str, optional
            Suffix appended to the saved filename.
        fig, ax : optional
            Existing matplotlib figure / axis to draw into.
        output_dir : str or Path, optional
            Folder for the saved / exported files, instead of the scan's
            analysis directory (or the displayer's ``output_subdir``).
        **kwargs
            Forwarded to ``displayer.display``.

        Returns
        -------
        (fig, ax)
        """
        fig, ax = displayer.display(self, fig=fig, ax=ax, **kwargs)
        if save:
            displayer.save(fig, self, suffix=suffix, output_dir=output_dir)
        if export:
            displayer.export(self, suffix=suffix, output_dir=output_dir)
        return fig, ax

    def merge_data_frame_to_sfile(self,
                                  add_columns_df,
                                  diagnostic,
                                  overwrite_columns=True,
                                  analysis_label=None,
                                  keep_unanalysed=True,
                                  target_filename=None,
                                  ):
        """
        Write per-shot result columns into the sfile on disk (and a summary
        file next to it). The sfile is re-read first, so columns written by
        others since loading are kept, and the read-merge-write holds a lock
        file (``<sfile>.lock``) so two writers can't interleave.

        target_filename: write into this table instead of ``self.sfilename``
            (e.g. ``scans/ScanNNN/ScanDataScanNNN.txt``); same format.

        overwrite_columns=True: an existing column of the same name is
            updated (see ``keep_unanalysed``). False: the new columns are
            added alongside with a numeric suffix.
        keep_unanalysed=True: only the shots in ``add_columns_df`` are
            updated; other shots keep their values. False: the whole column
            is replaced, so shots absent from ``add_columns_df`` become NaN.
        """
        target = os.fspath(target_filename) if target_filename is not None else self.sfilename
        add_columns_df = rename_result_columns(add_columns_df, diagnostic, analysis_label)

        with file_lock(target):
            sfile_data = pd.read_csv(target, sep='\t')
            if overwrite_columns:
                merged_df = merge_results(sfile_data, add_columns_df, keep_unanalysed=keep_unanalysed)
            else:
                # Use suffixes to avoid column conflicts
                merged_df = pd.merge(sfile_data, add_columns_df, on=['scan', 'Shotnumber'],
                                     how='left', suffixes=('', '_2'))
                col_rename = {}
                for col in merged_df.columns:
                    if col.endswith('_2'):
                        base_name = col[:-2]
                        suffix_num = sum(base_name in c for c in merged_df.columns) - 1
                        col_rename[col] = f"{base_name}_{suffix_num}"
                merged_df = merged_df.rename(columns=col_rename)
            merged_df.to_csv(target, index=False, sep='\t')

        analysis_dir = get_analysis_dir(self.top_dir, self.scan, make_dir=True)
        add_columns_path = os.path.join(analysis_dir, f"Scan{int(self.scan):03d}_{diagnostic}_{analysis_label}_Summary.txt")
        add_columns_df.to_csv(add_columns_path, index=False, sep='\t')
        print(f'Columns added to {target}')

    def mean_std_diagnostic(self, analyzer, bg=None, ddof=0, show_progress=False, rows=None):
        """
        Pixel-wise mean / std of analyzed data across all active shots.

        Parameters
        ----------
        analyzer : DiagnosticAnalyzer
        bg : optional, forwarded to ``_iter_shots``.
        ddof : int, optional
            Delta degrees of freedom (0 = population std, 1 = sample std).
        show_progress : bool, optional
            Show a tqdm bar over shots.
        rows : DataFrame, optional
            Shots to use (e.g. ``scan.rows_in_bin(b)``); default all selected shots.

        Returns
        -------
        mean_data, std_data : np.ndarray or (None, None) if no shots produced data.
        """
        arrs = [
            np.asarray(d) for _, d, _, _ in self._iter_shots(
                analyzer, bg=bg, show_progress=show_progress, rows=rows
            )
            if d is not None
        ]
        if not arrs:
            return None, None

        first_shape = arrs[0].shape
        for i, arr in enumerate(arrs):
            if arr.shape != first_shape:
                raise ValueError(
                    f"Analyzed data shape mismatch: shot 0 has shape {first_shape}, "
                    f"but shot {i} has shape {arr.shape}"
                )

        stack = np.stack(arrs, axis=0)
        return np.nanmean(stack, axis=0), np.nanstd(stack, axis=0, ddof=ddof)

    def aggregate_per_bin(self, analyzer, bg=None, bins=None, ddof=0):
        """
        Compute per-bin mean / std of analyzed data arrays.

        Parameters
        ----------
        analyzer : DiagnosticAnalyzer
        bg : optional, forwarded to ``mean_std_diagnostic``.
        bins : iterable of int, optional
            Bin numbers to include. Defaults to all unique values in
            ``active_data['temp Bin number']``.
        ddof : int, optional
            Delta degrees of freedom for the std.

        Returns
        -------
        bins : np.ndarray, shape (B,)
            Bin numbers used.
        mean_per_bin : np.ndarray, shape (B, *data_shape) or None
            NaN-filled for bins that produced no data.
        std_per_bin : np.ndarray, shape (B, *data_shape) or None

        """
        bins = self.bins() if bins is None else np.asarray(list(bins))

        mean_per_bin = [None] * len(bins)
        std_per_bin = [None] * len(bins)
        data_shape = None
        for i, (b, rows) in enumerate(self.iter_bins(bins)):
            mean_data, std_data = self.mean_std_diagnostic(analyzer, bg=bg, ddof=ddof, rows=rows)
            if mean_data is None:
                continue
            data_shape = mean_data.shape
            mean_per_bin[i] = mean_data
            std_per_bin[i] = std_data

        if data_shape is None:
            return bins, None, None

        nan_fill = np.full(data_shape, np.nan)
        mean_per_bin = [x if x is not None else nan_fill for x in mean_per_bin]
        std_per_bin = [x if x is not None else nan_fill for x in std_per_bin]

        return bins, np.asarray(mean_per_bin), np.asarray(std_per_bin)

    def _resolve_bg_for_multi(self, analyzer, bg, context):
        """
        Resolve the per-row background for a ``MultiDiagnosticAnalyzer``.

        ``bg`` can be:
          - ``None`` — returned as-is.
          - a dict ``{diagnostic_name: bg_spec}`` — each ``bg_spec`` is
            resolved through ``_resolve_bg_for_row`` using that input's
            sub-analyzer as the loader. Entries without a registered
            sub-analyzer are returned as-is (assumed pre-loaded).
          - anything else — passed through; the multi analyzer's
            ``analyze_data`` is responsible for interpreting it.
        """
        if bg is None or not isinstance(bg, dict):
            return bg

        out = {}
        for name, _ in analyzer.inputs:
            sub = analyzer.sub_analyzers.get(name)
            sub_bg = bg.get(name)
            if sub is None:
                out[name] = sub_bg
            else:
                out[name] = self._resolve_bg_for_row(sub, sub_bg, context)
        return out

    @staticmethod
    def _resolve_bg_for_row(analyzer, bg, context, debug_bg=False, debug_once=True):
        """
        This function is used when the bg is not the same for every shot in the scan.
        If bg is a function that takes the argument (context), it will return the bg for that row dict.
        You just need to write the function that selects the correct bg for that shot

        bg can be:
        - None
        - already-loaded background (returned as-is)
        - a path/filename (loaded via analyzer.load_data)
        - a callable: bg(context) -> None | loaded_bg | path
        - an object with .get(context) -> None | loaded_bg | path
        """

        if bg is None:
            return None

        # Provider object with .get(context); pandas objects also have
        # .get, but a DataFrame / Series is loaded bg data (e.g. a spectrum)
        if hasattr(bg, "get") and callable(bg.get) and not isinstance(bg, (pd.DataFrame, pd.Series)):
            bg_spec = bg.get(context)

        # Callable provider
        elif callable(bg):
            bg_spec = bg(context)

        # Static
        else:
            bg_spec = bg

        if bg_spec is None:
            return None

        # If it's a path-like, load it
        if isinstance(bg_spec, (str, Path, os.PathLike)):
            bg_path = str(bg_spec)

            if debug_bg:
                # Use an attribute on analyzer to remember if we've printed already
                if (not debug_once) or (not getattr(analyzer, "_bg_debug_printed", False)):
                    print(f"[BG] loading background from: {bg_path}")
                    analyzer._bg_debug_printed = True

            return analyzer.load_data(bg_path)

        # Otherwise assume it's already loaded bg data
        return bg_spec
    
    def compute_bin_summary(self, mode='mean'):
        valid_modes = ('mean', 'median', 'std_err', 'max', 'min')
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}.")

        numeric_cols = self.active_data.select_dtypes(include=np.number)
        grouped = numeric_cols.groupby('temp Bin number')

        if mode == 'mean':
            center_df = grouped.mean().reset_index()
            spread_df = grouped.std(ddof=1).reset_index()

        elif mode == 'median':
            center_df = grouped.median().reset_index()
            q25 = grouped.quantile(0.25)
            q75 = grouped.quantile(0.75)
            spread_df = (q75 - q25).reset_index()

        elif mode == 'std_err':
            center_df = grouped.mean().reset_index()
            spread_df = grouped.sem().reset_index()

        elif mode == 'max':
            center_df = grouped.max().reset_index()
            spread_df = center_df.copy()
            spread_df[spread_df.columns.difference(['temp Bin number'])] = 0

        elif mode == 'min':
            center_df = grouped.min().reset_index()
            spread_df = center_df.copy()
            spread_df[spread_df.columns.difference(['temp Bin number'])] = 0

        return center_df, spread_df

    # LabVIEW controls parsers, kept as static methods for existing callers
    parse_column_math_from_lv_controls = staticmethod(parse_column_math_from_lv_controls)
    parse_search_replace_from_lv_controls = staticmethod(parse_search_replace_from_lv_controls)

    def open_scan_dir(self):
        scan_dir = os.path.join(self.top_dir, 'scans', 'Scan%03d' %self.scan)
        open_directory_in_explorer(scan_dir)
        
    def open_analysis_dir(self):
        if self.analysis_dir is not None:
            open_directory_in_explorer(self.analysis_dir)
            
    def open_top_dir(self):
        open_directory_in_explorer(self.top_dir)

