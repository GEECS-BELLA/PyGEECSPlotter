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
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from PyGEECSPlotter.navigation_utils import *
from PyGEECSPlotter.utils import parse_controls_from_python, write_controls_from_python
from PyGEECSPlotter.binning import compute_bin_numbers
# from PyGEECSPlotter.navigation_utils import get_analysis_dir, get_analysis_diagnostic_path, open_directory_in_explorer, get_analysed_shot_save_path

import PyGEECSPlotter.plotting as gplt
colors = gplt.configure_plotting()



class ScanDataAnalyzer:
    def __init__(self, 
                sfilename=None, 
                top_dir=None, 
                experiment_dir=None,
                year=None, 
                month=None, 
                day=None, 
                scan=None
                ):
                
        self.sfilename = sfilename
        self.top_dir = top_dir
        self.experiment_dir = experiment_dir
        self._year = None
        self._month = None
        self._day = None
        self._scan = None
        self.year = year
        self.month = month
        self.day = day
        self.scan = scan
        self.scan_parameter = 'Shotnumber'

        if sfilename:
            self._init_from_sfilename()
        elif top_dir and scan:
            self._init_from_top_dir_and_scan()
        elif experiment_dir and all([year, month, day, scan]):
            self._init_from_experiment_path()
        else:
            raise ValueError("Invalid initialization arguments. Use one of:\n"
                            "1. sfilename\n"
                            "2. top_dir and scan\n"
                            "3. experiment_dir, year, month, day, scan")
            
        self.analysis_dir = None
        self.data_columns = None
        self.data = None
        self.aborted = False
        self.print_data = False

        self.scan_title = self.scan_data_title()
        self._mask = None

    def _init_from_sfilename(self):
        top_dir, year, month, day = get_top_dir_from_sfilename(self.sfilename)
        self.top_dir = top_dir
        self.year = year
        self.month = month
        self.day = day

    def _init_from_top_dir_and_scan(self):
        self.sfilename = get_sfilename_from_top_dir(self.top_dir, self.scan)
        top_dir, year, month, day = get_top_dir_from_sfilename(self.sfilename)
        self.top_dir = top_dir
        self.year = year
        self.month = month
        self.day = day

    def _init_from_experiment_path(self):
        self.top_dir = get_top_dir(self.experiment_dir, self.year, self.month, self.day)
        self.sfilename = get_sfilename_from_top_dir(self.top_dir, self.scan)

    @staticmethod
    def _to_int(value, name):
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            raise TypeError(f"'{name}' must be an integer or integer-castable value, got {type(value).__name__!r}: {value!r}")

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



    def __repr__(self):
        return (f"ExperimentFile(sfilename={self.sfilename}, scan={self.scan}, "
                f"date={self.year}-{self.month:02d}-{self.day:02d}, "
                f"experiment_dir={self.experiment_dir})")
    
    
    def _init_mask(self):
        """Initialize the inclusion mask to include all rows."""
        self._mask = np.ones(len(self.data), dtype=bool)

    def save_mask(self):
        """Save the current mask state."""
        return self._mask.copy()

    def restore_mask(self, saved_mask):
        """Restore a previously saved mask state."""
        self._mask = saved_mask


    @property
    def active_data(self):
        """Return only the rows currently included by the mask."""
        return self.data[self._mask].reset_index(drop=True)
    
    def scan_data_title(self, append_text=None):
        if append_text is not None:
            title = f'{self.year}-{self.month:02d}-{self.day:02d} - {os.path.basename(self.sfilename)} - {append_text}'
        else:
            title = f'{self.year}-{self.month:02d}-{self.day:02d} - {os.path.basename(self.sfilename)}'
        return title
    

    def load_scan_data(self, 
        search_replace_filename=None, 
        column_math_filename=None, 
        analyzer=None,
        parse_from='python', 
        remove_missing_diagnostic_files=True,
        ):

        sfile_data = pd.read_csv(self.sfilename, sep='\t')
        if len(sfile_data) == 0:
            self.data = sfile_data
            self.scan_parameter = 'Shotnumber'
            self.aborted = True

        else:
            scan_parameter, scan = get_scan_parameter(self.top_dir, sfile_data)
            self.scan = scan
            self.scan_parameter = scan_parameter

            self.data = sfile_data.copy()
            self._init_mask()
            for col in self.data.columns:
                self.data.rename(columns={col: get_parameter_alias(col)}, inplace=True)

            self.data['temp Bin number'] = self.data['Bin #']

            if search_replace_filename is not None:
                self.add_search_replace_column_names(search_replace_filename, parse_from=parse_from)
                self.add_search_replace_scan_parameter(search_replace_filename, parse_from=parse_from)
            if column_math_filename is not None:
                self.add_column_math(column_math_filename, parse_from=parse_from)
            
            if analyzer is not None:
                self.add_file_list_to_scan_data(analyzer.diagnostic, analyzer.file_ext, remove_missing_diagnostic_files)
            self.set_analysis_dir()

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
        if parse_from == 'labview':
            search_replace_pairs = ScanDataAnalyzer.parse_search_replace_from_lv_controls(search_replace_filename)
        elif parse_from == 'python':
            search_replace_pairs = parse_controls_from_python(search_replace_filename)
        else:
            print("parse_from options 'labview' or 'python'. No search and replace done")
            return
        
        for col in self.data.columns:
            new_col_name = col
            for pair in search_replace_pairs:
                new_col_name = new_col_name.replace(pair['Search'], pair['Replace'])
            self.data.rename(columns={col: new_col_name}, inplace=True)

    def add_search_replace_scan_parameter(self, search_replace_filename, parse_from='labview'):
        if parse_from == 'labview':
            search_replace_pairs = ScanDataAnalyzer.parse_search_replace_from_lv_controls(search_replace_filename)
        elif parse_from == 'python':
            search_replace_pairs = parse_controls_from_python(search_replace_filename)
        else:
            print("parse_from options 'labview' or 'python'. No search and replace done")
            return
        
        new_param = self.scan_parameter
        for pair in search_replace_pairs:
            new_param = new_param.replace(pair['Search'], pair['Replace'])
        self.scan_parameter = new_param

    def add_column_math(self, column_math_filename, parse_from='labview'):
        if parse_from == 'labview':
            math_clusters_list = ScanDataAnalyzer.parse_column_math_from_lv_controls(column_math_filename)
        elif parse_from == 'python':
            math_clusters_list = parse_controls_from_python(column_math_filename)
        else:
            print("parse_from options 'labview' or 'python'. No column math added")
            return

        for math_cluster in math_clusters_list:
            column_name = math_cluster['column_name']
            all_vars_present = all(
                v['column_name'] in self.data.columns for v in math_cluster['variables']
            )

            if all_vars_present:
                formula = math_cluster['formula'].replace('^', '**')
                for func in ["sqrt", "exp", "sin", "cos", "abs", "min", "max"]:
                    formula = formula.replace(func, f"np.{func}")

                # Make variable assignments
                local_vars = {}
                for v in math_cluster['variables']:
                    local_vars[v['vars']] = self.data[v['column_name']]
                
                # Evaluate and assign result
                try:
                    result = eval(formula, {"np": np}, local_vars)
                    self.data[column_name] = result
                except Exception as e:
                    if self.print_data:
                        print(f"Error evaluating formula for {column_name}: {e}")
            else:
                if self.print_data:
                    print(f"Column {column_name} not added — missing input columns.")

    def merge_column_math_to_sfile(self, column_math_filename, parse_from='labview'):
        if not self.aborted:
            if parse_from == 'labview':
                math_clusters_list = ScanDataAnalyzer.parse_column_math_from_lv_controls(column_math_filename)
            elif parse_from == 'python':
                math_clusters_list = parse_controls_from_python(column_math_filename)
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
            print('No directory found')
            self.data['%s file_list' % diagnostic] = ''
            self.data['%s file_exists' % diagnostic] = 0
            if remove_missing_files:
                missing_condition = self.data['%s file_exists' % diagnostic] == 0
                self._mask = self._mask & ~missing_condition.to_numpy()
                if self.print_data:
                    print('Masked out %d lines from scan_data for missing files' % missing_condition.sum())
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
            missing_condition = self.data['%s file_exists' % diagnostic] == 0
            n_missing = missing_condition.sum()
            self._mask = self._mask & ~missing_condition.to_numpy()
            if self.print_data:
                print('Masked out %d lines from scan_data for missing files' % n_missing)



    def filter_scan_data(self, filter_parameter, lower_bound, upper_bound,
                        filter_exclusive=False):
        """
        Filter scan data based on a specified parameter and value range.

        Updates the internal boolean mask so that subsequent filters are
        combined with AND logic. The underlying `self.data` is never modified.

        Parameters
        ----------
        filter_parameter : str
            Column name in `self.data` to apply the filter on.
        lower_bound : float
            Lower bound of the filtering range.
        upper_bound : float
            Upper bound of the filtering range.
        filter_exclusive : bool, optional
            If True, keep rows outside the range (exclude rows inside).
            If False, keep rows inside the range (exclude rows outside).
        """

        if not hasattr(self, '_mask') or self._mask is None:
            self._init_mask()

        if filter_exclusive:
            new_condition = (
                (self.data[filter_parameter] < lower_bound) |
                (self.data[filter_parameter] > upper_bound)
            )
        else:
            new_condition = (
                (self.data[filter_parameter] > lower_bound) &
                (self.data[filter_parameter] < upper_bound)
            )

        self._mask = self._mask & new_condition.to_numpy()

        print(
            '%d / %d shots included. Filtered based on: %s'
            % (self._mask.sum(), len(self.data), get_parameter_alias(filter_parameter))
        )


    def filter_scan_data_by_array(self, filter_parameter, values,
                                filter_exclusive=False):
        """
        Filter scan data based on whether a parameter value is in a given array.

        Updates the internal boolean mask so that subsequent filters are
        combined with AND logic. The underlying `self.data` is never modified.

        Parameters
        ----------
        filter_parameter : str
            Column name in `self.data` to filter on.
        values : array-like
            List, numpy array, or pandas Series of allowed (or excluded) values.
        filter_exclusive : bool, optional
            If True, keep rows whose values are NOT in `values`.
            If False, keep rows whose values ARE in `values`.
        """

        if not hasattr(self, '_mask') or self._mask is None:
            self._init_mask()

        values = np.asarray(values)

        if filter_exclusive:
            new_condition = ~self.data[filter_parameter].isin(values)
        else:
            new_condition = self.data[filter_parameter].isin(values)

        self._mask = self._mask & new_condition.to_numpy()

        print(
            '%d / %d shots included. Filtered based on: %s'
            % (self._mask.sum(), len(self.data), get_parameter_alias(filter_parameter))
        )

    def reset_filters(self):
        """
        Reset the inclusion mask so that all rows in `self.data` are included.
        """
        self._init_mask()
        print('Filters reset. %d shots included.' % self._mask.sum())

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

    def _iter_shots(self, analyzer, bg=None, show_progress=True):
        """
        Yield ``(context, data, results, aux)`` for each active shot.

        Centralizes the per-shot iteration shell:
          1) iterate ``active_data``
          2) load the diagnostic file (if any)
          3) resolve the per-row background
          4) call ``analyzer.analyze_data``

        Used by ``analyze_scan`` and any per-shot aggregation
        (``mean_std_diagnostic``, ``aggregate_per_bin``, custom workflows).

        Parameters
        ----------
        analyzer : DiagnosticAnalyzer
        bg : optional
            Background spec passed through ``_resolve_bg_for_row``.
        show_progress : bool, optional
            Wrap iteration in a tqdm progress bar.

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
        rows = self.active_data
        it = rows.iterrows()
        if show_progress:
            it = tqdm(it, total=rows.shape[0])

        for _, row in it:
            context = row.to_dict()

            if analyzer.diagnostic is not None:
                filename = context.get(f'{analyzer.diagnostic} file_list')
                data = analyzer.load_data(filename)
            else:
                data = None

            bg_i = self._resolve_bg_for_row(analyzer, bg, context)
            data, results, aux = analyzer.analyze_data(data, bg=bg_i, context=context)

            yield context, data, results, aux

    def analyze_scan(self, analyzer,
        bg=None,
        display_data=False,
        write_columns_to_sfile=False,
        overwrite_columns=True,
        analysis_label='',
        write_analyzed=False,
        write_lineouts=False,
        close_displayed=True,
        ):
        """
        Run the per-shot pipeline over active shots and handle side effects.

        Iteration is delegated to ``_iter_shots``. This method applies the
        optional per-shot side effects (display, write_analyzed,
        write_lineouts, write_displayed), accumulates scalar results, and
        optionally merges them back into the sfile.

        Parameters
        ----------
        analyzer : DiagnosticAnalyzer
            Provides ``load_data``, ``analyze_data``, and optionally
            ``display_data`` / ``write_analyzed_data`` /
            ``write_analyzed_lineouts`` / ``write_displayed_data``.
        bg : optional
            Background spec; see ``_resolve_bg_for_row``.

        Returns
        -------
        add_columns_df : DataFrame or None
        """
        rows = []
        analysis_dir = None

        for context, data, results, aux in self._iter_shots(analyzer, bg=bg):
            scan, shot_num = context['scan'], context['Shotnumber']
            rows.append({'scan': scan, 'Shotnumber': shot_num, **results})

            if data is None:
                continue

            fig = None
            if display_data:
                filename = context.get(f'{analyzer.diagnostic} file_list', '')
                fig, _ = analyzer.display_data(
                    data, return_dict=results, title=os.path.basename(filename)
                )

            if write_analyzed:
                if analysis_dir is None:
                    analysis_dir = self.get_scan_data_analysis_dir(make_dir=True)
                analyzer.write_analyzed_data(data, analysis_dir, scan, shot_num, context=context)
                if write_lineouts:
                    analyzer.write_analyzed_lineouts(aux, analysis_dir, scan, shot_num)
                if display_data:
                    analyzer.write_displayed_data(fig, analysis_dir, scan, shot_num)

            if close_displayed and fig is not None:
                plt.close(fig)

        add_columns_df = pd.DataFrame(rows) if rows else None

        if write_columns_to_sfile and add_columns_df is not None and len(self.data) > 0:
            diag_str = analyzer.output_diagnostic if analyzer.output_diagnostic is not None \
                else analyzer.diagnostic
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
            )

        return add_columns_df
    
    def get_scan_data_analysis_dir( self, make_dir=True ):
        return get_analysis_dir(self.top_dir, self.scan, make_dir=True)

    def display_scan(self, displayer, save=False, suffix='', fig=None, ax=None, **kwargs):
        """
        Render a scan-level figure using a ``ScanDisplayer``.

        Parameters
        ----------
        displayer : ScanDisplayer
            Display object implementing ``display(scan, *, fig, ax)``.
        save : bool, optional
            If True, save the resulting figure via ``displayer.save``.
        suffix : str, optional
            Suffix appended to the saved filename.
        fig, ax : optional
            Existing matplotlib figure / axis to draw into.
        **kwargs
            Forwarded to ``displayer.display``.

        Returns
        -------
        (fig, ax)
        """
        fig, ax = displayer.display(self, fig=fig, ax=ax, **kwargs)
        if save:
            displayer.save(fig, self, suffix=suffix)
        return fig, ax

    def merge_data_frame_to_sfile(self, 
                                  add_columns_df, 
                                  diagnostic,
                                  overwrite_columns=True, 
                                  analysis_label=None, 
                                  ):
        # Load the original data from the sfile
        sfile_data = pd.read_csv(self.sfilename, sep='\t')
        
        # Filter out any non-numeric columns in add_columns_df
        add_columns_df = add_columns_df.select_dtypes(include=[np.number, 'float64', 'int64', 'bool'])

        # Apply renaming to add_columns_df, excluding 'scan' and 'Shotnumber'
        if diagnostic or analysis_label:
            def rename_columns(col):
                if col in ['scan', 'Shotnumber']:
                    return col  # Do not rename 'scan' or 'Shotnumber'
                new_col = col
                if diagnostic:
                    new_col = f"{diagnostic} {new_col}"
                if analysis_label:
                    new_col = f"{new_col} {analysis_label}"
                return new_col

            add_columns_df = add_columns_df.rename(columns=rename_columns)

        if overwrite_columns:
            # Drop the overlapping columns from sfile_data so they can be overwritten
            sfile_data = sfile_data.drop(columns=[col for col in add_columns_df.columns 
                                                  if col in sfile_data.columns and col not in ['scan', 'Shotnumber']])
            # Merge to overwrite columns
            merged_df = pd.merge(sfile_data, add_columns_df, on=['scan', 'Shotnumber'], how='left')
        else:
            # Use suffixes to avoid column conflicts
            merged_df = pd.merge(sfile_data, add_columns_df, on=['scan', 'Shotnumber'], how='left', suffixes=('', '_2'))

            # Rename columns with incremental suffixes if needed
            col_rename = {}
            for col in merged_df.columns:
                if col.endswith('_2'):
                    base_name = col[:-2]
                    # Ensure unique naming by counting existing occurrences
                    suffix_num = sum(base_name in c for c in merged_df.columns) - 1
                    new_name = f"{base_name}_{suffix_num}"
                    col_rename[col] = new_name

            # Apply the renaming to avoid conflicts
            merged_df = merged_df.rename(columns=col_rename)

        # Save the merged data back to the sfile
        analysis_dir = get_analysis_dir(self.top_dir, self.scan, make_dir=True)
        add_columns_path = os.path.join(self.analysis_dir, f"Scan{int(self.scan):03d}_{diagnostic}_{analysis_label}_Summary.txt")
        add_columns_df.to_csv( add_columns_path , index=False, sep='\t' )
        merged_df.to_csv(self.sfilename, index=False, sep='\t')
        print(f'Columns added to {self.sfilename}')

    def mean_std_diagnostic(self, analyzer, bg=None, ddof=0, show_progress=False):
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

        Returns
        -------
        mean_data, std_data : np.ndarray or (None, None) if no shots produced data.
        """
        arrs = [
            np.asarray(d) for _, d, _, _ in self._iter_shots(
                analyzer, bg=bg, show_progress=show_progress
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

        Notes
        -----
        Restores the scan's filter mask after running, regardless of error.
        """
        if bins is None:
            bins = np.unique(self.active_data['temp Bin number'])
        bins = np.asarray(list(bins))

        mean_per_bin = [None] * len(bins)
        std_per_bin = [None] * len(bins)
        saved = self.save_mask()
        data_shape = None

        try:
            for i, b in enumerate(bins):
                self.restore_mask(saved)
                self.filter_scan_data('temp Bin number', b - 0.1, b + 0.1)
                mean_data, std_data = self.mean_std_diagnostic(analyzer, bg=bg, ddof=ddof)
                if mean_data is None:
                    continue
                data_shape = mean_data.shape
                mean_per_bin[i] = mean_data
                std_per_bin[i] = std_data
        finally:
            self.restore_mask(saved)

        if data_shape is None:
            return bins, None, None

        nan_fill = np.full(data_shape, np.nan)
        mean_per_bin = [x if x is not None else nan_fill for x in mean_per_bin]
        std_per_bin = [x if x is not None else nan_fill for x in std_per_bin]

        return bins, np.asarray(mean_per_bin), np.asarray(std_per_bin)

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

        # Provider object with .get(context)
        if hasattr(bg, "get") and callable(bg.get):
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

        numeric_cols = self.data.select_dtypes(include=np.number)
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

    @staticmethod
    def parse_column_math_from_lv_controls(file_path):
        clusters = []
        current_cluster = {}
        variables = []

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('Value') and '.Cluster.New (or old) column name' in line:
                    if current_cluster:
                        current_cluster['variables'] = variables
                        clusters.append(current_cluster)
                        variables = []
                    current_cluster = {'column_name': line.split('=')[1].strip()}
                elif line.startswith('Value') and '.Cluster.Formula' in line:
                    current_cluster['formula'] = line.split('=')[1].strip()
                elif '.Cluster.vars' in line and '.Cluster.Column name' in line:
                    # This is not expected to happen as per the provided format,
                    # it seems to be a misunderstanding of the format.
                    pass
                elif '.Cluster.vars' in line or '.Cluster.Column name' in line:
                    var_name = line.split('=')[1].strip()
                    var_key = 'vars' if '.Cluster.vars' in line else 'column_name'
                    if variables and var_key in variables[-1]:
                        # If the last variable already has this key, start a new variable
                        variables.append({var_key: var_name})
                    elif variables:
                        variables[-1][var_key] = var_name
                    else:
                        variables.append({var_key: var_name})

            # Add the last cluster if it exists
            if current_cluster:
                current_cluster['variables'] = variables
                clusters.append(current_cluster)

        return clusters

    @staticmethod
    def parse_search_replace_from_lv_controls(file_path):
        search_replace_pairs = []
        
        with open(file_path, 'r') as file:
            text_content = file.read()

        lines = text_content.strip().split('\n')
        for line in lines:
            if 'Cluster.Search=' in line:
                search_term = line.split('=')[1]
                if "Bella" in search_term:
                    continue
                replace_term = next((l.split('=')[1] for l in lines if l.startswith(line.split('.Search')[0] + '.Replace=')), '')
                search_replace_pairs.append({'Search': search_term, 'Replace': replace_term})

        return search_replace_pairs

    def open_scan_dir(self):
        scan_dir = os.path.join(self.top_dir, 'scans', 'Scan%03d' %self.scan)
        open_directory_in_explorer(scan_dir)
        
    def open_analysis_dir(self):
        if self.analysis_dir is not None:
            open_directory_in_explorer(self.analysis_dir)
            
    def open_top_dir(self):
        open_directory_in_explorer(self.top_dir)

