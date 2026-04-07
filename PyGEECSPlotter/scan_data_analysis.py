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

from PyGEECSPlotter.navigation_utils import *
from PyGEECSPlotter.utils import parse_controls_from_python, write_controls_from_python
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
        self.filtered_out_data = None
        self.aborted = False
        self.print_data = False

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



    def __repr__(self):
        return (f"ExperimentFile(sfilename={self.sfilename}, scan={self.scan}, "
                f"date={self.year}-{self.month:02d}-{self.day:02d}, "
                f"experiment_dir={self.experiment_dir})")
    

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

            self.filtered_out_data = pd.DataFrame(columns=self.data.columns)

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
        
        if os.path.exists( os.path.join(self.top_dir, 'scans', 'Scan%03d' %self.scan, diagnostic) ):
            diagnostic_dir = os.path.join(self.top_dir, 'scans', 'Scan%03d' %self.scan, diagnostic)
            analysis_diagnostic = False
        elif os.path.exists( os.path.join(self.top_dir, 'analysis', 'Scan%03d' %self.scan, diagnostic) ):
            diagnostic_dir = os.path.join(self.top_dir, 'analysis', 'Scan%03d' %self.scan, diagnostic)
            analysis_diagnostic = True
        else:
            print('No directory found')

            self.data['%s file_list' %diagnostic] = ''
            self.data['%s file_exists' %diagnostic] = 0

            if remove_missing_files:
                self.data = self.data[self.data['%s file_exists' %diagnostic] != 0].reset_index(drop=True)
                if self.print_data:
                    n_missing = np.sum(1 - np.array(file_exists))
                    print('Removed %d lines from scan_data for missing files' %n_missing)

            return

        file_list = []
        file_exists = []
        for i in range(len(self.data)):
            scan = self.data['scan'][i]
            shot_num = self.data['Shotnumber'][i]

            if analysis_diagnostic:
                diagnostic_dir = os.path.join(self.top_dir, 'analysis', 'Scan%03d' %scan, diagnostic)
            else:
                diagnostic_dir = os.path.join(self.top_dir, 'scans', 'Scan%03d' %scan, diagnostic)

            basename = r'Scan%03d_%s_%03d%s' %(scan, diagnostic, shot_num, file_ext)
            filename = os.path.join(diagnostic_dir, basename)

            file_list.append( filename )
            file_exists.append( os.path.exists( filename ) )    

            if not os.path.exists( filename ) and self.print_data:
                print('Did not save : %s' %os.path.basename(filename))

        if self.print_data:
            print('Files to analyse : %d' %len(file_list) )

        self.data['%s file_list' %diagnostic] = file_list
        self.data['%s file_exists' %diagnostic] = file_exists

        if remove_missing_files:
            self.data = self.data[self.data['%s file_exists' %diagnostic] != 0].reset_index(drop=True)
            if self.print_data:
                n_missing = np.sum(1 - np.array(file_exists))
                print('Removed %d lines from scan_data for missing files' %n_missing)

    def filter_scan_data(self, filter_parameter, lower_bound, upper_bound,
                        filter_exclusive=False, update_data=False):
        """
        Filter scan data based on a specified parameter and value range.

        Instead of discarding excluded rows, this function moves them into
        `self.filtered_out_data`, which acts as a holding DataFrame for filtered-out data.

        Parameters
        ----------
        filter_parameter : str
            Column name in `self.data` to apply the filter on.
        lower_bound : float
            Lower bound of filtering range.
        upper_bound : float
            Upper bound of filtering range.
        filter_exclusive : bool, optional
            If True, keep rows outside the range and move rows inside the range
            to `self.filtered_out_data`.
            If False, keep rows inside the range and move rows outside the range
            to `self.filtered_out_data`.
        update_data : bool, optional
            If True, update `self.data` in place. Defaults to False.

        Returns
        -------
        filtered_scan_data : pd.DataFrame
            The rows kept after filtering.
        moved_scan_data : pd.DataFrame
            The rows moved into `self.filtered_out_data`.
        """

        if not hasattr(self, 'filtered_out_data') or self.filtered_out_data is None:
            self.filtered_out_data = pd.DataFrame(columns=self.data.columns)

        if filter_exclusive:
            keep_idcs = (
                (self.data[filter_parameter] < lower_bound) |
                (self.data[filter_parameter] > upper_bound)
            )
        else:
            keep_idcs = (
                (self.data[filter_parameter] > lower_bound) &
                (self.data[filter_parameter] < upper_bound)
            )

        filtered_scan_data = self.data.loc[keep_idcs].copy().reset_index(drop=True)
        moved_scan_data = self.data.loc[~keep_idcs].copy().reset_index(drop=True)

        print(
            '%d / %d shots included. Filtered based on : %s'
            % (len(filtered_scan_data), len(self.data), get_parameter_alias(filter_parameter))
        )

        if update_data:
            if not moved_scan_data.empty:
                self.filtered_out_data = pd.concat(
                    [self.filtered_out_data, moved_scan_data],
                    ignore_index=True
                )

            self.data = filtered_scan_data

        return filtered_scan_data, moved_scan_data


    def reset_filters(self):
        """
        Move all previously filtered-out rows from `self.filtered_out_data`
        back into `self.data`, then empty `self.filtered_out_data`.

        Memory-efficient version: avoids copies and intermediate column expansion.
        """

        if (
            not hasattr(self, 'filtered_out_data') or
            self.filtered_out_data is None or
            self.filtered_out_data.empty
        ):
            sort_cols = [col for col in ['scan', 'Shotnumber'] if col in self.data.columns]
            if sort_cols:
                self.data = self.data.sort_values(by=sort_cols).reset_index(drop=True)
            return self.data

        # Use pd.concat directly — it handles mismatched columns natively
        # by filling missing values with NaN, without needing to pre-expand
        # columns on either dataframe.
        # 
        # We do NOT call .copy() — concat already produces a new dataframe.
        self.data = pd.concat(
            [self.data, self.filtered_out_data],
            ignore_index=True,
            sort=False,  # preserve column order from self.data
        )

        # Reorder columns: self.data columns first (already guaranteed by sort=False
        # in concat when self.data is the first argument), but let's be explicit
        # to also ensure any filtered_out_data-only columns appear at the end.
        # This is already the default behavior of concat with sort=False, so
        # no additional reindex step is needed.

        # Sort if possible
        sort_cols = [col for col in ['scan', 'Shotnumber'] if col in self.data.columns]
        if sort_cols:
            self.data = self.data.sort_values(by=sort_cols).reset_index(drop=True)

        # Empty filtered_out_data, preserving the final column structure
        self.filtered_out_data = self.data.iloc[0:0].copy()

        return self.data

    def filter_scan_data_by_array(
            self,
            filter_parameter,
            values,
            filter_exclusive=False,
            update_data=False
        ):
        """
        Filter scan data based on whether a parameter value is in a given array.

        Instead of discarding excluded rows, this function moves them into
        `self.filtered_out_data`.

        Parameters
        ----------
        filter_parameter : str
            Column name in `self.data` to filter on.
        values : array-like
            List, numpy array, or pandas Series of allowed values.
        filter_exclusive : bool, optional
            If True, keep rows whose values are NOT in `values`, and move rows
            whose values ARE in `values` to `self.filtered_out_data`.
            If False, keep only rows whose values ARE in `values`, and move rows
            whose values are NOT in `values` to `self.filtered_out_data`.
        update_data : bool, optional
            If True, update `self.data` in place.

        Returns
        -------
        filtered_scan_data : pd.DataFrame
            The rows kept after filtering.
        moved_scan_data : pd.DataFrame
            The rows moved into `self.filtered_out_data`.
        """

        if not hasattr(self, 'filtered_out_data') or self.filtered_out_data is None:
            self.filtered_out_data = pd.DataFrame(columns=self.data.columns)

        # Ensure array-like input
        values = np.asarray(values)

        if filter_exclusive:
            keep_idcs = ~self.data[filter_parameter].isin(values)
        else:
            keep_idcs = self.data[filter_parameter].isin(values)

        filtered_scan_data = self.data.loc[keep_idcs].copy().reset_index(drop=True)
        moved_scan_data = self.data.loc[~keep_idcs].copy().reset_index(drop=True)

        print(
            '%d / %d shots included. Filtered based on : %s'
            % (
                len(filtered_scan_data),
                len(self.data),
                get_parameter_alias(filter_parameter)
            )
        )

        if update_data:
            if not moved_scan_data.empty:
                self.filtered_out_data = pd.concat(
                    [self.filtered_out_data, moved_scan_data],
                    ignore_index=True
                )

            self.data = filtered_scan_data

        return filtered_scan_data, moved_scan_data

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
        Processes scan data with analysis and optional display and file writing.

        Parameters:
            analyzer (object): Analyzer object with load_data, analyze_data, write_analyzed_data, and display_data methods.
            bg (optional): Background data or parameters for the analysis.
        Returns:
            add_columns_df (DataFrame) 
        """

        add_columns_df = None

        for i in range( len(self.data) ):
            row_dict = self.data.iloc[i].to_dict()
            scan, shot_num = row_dict['scan'], row_dict['Shotnumber']
            filename = row_dict[f'{analyzer.diagnostic} file_list']
            
            data = analyzer.load_data(filename)
            
            bg_i = self._resolve_bg_for_row(analyzer, bg, row_dict)
            data, return_dict, lineouts = analyzer.analyze_data(data, bg=bg_i, row_dict=row_dict)
            
            add_columns_df = ScanDataAnalyzer.append_to_add_columns_df( scan, shot_num, return_dict, add_columns_df )
            
            if data is not None:
                if display_data:
                    fig, ax = analyzer.display_data(data, return_dict=return_dict, title=os.path.basename(filename))
            
                if write_analyzed:
                    analysis_dir = self.get_scan_data_analysis_dir( make_dir=True )
                    analyzer.write_analyzed_data( data, analysis_dir, scan, shot_num )

                    if write_lineouts:
                        analyzer.write_analyzed_lineouts( lineouts, analysis_dir, scan, shot_num )
            
                    if display_data:
                        analyzer.write_displayed_data( fig, analysis_dir, scan, shot_num )

                if close_displayed and display_data:
                    plt.close( fig )

        if write_columns_to_sfile and len(self.data) > 0:
            if analyzer.output_diagnostic is not None:
                diag_str = analyzer.output_diagnostic
            else:
                diag_str = analyzer.diagnostic

            analysis_dir = self.get_scan_data_analysis_dir( make_dir=True )
            controls_path = os.path.join(analysis_dir, '%s analyzer_controls %s.txt' % (diag_str, analysis_label) )
            write_controls_from_python(controls_path, analyzer.analyzer_dict)

            self.merge_data_frame_to_sfile(add_columns_df, 
                            diag_str,
                            overwrite_columns=overwrite_columns, 
                            analysis_label=analysis_label, 
                                        )

        return add_columns_df
    
    def get_scan_data_analysis_dir( self, make_dir=True ):
        return get_analysis_dir(self.top_dir, self.scan, make_dir=True)

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

    def analyze_scan_data_mean_std(
            self,
            analyzer,
            bg=None,
            ignore_none=True,
            ddof=0,
        ):
        """
        Analyze all shots in self.data and return the mean and standard deviation
        of the `data` returned by analyzer.analyze_data.

        Parameters
        ----------
        analyzer : object
            Analyzer object with load_data and analyze_data methods.
        bg : optional
            Background data or parameters for the analysis.
        ddof : int, optional
            Delta degrees of freedom for the standard deviation.
            Use ddof=0 for population std, ddof=1 for sample std.

        Returns
        -------
        mean_data : np.ndarray
            Pixelwise mean of analyzed data over all valid shots.
        std_data : np.ndarray
            Pixelwise standard deviation of analyzed data over all valid shots.

        Notes
        -----
        - All returned `data` arrays must have the same shape.
        - This function does not display data or write output files.
        - This function ignores `return_dict` and `lineouts`; it is only for
        aggregating the main analyzed `data`.
        """

        data_list = []

        for i in range(len(self.data)):
            row_dict = self.data.iloc[i].to_dict()
            filename = row_dict[f'{analyzer.diagnostic} file_list']

            data = analyzer.load_data(filename)

            bg_i = self._resolve_bg_for_row(analyzer, bg, row_dict)
            data, return_dict, lineouts = analyzer.analyze_data(
                data,
                bg=bg_i,
                row_dict=row_dict
            )

            if data is not None:
                data_list.append(np.asarray(data))

        if len(data_list) == 0:
            raise ValueError("No valid analyzed data found in self.data")

        first_shape = data_list[0].shape
        for i, arr in enumerate(data_list):
            if arr.shape != first_shape:
                raise ValueError(
                    f"Analyzed data shape mismatch: shot 0 has shape {first_shape}, "
                    f"but shot {i} has shape {arr.shape}"
                )

        data_stack = np.stack(data_list, axis=0)
        mean_data = np.nanmean(data_stack, axis=0)
        std_data = np.nanstd(data_stack, axis=0, ddof=ddof)

        return mean_data, std_data

    @staticmethod
    def _resolve_bg_for_row(analyzer, bg, row_dict, debug_bg=False, debug_once=True):
        """
        This function is used when the bg is not the same for every shot in the scan.
        If bg is a function that takes the argument (row_dict), it will return the bg for that row dict.
        You just need to write the function that selects the correct bg for that shot

        bg can be:
        - None
        - already-loaded background (returned as-is)
        - a path/filename (loaded via analyzer.load_data)
        - a callable: bg(row_dict) -> None | loaded_bg | path
        - an object with .get(row_dict) -> None | loaded_bg | path
        """

        if bg is None:
            return None

        # Provider object with .get(row_dict)
        if hasattr(bg, "get") and callable(bg.get):
            bg_spec = bg.get(row_dict)

        # Callable provider
        elif callable(bg):
            bg_spec = bg(row_dict)

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

    def scan_data_mean_std_per_bin(self):
        """
        Compute mean and standard deviation per 'temp Bin number' for all numeric columns.
        Returns
        -------
        mean_df : pandas.DataFrame
            Mean values per bin, including 'temp Bin number' as a column.
        std_df : pandas.DataFrame
            Standard deviation values per bin, including 'temp Bin number' as a column.
        """
        tmp_data = self.data.select_dtypes(include=[np.number, 'float64', 'int64', 'bool'])
        
        mean_df = tmp_data.groupby('temp Bin number').agg(lambda x: np.nanmean(x)).reset_index()
        std_df  = tmp_data.groupby('temp Bin number').agg(lambda x: np.nanstd(x)).reset_index()
        
        return mean_df, std_df


    @staticmethod
    def append_to_add_columns_df(scan, shot, return_dict, add_columns_df):
        data = {'scan': scan, 'Shotnumber': shot}
        data.update(return_dict) 

        if add_columns_df is None:
            add_columns_df = pd.DataFrame([data])
        else:
            add_columns_df = pd.concat([add_columns_df, pd.DataFrame([data])], ignore_index=True)
        return add_columns_df

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

