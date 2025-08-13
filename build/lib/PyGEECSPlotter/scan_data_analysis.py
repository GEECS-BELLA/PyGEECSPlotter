# PyGEECS base functions
# Author: Alex Picksley
# Version 0.4
# Created: 2023-02-26
# Last Modified: 2025-07-30

import numpy as np
import os, sys
import re
import glob
import json
import datetime
import pandas as pd
from pathlib import Path

from PyGEECSPlotter.navigation_utils import *
from PyGEECSPlotter.utils import parse_controls_from_python, write_controls_from_python
from PyGEECSPlotter.navigation_utils import get_analysis_dir, get_analysis_diagnostic_path, open_directory_in_explorer

class ScanDataAnalyzer:
    def __init__(self, 
                 sfilename=None, 
                 top_dir=None, 
                 experiment_dir=None,
                 year=None, 
                 month=None, 
                 day=None, 
                 scan=None,
                 diagnostic=None,
                 file_ext=None):
        
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
        self.aborted = False
        self.print_data = False
        
        self.diagnostic = diagnostic
        self.file_ext = file_ext


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
                
            self.add_file_list_to_scan_data(remove_missing_files=remove_missing_diagnostic_files)
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
                    

    def add_file_list_to_scan_data(self, remove_missing_files=True):
        if self.diagnostic is None or self.file_ext is None:
            print('No diagnostic selected')
            return
        
        if os.path.exists( os.path.join(self.top_dir, 'scans', 'Scan%03d' %self.scan, self.diagnostic) ):
            diagnostic_dir = os.path.join(self.top_dir, 'scans', 'Scan%03d' %self.scan, self.diagnostic)
            analysis_diagnostic = False
        elif os.path.exists( os.path.join(self.top_dir, 'analysis', 'Scan%03d' %self.scan, self.diagnostic) ):
            diagnostic_dir = os.path.join(self.top_dir, 'analysis', 'Scan%03d' %self.scan, self.diagnostic)
            analysis_diagnostic = True
        else:
            print('No directory found')

            self.data['%s file_list' %self.diagnostic] = ''
            self.data['%s file_exists' %self.diagnostic] = 0

            if remove_missing_files:
                self.data = self.data[self.data['%s file_exists' %self.diagnostic] != 0].reset_index(drop=True)
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
                diagnostic_dir = os.path.join(self.top_dir, 'analysis', 'Scan%03d' %scan, self.diagnostic)
            else:
                diagnostic_dir = os.path.join(self.top_dir, 'scans', 'Scan%03d' %scan, self.diagnostic)

            basename = r'Scan%03d_%s_%03d%s' %(scan, self.diagnostic, shot_num, self.file_ext)
            filename = os.path.join(diagnostic_dir, basename)

            file_list.append( filename )
            file_exists.append( os.path.exists( filename ) )    

            if not os.path.exists( filename ) and self.print_data:
                print('Did not save : %s' %os.path.basename(filename))

        if self.print_data:
            print('Files to analyse : %d' %len(file_list) )

        self.data['%s file_list' %self.diagnostic] = file_list
        self.data['%s file_exists' %self.diagnostic] = file_exists

        if remove_missing_files:
            self.data = self.data[self.data['%s file_exists' %self.diagnostic] != 0].reset_index(drop=True)
            if self.print_data:
                n_missing = np.sum(1 - np.array(file_exists))
                print('Removed %d lines from scan_data for missing files' %n_missing)

    def filter_scan_data(self, filter_parameter, lower_bound, upper_bound, filter_exclusive=False):
        """
        Filter scan data based on a specified parameter and value range.

        This function filters rows in a DataFrame based on whether the values of a specified parameter fall within (inclusive or exclusive) a given range. Optionally, it can print the count of rows that were included after filtering relative to the total count.

        Parameters:
        - filter_parameter (str): The column name in `scan_data` to apply the filter on.
        - lower_bound (float): The lower bound of the filtering range.
        - upper_bound (float): The upper bound of the filtering range.
        - filter_exclusive (bool, optional): If True, rows with `filter_parameter` values outside the [lower_bound, upper_bound] range are included. If False, only rows with `filter_parameter` values inside this range are included. Defaults to False.
        
        Returns:
        - filtered_scan_data (pd.DataFrame): A DataFrame containing only the rows that meet the filtering criteria.

        The function supports both inclusive and exclusive filtering and provides an option to visualize the filtering impact through printed output.
        """

        if filter_exclusive:
            filter_idcs = (self.data[filter_parameter] < lower_bound) | (self.data[filter_parameter] >  upper_bound)
        else:
            filter_idcs = (self.data[filter_parameter] > lower_bound) & (self.data[filter_parameter] <  upper_bound)

        filtered_scan_data = self.data.loc[filter_idcs].reset_index(drop=True)

        print('%d / %d shots included. Filtered based on : %s ' %(len(filtered_scan_data), len(self.data), get_parameter_alias(filter_parameter)))

        self.data = filtered_scan_data

    def get_bg_file_path(self, which_scan='last'):
        """
        Retrieves the background file path for a specific diagnostic from the analysis directory.

        This function searches for the background file based on the specified scan. It can retrieve
        the first, last, a specific scan's background file, or a file matching a given string.
        The file is expected to have the naming pattern '*<diagnostic>_averaged.png' within the 'analysis' directory.

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
            bg_file_paths = glob.glob(os.path.join(self.top_dir, 'analysis', '*%s_averaged.png' % self.diagnostic))
            
            if not bg_file_paths:
                print(f"No background files found for diagnostic '{self.diagnostic}'.")
                return None

            if which_scan == 'first':
                bg_file_path = bg_file_paths[0]
            elif which_scan == 'last':
                bg_file_path = bg_file_paths[-1]
            elif isinstance(which_scan, int):
                scan_paths = glob.glob(os.path.join(self.top_dir, 'analysis', '*Scan%03d%s_averaged.png' % (which_scan, self.diagnostic)))       
                if len(scan_paths) == 1:
                    bg_file_path = scan_paths[-1]
                else:
                    print(f"No background file found for Scan {which_scan} with diagnostic '{self.diagnostic}'.")
                    return None
            elif isinstance(which_scan, str):
                # Find the file that contains the specified string
                matching_files = [path for path in bg_file_paths if which_scan in os.path.basename(path)]
                if matching_files:
                    bg_file_path = matching_files[0]  # You could return the first match or handle multiple matches differently
                else:
                    print(f"No background file found matching '{which_scan}' for diagnostic '{self.diagnostic}'.")
                    return None
            else:
                print("Invalid value for 'which_scan'. Must be 'first', 'last', an integer, or a string.")
                return None

            if self.print_data:
                print(f"Bg File Path             : {bg_file_path}")
            
            return bg_file_path
        
        except IndexError:
            print(f"No background file found for diagnostic '{self.diagnostic}'.")
            return None
        except Exception as e:
            print(f"An error occurred while retrieving the background file path: {e}")
            return None

    def analyze_scan(self, analyzer, 
        bg=None, 
        display_data=False,
        write_columns_to_sfile=False, 
        overwrite_columns=True, 
        analysis_label=''
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

        for i in range(len(self.data)):
            scan = int(self.data['scan'][i])
            shot_num = int(self.data['Shotnumber'][i])
            filename = self.data['%s file_list' %self.diagnostic][i]    

            data = analyzer.load_data(filename)
            data, return_dict = analyzer.analyze_data(data, bg=bg)

            add_columns_df = ScanDataAnalyzer.append_to_add_columns_df(scan, shot_num, return_dict, add_columns_df)

            if display_data:
                fig, ax = analyzer.display_data(data, title=os.path.basename(filename))
                
        if write_columns_to_sfile and len(self.data) > 0:
            analysis_dir = get_analysis_dir(self.top_dir, self.scan, make_dir=True)
            controls_path = os.path.join(analysis_dir, '%s analyzer_controls %s.txt' % (self.diagnostic, analysis_label) )
            write_controls_from_python(controls_path, analyzer.analyzer_dict)
            
            self.merge_data_frame_to_sfile(add_columns_df, 
                              overwrite_columns=overwrite_columns, 
                              analysis_label=analysis_label, 
                                          )

        return add_columns_df

    def merge_data_frame_to_sfile(self, add_columns_df, 
                                  overwrite_columns=True, 
                                  analysis_label=None, 
                                  ):
        # Load the original data from the sfile
        sfile_data = pd.read_csv(self.sfilename, sep='\t')
        
        # Filter out any non-numeric columns in add_columns_df
        add_columns_df = add_columns_df.select_dtypes(include=['float64', 'int64', 'bool'])

        # Apply renaming to add_columns_df, excluding 'scan' and 'Shotnumber'
        if self.diagnostic or analysis_label:
            def rename_columns(col):
                if col in ['scan', 'Shotnumber']:
                    return col  # Do not rename 'scan' or 'Shotnumber'
                new_col = col
                if self.diagnostic:
                    new_col = f"{self.diagnostic} {new_col}"
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
        add_columns_path = os.path.join(self.analysis_dir, f"Scan{int(self.scan):03d}_{self.diagnostic}_{analysis_label}_Summary.txt")
        add_columns_df.to_csv( add_columns_path , index=False, sep='\t' )
        merged_df.to_csv(self.sfilename, index=False, sep='\t')
        print(f'Columns added to {self.sfilename}')

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
