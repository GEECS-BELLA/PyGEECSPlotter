# PyGEECS fake s-files to analyze a local data_directory
# Author: Alex Picksley
# Version 0.4
# Created: 2023-02-26
# Last Modified: 2026-02-11

import numpy as np
import os, sys
import matplotlib.pyplot as plt
import re
import glob
import json
import datetime
import pandas as pd
from pathlib import Path

from PyGEECSPlotter.scan_data_analysis import ScanDataAnalyzer
from PyGEECSPlotter.navigation_utils import *
from PyGEECSPlotter.utils import parse_controls_from_python, write_controls_from_python
# from PyGEECSPlotter.navigation_utils import get_analysis_dir, get_analysis_diagnostic_path, open_directory_in_explorer, get_analysed_shot_save_path

import PyGEECSPlotter.plotting as gplt
colors = gplt.configure_plotting()

class DataDirAnalyzer(ScanDataAnalyzer):
    def __init__(self, data_dir, file_ext='.png', scan=1):
        self.data_dir = data_dir
        self.file_ext = file_ext
        self.scan = scan
        self.scan_parameter = 'Shotnumber'

        file_list = sorted(glob.glob(os.path.join(data_dir, f'*{file_ext}')))
        shotnumbers = np.arange(len(file_list)) + 1

        df = pd.DataFrame()
        df['scan'] = np.int64(self.scan * np.ones(len(file_list)))
        df['Shotnumber'] = shotnumbers
        df['diagnostic file_list'] = file_list
        df['diagnostic file_exists'] = np.ones(len(file_list))

        self.data = df

        self.analysis_dir = os.path.join(data_dir, 'analysis')
        os.makedirs(self.analysis_dir, exist_ok=True)
        self.data_out_filename = os.path.join(self.analysis_dir, 'scalar_analysis.txt')

        # Initialise mask after self.data is built
        self._init_mask()


    def __repr__(self):
        return (f"DataDir(data_dir={self.data_dir}, scan={self.scan}, "
                f"num_shots={len(self.data)})")

    def get_scan_data_analysis_dir( self, make_dir=True ):
        return self.analysis_dir
    
    def merge_data_frame_to_sfile(self, 
                                  add_columns_df, 
                                  diagnostic,
                                  overwrite_columns=True, 
                                  analysis_label=None, 
                                  ):
        
        add_columns_df = add_columns_df.select_dtypes(include=[np.number, 'float64', 'int64', 'bool'])
        add_columns_df.to_csv(self.data_out_filename, index=False, sep='\t')
        print(f'Columns added to {self.data_out_filename}')
    