# PyGEECS base functions
# Author: Alex Picksley
# Version 0.4
# Created: 2023-02-26
# Last Modified: 2025-07-30

import numpy as np
import os, sys
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from PyGEECSPlotter.scan_data_analysis import ScanDataAnalyzer

class ScanDataOverview(ScanDataAnalyzer):
    def __init__(self, 
                sfilename=None, 
                top_dir=None, 
                experiment_dir=None,
                year=None, 
                month=None, 
                day=None, 
                scan=None
                ):
        super().__init__(
            sfilename=sfilename,
            top_dir=top_dir,
            experiment_dir=experiment_dir,
            year=year,
            month=month,
            day=day,
            scan=scan
        )
        

    def analyze_scan(self, analyzer, bg=None, bins=None):
        if bins is None:
            bins = np.unique( self.active_data['temp Bin number'] )
        mean_per_bin = [None] * len(bins)
        std_per_bin = [None] * len(bins)

        saved_mask = self.save_mask()
        data_shape = None

        for i, binn in enumerate(bins):
            self.filter_scan_data('temp Bin number', binn - 0.1, binn + 0.1)
            mean_data, std_data = self.mean_std_diagnostic(
                analyzer,
                bg=bg,
                ddof=0,
            )
            self.restore_mask(saved_mask)

            if mean_data is None or std_data is None:
                continue

            data_shape = mean_data.shape
            mean_per_bin[i] = mean_data
            std_per_bin[i] = std_data

        if data_shape is None:
            return None, None

        nan_fill = np.full(data_shape, np.nan)
        mean_per_bin = [x if x is not None else nan_fill for x in mean_per_bin]
        std_per_bin  = [x if x is not None else nan_fill for x in std_per_bin]

        return np.asarray(mean_per_bin), np.asarray(std_per_bin)