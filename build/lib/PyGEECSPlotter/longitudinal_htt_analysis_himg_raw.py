import numpy as np
import sys, os
from typing import Optional, Dict, Tuple 


import sys, os
sys.path.append('./../..')
import wavekit_py as wkpy
import time 
import ctypes

from PyGEECSPlotter.wavefront_analysis import WavefrontAnalyzer
from PyGEECSPlotter.utils import super_gaussian, merge_dicts_overwrite, get_lineout_width
import imageio as imio


class LongitudinalHTTAnalyzerHimgRaw(WavefrontAnalyzer):
    """
    Derived class to return the raw data from a HASO camera when the .himg file has been saved
    """
    
    def __init__(
        self,
        # >>> base-class params
        diagnostic: Optional[str] = None,
        file_ext: Optional[str] = None,
        analyzer_dict: Optional[Dict] = None,
        display_dict: Optional[Dict] = None,
        output_diagnostic: Optional[str] = None,
        output_file_ext: Optional[str] = None,
        # >>> wavefront-specific params
        *,
        config_file_path: Optional[str] = None,
        start_subpupil_size: Tuple[int, int] = (20, 20),
        denoising_strength: float = 0.0,
        lift_on=False,
    ):
        # Initialize WavefrontAnalyzer (which initializes ImageAnalyzer)
        super().__init__(
            diagnostic=diagnostic,
            file_ext=file_ext,
            analyzer_dict=analyzer_dict,
            display_dict=display_dict,
            output_diagnostic=output_diagnostic,
            output_file_ext=output_file_ext,
            config_file_path=config_file_path,
            start_subpupil_size=start_subpupil_size,
            denoising_strength=denoising_strength,
            lift_on=lift_on,
        )

    def load_data(self, filename):
        return self.load_raw_data(filename)

    def analyze_data(self, data, analyzer_dict=None, bg=None):
        if analyzer_dict is None:
            analyzer_dict = self.analyzer_dict

        if data is None:
            print("Warning: analyze_data() called with None input — skipping analysis.")
            return None, {}

        return_dict = self.compute_data_counts(data)
        return data, return_dict 

    def write_analyzed_data(self, filepath, data):
        imio.imwrite(filepath, data.astype('uint8'))