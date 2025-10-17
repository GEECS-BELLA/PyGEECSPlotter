# Dervied class WavefrontAnalyzer from ImageAnalyzer for PyGEECSPlotter
# Author: Alex Picksley
# Version 0.4
# Created: 2024-02-26
# Last Modified: 2025-02-19

import numpy as np
from typing import Optional, Dict, Tuple  
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
from scipy.ndimage import affine_transform
from scipy.optimize import least_squares
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

import sys, os
sys.path.append('./../..')
import wavekit_py as wkpy
import time 
import ctypes

# import PyGEECSPlotter.main as pygc
import PyGEECSPlotter.ni_imread as ni_imread

from PyGEECSPlotter.image_analysis import ImageAnalyzer

class WavefrontAnalyzer(ImageAnalyzer):
    """
    Base class for image analysis. 
    Derive from this class to customize analysis steps for specific image types.
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
    ):
        # Initialize all base-class attributes
        super().__init__(
            diagnostic=diagnostic,
            file_ext=file_ext,
            analyzer_dict=analyzer_dict,
            display_dict=display_dict,
            output_diagnostic=output_diagnostic,
            output_file_ext=output_file_ext,
        )

        # Store wavefront config even if we don't init the engine yet
        self.config_file_path = config_file_path
        self.start_subpupil_size = start_subpupil_size
        self.denoising_strength = denoising_strength

        # Lazily/optionally initialize Haso engine + compute sets
        self.hasoengine = None
        self.compute_phase_set_zernike = None
        self.compute_phase_set_zonal = None
        self.pupil = None
        self.pupil_dict = None

        if config_file_path is not None:
            self.hasoengine = wkpy.HasoEngine(config_file_path=config_file_path)
            self.hasoengine.set_preferences(
                wkpy.uint2D(*start_subpupil_size),
                denoising_strength,
                False,
            )

            self.compute_phase_set_zernike = wkpy.ComputePhaseSet(
                type_phase=wkpy.E_COMPUTEPHASESET.MODAL_ZERNIKE
            )
            self.compute_phase_set_zonal = wkpy.ComputePhaseSet(
                type_phase=wkpy.E_COMPUTEPHASESET.ZONAL
            )
            # Tweak as in your original
            self.compute_phase_set_zonal.set_zonal_prefs(10, 1000, 1e-5)


    # -------------------------------------------------------------------
    # Public pipeline method
    # -------------------------------------------------------------------
    def load_data(self, filename):
        file_ext = os.path.splitext(filename)[-1]
        if 'himg' in file_ext:
            image = wkpy.Image(image_file_path = filename)
            hasoslopes = self.hasoengine.compute_slopes(
                image,
                False
                )[1]
            return hasoslopes
        elif 'has' in file_ext:
            return wkpy.HasoSlopes(has_file_path = filename)
        elif 'png' in file_ext:
            data = ni_imread.read_imaq_image('%s' % filename)

            with open('%s.txt' % filename[:-4]) as f:
                lines = f.readlines()
            scale_min = float(lines[1].split(' ')[2])
            scale_max = float(lines[2].split(' ')[2])

            return data * (scale_max - scale_min) / (2**16 - 1) + scale_min
        else:
            return None

    def load_raw_data(self, filename):
        file_ext = os.path.splitext(filename)[-1]
        if 'himg' in file_ext:
            image = wkpy.Image(image_file_path = filename)
            return image.get_data()
        else:
            return None


    def analyze_data(self, data, analyzer_dict={}, bg=None):
        if analyzer_dict.get('bg_file', False) and bg is not None:
            data -= bg

        if analyzer_dict.get('set_max_to_nan', True):
            data[data == np.nanmax(data)] = np.nan

        if analyzer_dict.get('filter_tilts_and_curv', False):
            fitted_tilt_curv = self.fit_2d_polynomial(data, 
                                                      roi_bounds=analyzer_dict.get('tilts_and_curv_roi', None), 
                                                      exclude_nan=True)

        results = self.compute_phase_shifts(data, shifts_when='')

        if analyzer_dict.get('outlier_threshold', None) is not None:
            data = self.remove_outliers(data, threshold=3)
            results.update( self.compute_phase_shifts(data, shifts_when='outliers removed') )

        return data, results
        
    # def write_analyzed_data(self, save_path, data):
    #     pygc.write_binary(data, save_path)
    

    # -------------------------------------------------------------------
    # Utility methods (can be overridden or extended in subclasses)
    # -------------------------------------------------------------------

    @staticmethod
    def compute_phase_shifts(data, shifts_when=''):
        max_data = np.nanmax(data)
        min_data = np.nanmin(data)
        ptp = np.nanmax(data) - np.nanmin(data)
        rms = np.nanstd(data)

        return {f'max shift {shifts_when}': max_data,
                f'min shift {shifts_when}': min_data,
                f'ptp shift {shifts_when}': ptp,
                f'rms shift {shifts_when}': rms,
               }