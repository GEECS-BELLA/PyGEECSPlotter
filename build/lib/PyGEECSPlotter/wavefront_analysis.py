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
import imageio as imio
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
        lift_on=False,
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
        self.lift_on = lift_on

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

            # Set lift option
            if self.lift_on:
                self.hasoengine.set_lift_option(self.lift_on, self.analyzer_dict['probe_wl']*1e9)

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
            try:
                image = wkpy.Image(image_file_path=filename)
                hasoslopes = self.hasoengine.compute_slopes(image, False)[1]
                return hasoslopes
            except Exception as e:
                print(f"Failed to compute slopes for {filename}: {e}")
                return None
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
        if 'himg' in self.file_ext:
            if bg is not None:
                slopes = wkpy.SlopesPostProcessor.apply_substractor(data, bg)
            else: 
                slopes = data
            
            hasodata = wkpy.HasoData(hasoslopes=slopes)
            phase = wkpy.Compute.phase_zonal(compute_phase_set=self.compute_phase_set_zonal, hasodata=hasodata)
            data, pupil = phase.get_data()

            results = self.compute_phase_shifts(data, shifts_when='')
            


        else:
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
        
    def write_analyzed_data(self, bin_filepath, data, nan_value=0):
        """
        Write a 2D NumPy array to a binary PNG image and create a scaling information text file.
        
        Parameters:
        - bin_filepath (str): The base file path for the PNG image and scaling text file.
        - data (numpy.ndarray): The input 2D array to be saved.
        - nan_value (float, optional): The scalar value to replace NaN values with. Default is 0.
        
        Returns:
        - None
        
        The function scales the input data to fit within a 16-bit range (0-65535) and saves it as a binary PNG image. 
        The scaling factors (min and max) used for scaling are saved in a text file with the same name.
        """
        
        np.save(bin_filepath + '.npy', data, allow_pickle=True, fix_imports=True)

        # Replace NaN values with the specified scalar
        data = np.nan_to_num(data, nan=nan_value)
    
        # Calculate the scaling factors
        data_int = (65535 * ((data - np.min(data)) / np.ptp(data))).astype(np.uint16)
    
        # Create scaling information lines
        lines = ['[Scaling]', 'min = %f' % np.min(data), 'max = %f' % np.max(data)]
    
        # Write scaling information to a text file
        with open(bin_filepath + '.txt', 'w') as f:
            f.write('\n'.join(lines))
    
        # Save the scaled data as a binary PNG image
        imio.imwrite(bin_filepath + '.png', data_int)
    

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
    @staticmethod
    def intensity_reconstruction(hasoslopes):
        return hasoslopes.get_intensity()

    @staticmethod
    def subtract_slopesdata(data1, data2):
        return wkpy.SlopesPostProcessor.apply_substractor(data1, data2)