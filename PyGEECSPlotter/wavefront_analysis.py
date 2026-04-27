# Dervied class WavefrontAnalyzer from ImageAnalyzer for PyGEECSPlotter
# Author: Alex Picksley
# Version 0.4
# Created: 2024-02-26
# Last Modified: 2025-02-19

import numpy as np
from typing import Optional, Dict, Tuple  
import pandas as pd
import matplotlib.pyplot as plt
import imageio as imio
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

import sys, os
sys.path.append(r'N:\Software\Installs and Manuals\Imagine Optic\wavekit_dlls')
import wavekit_py as wkpy
import time 
import ctypes

import PyGEECSPlotter.ni_imread as ni_imread
from PyGEECSPlotter.navigation_utils import get_analysed_shot_save_path


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
        start_subpupil: Tuple[int, int] = (20, 20),
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
        self.start_subpupil = start_subpupil
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
                wkpy.uint2D(*start_subpupil),
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
        if not os.path.exists( filename ):
            return None

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
        elif 'npy' in file_ext:
            return np.load( filename )
        else: 
            return None

    def load_raw_data(self, filename):
        file_ext = os.path.splitext(filename)[-1]
        if 'himg' in file_ext:
            image = wkpy.Image(image_file_path = filename)
            return image.get_data()
        else:
            return None


    def analyze_data(self, data, analyzer_dict=None, row_dict={}, bg=None):
        if analyzer_dict is None:
            analyzer_dict = self.analyzer_dict

        if data is None:
            print("Warning: analyze_data() called with None input — skipping analysis.")
            return None, {}, {}

        if 'himg' in self.file_ext or 'has' in self.file_ext:
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

        return data, results, {}

    def average_file_list(self, file_list):
        slopes_list = []
        for filename in file_list:
            data = self.load_data(filename)
            if data is not None:
                slopes_list.append(data)
            else:
                print(f"Skipping {filename} (load failed)")

        # Check that we have at least one valid slope
        if not slopes_list:
            print("No valid slopes found. Returning None.")
            return None

        # Initialize average with the first valid slope
        avg_slopes = slopes_list[0]

        # Add the rest
        for slopes in slopes_list[1:]:
            avg_slopes = wkpy.SlopesPostProcessor.apply_adder(avg_slopes, slopes)

        # Scale by the number of valid slopes
        n_valid = len(slopes_list)
        avg_slopes = wkpy.SlopesPostProcessor.apply_scaler(avg_slopes, 1 / n_valid)

        return avg_slopes


    def average_scan_data(self, scan_data):
        file_list = list(scan_data.data[f'{self.diagnostic} file_list'])
        return self.average_file_list(file_list)
    
    def write_analyzed_data(self, bin_filepath, data, nan_value=0):
        #### THIS NEEDS TO BE FIXED!!! NOT STABLE !!!
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
        
        np.save(bin_filepath + '.npy', data, allow_pickle=True)

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
    def subtract_slopes_reference(data, bg=None):
        try:
            return wkpy.SlopesPostProcessor.apply_substractor(data, bg) if bg is not None else data
        except Exception:
            return None

    @staticmethod
    def _is_wavekit_recoverable_exception(exc: Exception) -> bool:
        """
        Classify WaveKit exceptions that should NOT crash analysis.
        We treat these as 'bad shot / insufficient data' and skip gracefully.
        """
        msg = str(exc)
        # WaveKit sometimes nests exceptions like: Exception('io_wavekit_compute : ', Exception('IO_Error', b'...'))
        # so we match substrings that appear in either the outer or inner string repr.
        patterns = [
            "bad_pupil_error",
            "All subapertures are off",
            "modal_projection_error",
            "Not enough lit subapertures",
            "not enough lit subapertures",
            "IO_Error",
        ]
        return any(p in msg for p in patterns)

    # -------------------------------------------------------------------
    # Utility methods (can be overridden or extended in subclasses)
    # -------------------------------------------------------------------

    def zernike_reconstruction(self, 
                               hasoslopes, 
                               hasodata,
                               pupil_center,
                               pupil_radius,
                               nb_modes=32,
                               coefs_to_filter=[],
                               phasemap_aberration_filter=[1,1,1,1,1],
                               nan_to_zero=False
                              ):
        
        modal_coef = wkpy.ModalCoef(modal_type = wkpy.E_MODAL.ZERNIKE)
        modal_coef.set_zernike_prefs(
            wkpy.E_ZERNIKE_NORM.STD,
            nb_modes,
            coefs_to_filter,
            wkpy.ZernikePupil_t(
                pupil_center,
                pupil_radius
                ),
        )

        wkpy.Compute.coef_from_hasodata(self.compute_phase_set_zernike, hasodata, modal_coef)

        size = modal_coef.get_dim()
        data_coeffs, data_indexes, pupil = modal_coef.get_data()

        phase = wkpy.Phase(
                hasoslopes=hasoslopes,  # Pass the HasoSlopes object
                type_=2,                # Set type_ to 2 for Zernike reconstruction
                filter_=phasemap_aberration_filter,        # Aberration filter list
                nb_coeffs=nb_modes     # Number of Zernike coefficients
            )
        data, pupil = phase.get_data()
        if nan_to_zero:
            data[np.isnan(data)] = 0.0

        zernike_dict = {f'zernike_{index}': coeff for index, coeff in zip(data_indexes, data_coeffs)}

        names = ['tilt_0', 'tilt_90', 'focus', 'astig_0', 'astig_45', 'coma_0', 'coma_90', 'spherical', 'trefoil_0', 'trefoil_90',
                '5th_astig_0', '5th_astig_45', '5th_coma_0', '5th_coma_90', '5th_spherical', 'tetrafoil_0', 'tetrafoil_45']
        named_indexes = list(range(1, len(names) + 1))

        zernike_dict = {
            (f'zernike_{i:02d}_{names[i - 1]} (um)' if i in named_indexes else f'zernike_{i:02d} (um)'): coeff
            for i, coeff in zip(data_indexes, data_coeffs)
        }

        zernike_phase_statistics = WavefrontAnalyzer.compute_phase_shifts( data, shifts_when='zernike' )
        
        return data, pupil, zernike_dict, zernike_phase_statistics

    def zonal_reconstruction(self, hasodata, phasemap_aberration_filter=[1, 1, 1], nan_to_zero=False):
        self.compute_phase_set_zonal.set_zonal_filter(phasemap_aberration_filter)
        phase = wkpy.Compute.phase_zonal(compute_phase_set=self.compute_phase_set_zonal, hasodata=hasodata)
        data, pupil = phase.get_data()
        if nan_to_zero:
            data[np.isnan(data)] = 0.0
        zonal_phase_statistics = {
            'zonal_phase_rms (um)' : phase.get_statistics().rms,
            'zonal_phase_pv (um)' : phase.get_statistics().pv,
            'zonal_phase_max (um)' : phase.get_statistics().max,
            'zonal_phase_min (um)' : phase.get_statistics().min,
        }
        return data, pupil, zonal_phase_statistics
    
    @staticmethod
    def get_pupil(hasoslopes):
        pupil = wkpy.Pupil(hasoslopes = hasoslopes)
        center, radius = wkpy.ComputePupil.fit_zernike_pupil(
            pupil,
            wkpy.E_PUPIL_DETECTION.AUTOMATIC,
            wkpy.E_PUPIL_COVERING.INSCRIBED,
            False)

        pupil_dict = {
            'pupil_center_x' : center.X,
            'pupil_center_y' : center.Y,
            'pupil_radius' : radius,
        }
        
        return pupil, pupil_dict

    @staticmethod
    def slopes_geometric_properties(hasoslopes):
        properties = hasoslopes.get_geometric_properties()
        geometric_properties = {
            'geometric_tilt_x (mrad)' : properties[0], 
            'geometric_tilt_y (mrad)': properties[1],
            'geometric_radius (mm)' : properties[2],
            'geometric_focus_x_pos (mm)' : properties[3],
            'geometric_focus_y_pos (mm)' : properties[4],
            'geometric_astig_angle (rad)' : properties[5],
            'geometric_sagittal (mm)' : properties[6],
            'geometric_tangential (mm)' : properties[7],
            'geometric_curvature (per m)' : 1.0/(properties[2]*1e-3)
        }
        return geometric_properties


    


