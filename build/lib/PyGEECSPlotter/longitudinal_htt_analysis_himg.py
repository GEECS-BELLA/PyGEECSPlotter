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


class LongitudinalHTTAnalyzerHimg(WavefrontAnalyzer):
    """
    Derived class to analyze longitudinal images.
    Leverages the improved ImageAnalyzer base class methods,
    including the newly added 'apply_elliptical_mask'.
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
    
    def analyze_data(self, data, analyzer_dict=None, bg=None):
        if analyzer_dict is None:
            analyzer_dict = self.analyzer_dict

        if data is None:
            print("Warning: analyze_data() called with None input — skipping analysis.")
            return None, {}


        if analyzer_dict.get('intensity_only', False):
            data_out = self.intensity_reconstruction(data)

            results = self.compute_data_counts(data_out)

            x, y, x0, y0 = self.get_spatial_coords(
                data_out,
                method=analyzer_dict.get('centroid_method', 'center'),
                x0=analyzer_dict.get('x0', None),
                y0=analyzer_dict.get('y0', None),
                dx=analyzer_dict.get('dx', 1),
                dy=analyzer_dict.get('dy', 1),
                centroid_thresh=analyzer_dict.get('centroid_threshold_low', 0.85)
            )
            x_lo, y_lo = self.get_lineouts(
                data_out,
                x0,
                y0,
                method=analyzer_dict.get('lineout_method', 'center'),
                Nlo=analyzer_dict.get('Nlo', 2)
            )
            results['x0']    = x0
            results['y0']    = y0
            results['x']    = x
            results['y']    = y
            results['x_lo'] = x_lo
            results['y_lo'] = y_lo
            results['imshow_extent'] = self.get_imshow_extent(x,y)

            return data_out, results
            
        else:
            # -----------------------------------------------------
            # Subtract Bg Phase
            # ----------------------------------------------------- 
            slopes = wkpy.SlopesPostProcessor.apply_substractor(data, bg)
            hasodata = wkpy.HasoData(hasoslopes=slopes)
            phase = wkpy.Compute.phase_zonal(compute_phase_set=self.compute_phase_set_zonal, hasodata=hasodata)
            data, pupil = phase.get_data()

            # -----------------------------------------------------
            # Convert to radians
            # ----------------------------------------------------- 
            probe_wl = analyzer_dict.get('probe_wl', 800e-9)
            data = 1e-6 * data * 2 * np.pi / probe_wl

            # -----------------------------------------------------
            # Subtract Polynomial Bg
            # ----------------------------------------------------- 

            centroid_method = analyzer_dict.get('centroid_method', 'manual')
            x0_init = analyzer_dict.get('plasma_x0', None)
            y0_init = analyzer_dict.get('plasma_y0', None)
            dx = analyzer_dict.get('dx', 1)
            dy = analyzer_dict.get('dy', 1)
            centroid_thresh = analyzer_dict.get('centroid_thresh', 0.85)

            x, y, x0_init, y0_init = self.get_spatial_coords(
                data,
                method=centroid_method,
                x0=x0_init,
                y0=y0_init,
                dx=dx,
                dy=dy,
                centroid_thresh=centroid_thresh
            )

            masked_data = self.apply_elliptical_mask(
                            data,
                            x0_init, y0_init,
                            analyzer_dict['plasma_horizontal_pixels'],
                            analyzer_dict['plasma_vertical_pixels'],
                            fill_value=np.nan,
                            invert=True
                        )

            predicted_bg = self.fit_2d_polynomial(
                            masked_data,
                            roi_bounds=None,
                            exclude_nan=True,
                            degree=2
                        )
            data = data - predicted_bg
            masked_data = masked_data - predicted_bg

            # -----------------------------------------------------
            # Calculate more accurate centroid
            # ----------------------------------------------------- 
            if self.lift_on:
                mask_pixels = analyzer_dict.get('mask_pixels', 125)
            else:
                mask_pixels = analyzer_dict.get('mask_pixels', 65)

            data_for_centroid = self.apply_elliptical_mask(
                data,
                x0_init, 
                y0_init, 
                mask_pixels, 
                mask_pixels,
                fill_value=np.nan,
                invert=False
            )

            measured_x0, measured_y0 = self._get_centroid(-data_for_centroid, thresh_min=0.4)
            if  np.isnan(measured_x0) or not np.isnan(measured_y0):
                measured_x0, measured_y0 = x0_init, y0_init

            x, y, x0_init, y0_init = self.get_spatial_coords(
                                        data,
                                        method='manual',
                                        x0=measured_x0,
                                        y0=measured_y0,
                                        dx=dx,
                                        dy=dy,
                                    )

            imshow_extent = self.get_imshow_extent(x,y)

            # -----------------------------------------------------
            # Lineouts
            # -----------------------------------------------------
            Nlo = analyzer_dict.get('Nlo', 2)
            lineout_method = analyzer_dict.get('lineout_method', 'center')
            
            
            x_lo, y_lo = self.get_lineouts(
                data, measured_x0, measured_y0,
                method=lineout_method,
                Nlo=Nlo
            )
            r_lo = self.rotational_average(data, measured_x0, measured_y0, interpolate_nans=True)
            r = np.arange(len(r_lo)) * dx


            # -----------------------------------------------------
            # Phase Shift Metrics
            # -----------------------------------------------------
            phase_shifts = self.compute_phase_shifts(data, shifts_when='')

            fwhm_xpix, hdx, ldx = get_lineout_width(-x_lo, measured_x0,
                                                        from_center=True, width_at=0.5)
            fwhm_ypix, hdy, ldy = get_lineout_width(-y_lo, measured_y0,
                                                        from_center=True, width_at=0.5)

            phase_shifts_bg = self.compute_phase_shifts(masked_data, shifts_when='bg subtracted')

            return_dict = {
                'probe_wl_used' :probe_wl,
                'x': x,
                'y': y,
                'x_lo': x_lo,
                'y_lo': y_lo,
                'r': r,
                'r_lo': r_lo,
                'imshow_extent': imshow_extent,
                'measured_x0': measured_x0,
                'measured_y0': measured_y0,
                'fwhm_x (um)': fwhm_xpix * dx,
                'fwhm_y (um)': fwhm_ypix * dy,
            }

            return_dict = merge_dicts_overwrite(return_dict, phase_shifts, phase_shifts_bg)
            return data, return_dict