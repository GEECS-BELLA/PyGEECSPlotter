# Dervied class WindmillWave from WavefrontAnalyzer, ImageAnalyzer for PyGEECSPlotter
# Author: Alex Picksley
# Version 0.1
# Created: 2025-12-02
# Last Modified: 2025-12-02

import numpy as np
import sys, os
from typing import Optional, Dict, Tuple 


import sys, os
sys.path.append('./../..')
import wavekit_py as wkpy
import time 
import ctypes

from PyGEECSPlotter.wavefront_analysis import WavefrontAnalyzer
from PyGEECSPlotter.utils import super_gaussian, merge_dicts_overwrite, get_lineout_width, gaussian_2d, fit_gaussian_2d
from PyGEECSPlotter.navigation_utils import get_analysed_shot_save_path

class WindmillWave(WavefrontAnalyzer):
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
        start_subpupil: Tuple[int, int] = (20, 20),
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
            start_subpupil=start_subpupil,
            denoising_strength=denoising_strength,
            lift_on=lift_on,
        )


    def analyze_data(self, data, analyzer_dict=None, row_dict=None, bg=None):
        """
        Robust wavefront pipeline (WindmillWave-style error handling).
    
        Returns:
            (zonal_data, return_dict, lineouts) on success
            (None, {}, {}) on any known WaveKit failure mode (e.g. bad pupil / all subapertures off),
            or when inputs are invalid.
        """
        if analyzer_dict is None:
            analyzer_dict = self.analyzer_dict

        data_out = {}

        filename = row_dict[f'{self.diagnostic} file_list']
        raw_data = self.load_raw_data(filename)
        raw_stats_dict = self.compute_data_counts(raw_data) 
        data_out["raw_data"] = raw_data
    
        # Match WindmillWave signature
        lineouts = {}
    
        if data is None:
            print("Warning: analyze_data() called with None input — skipping analysis.")
            return None, {}, {}
    
        # ------------------------------------------------------------
        # 1) Optional slopes subtraction (safe)
        # ------------------------------------------------------------
        hasoslopes = data
        if bg is not None:
            try:
                hasoslopes = wkpy.SlopesPostProcessor.apply_substractor(hasoslopes, bg)
            except Exception:
                # If reference subtraction fails, treat as bad shot
                return None, {}, {}
    
        # ------------------------------------------------------------
        # 2) Build HasoData (safe)
        # ------------------------------------------------------------
        try:
            hasodata = wkpy.HasoData(hasoslopes=hasoslopes)
        except Exception:
            return None, {}, {}
    
        # ------------------------------------------------------------
        # 3) Pupil selection (safe-ish)
        # ------------------------------------------------------------
        pupil_method = analyzer_dict.get("pupil_method", "auto")
    
        try:
            if pupil_method == "auto":
                self.pupil, self.pupil_dict = self.get_pupil(hasoslopes)
    
            elif pupil_method == "auto_first_shot":
                if self.pupil is None or self.pupil_dict is None:
                    self.pupil, self.pupil_dict = self.get_pupil(hasoslopes)
    
            elif pupil_method == "manual":
                if "manual_pupil_dict" in analyzer_dict:
                    self.pupil_dict = analyzer_dict["manual_pupil_dict"]
                    self.pupil = None
                else:
                    print("[WARNING] 'manual' pupil method selected but no 'manual_pupil_dict' provided. Defaulting to 'auto'.")
                    self.pupil, self.pupil_dict = self.get_pupil(hasoslopes)
    
            else:
                print(f"[WARNING] Invalid pupil_method '{pupil_method}'. Defaulting to 'auto'.")
                self.pupil, self.pupil_dict = self.get_pupil(hasoslopes)
    
        except Exception as e:
            if self._is_wavekit_recoverable_exception(e):
                return None, {}, {}
            raise
    
        if self.pupil_dict is None:
            return None, {}, {}
    
        pupil_center = wkpy.float2D(self.pupil_dict["pupil_center_x"], self.pupil_dict["pupil_center_y"])
        pupil_radius = self.pupil_dict["pupil_radius"]
    
        # ------------------------------------------------------------
        # 4) Aberration filter selection (same logic as your current code)
        # ------------------------------------------------------------
        if analyzer_dict.get("filter_tilts_and_curv", False):
            phasemap_aberration_filter = [0, 0, 0, 1, 1]
        else:
            phasemap_aberration_filter = [1, 1, 1, 1, 1]
    
        # ------------------------------------------------------------
        # 5) ZONAL reconstruction (HARD GATE)
        #    If this fails with known pupil errors, we stop and return None.
        # ------------------------------------------------------------
        try:
            zonal_data, zonal_pupil, zonal_phase_statistics = self.zonal_reconstruction(
                hasodata,
                phasemap_aberration_filter=phasemap_aberration_filter[:-2],
                nan_to_zero=analyzer_dict.get("set_nan_to_zero", False),
            )
            data_out["zonal_data"] = zonal_data
        except Exception as e:
            if self._is_wavekit_recoverable_exception(e):
                return None, {}, {}
            raise
    
        if zonal_data is None:
            return None, {}, {}
    
        # ------------------------------------------------------------
        # 6) ZERNIKE reconstruction (OPTIONAL / best-effort)
        #    Only attempted if zonal succeeded.
        # ------------------------------------------------------------
        zernike_dict = {}
        zernike_phase_statistics = {}
        try:
            _zernike_data, _zernike_pupil, zernike_dict, zernike_phase_statistics = self.zernike_reconstruction(
                hasoslopes,
                hasodata,
                pupil_center,
                pupil_radius,
                nb_modes=analyzer_dict.get("nb_modes", 32),
                coefs_to_filter=analyzer_dict.get("zernike_coefs_to_filter", []),
                phasemap_aberration_filter=phasemap_aberration_filter,
                nan_to_zero=analyzer_dict.get("set_nan_to_zero", False),
            )
        except Exception as e:
            if not self._is_wavekit_recoverable_exception(e):
                raise
    
        # ------------------------------------------------------------
        # 7) Downstream metrics (safe)
        # ------------------------------------------------------------
        try:
            geometric_properties = WindmillWave.slopes_geometric_properties(hasoslopes)
        except Exception:
            geometric_properties = {}
    
        try:
            results = self.compute_phase_shifts(zonal_data, shifts_when="")
        except Exception:
            results = {}

        # ------------------------------------------------------------
        # 8) Reconstruct intensity
        # ------------------------------------------------------------
    
        if analyzer_dict.get("reconstruct_intensity", True):
            try:
                intensity = WindmillWave.intensity_reconstruction(hasoslopes)
                data_out["intensity"] = intensity
            except Exception:
                pass

        # ------------------------------------------------------------
        # 9) Windmill laser pupil
        # ------------------------------------------------------------

        windmill_laser_pupil = None
        try:
            windmill_laser_pupil = self.apply_elliptical_mask(
                zonal_data,
                pupil_center.X, 
                pupil_center.Y, 
                pupil_radius, 
                pupil_radius,
                fill_value=np.nan,
                invert=False
            )
    
            fitted_tilt_curv = self.fit_2d_polynomial(windmill_laser_pupil, exclude_nan=True)
            windmill_laser_pupil = windmill_laser_pupil - fitted_tilt_curv
            windmill_laser_pupil = windmill_laser_pupil - np.nanmean(windmill_laser_pupil)
            
        except Exception:
            pass

        data_out["windmill_laser_pupil"] = windmill_laser_pupil
        
        x, y, _, _ = self.get_spatial_coords(
            zonal_data,
            method='manual',
            x0=pupil_center.X,
            y0=pupil_center.Y,
            dx=analyzer_dict.get('dx', 1),
            dy=analyzer_dict.get('dx', 1),
        )
        
        imshow_extent = self.get_imshow_extent(x,y)
        plot_dict = {'imshow_extent' : imshow_extent}

        # ------------------------------------------------------------
        # 10) Merge return_dict (same pattern as your current code)
        # ------------------------------------------------------------

        return_dict = merge_dicts_overwrite(
            self.pupil_dict,
            raw_stats_dict,
            zernike_dict,
            zernike_phase_statistics,
            geometric_properties,
            zonal_phase_statistics,
            results,
            plot_dict,
        )

        
        return data_out, return_dict, lineouts

    def display_data(self, data, display_dict=None, return_dict=None, title=None, fig=None, ax=None):
        if display_dict is None:
            display_dict = self.display_dict

        if fig is None or ax is None:
            fig, (ax_phi, ax_I, ax_wm) = plt.subplots(1, 3, constrained_layout=True, 
                                               figsize=display_dict.get('figsize', (16, 4)))
         
        fig, ax_phi = super().display_data( data['zonal_data'], 
                                          display_dict=display_dict['zonal_data'], 
                                          return_dict=return_dict,
                                          title=f'{title} Zonal', 
                                          fig=fig, ax=ax_phi
                                         )

        fig, ax_I = super().display_data( data['intensity'], 
                                          display_dict=display_dict['intensity'], 
                                          return_dict=return_dict,
                                          title=f'{title} Intensity', 
                                          fig=fig, ax=ax_I
                                         )

        fig, ax_wm = super().display_data( data['windmill_laser_pupil'], 
                                          display_dict=display_dict['windmill_laser_pupil'], 
                                          return_dict=return_dict,
                                          title=f'{title} Central 3 mm', 
                                          fig=fig, ax=ax_wm
                                         )

        return fig, (ax_phi, ax_I, ax_wm)

    def write_analyzed_data(self, data, analysis_dir, scan, shot_num):

        # ---- raw ----
        append_info = '_raw'
        save_path = get_analysed_shot_save_path(
            analysis_dir,
            f'{self.output_diagnostic}{append_info}',
            scan,
            shot_num,
            self.output_file_ext
        )
        super().write_analyzed_data(save_path, data['raw_data'])

        # ---- intensity ----
        append_info = '_intensity'
        save_path = get_analysed_shot_save_path(
            analysis_dir,
            f'{self.output_diagnostic}{append_info}',
            scan,
            shot_num,
            self.output_file_ext,
        )
        super().write_analyzed_data(save_path, data['intensity'])

        # ---- zonal ----
        append_info = '_zonal_data'
        save_path = get_analysed_shot_save_path(
            analysis_dir,
            f'{self.output_diagnostic}{append_info}',
            scan,
            shot_num,
            self.output_file_ext,
        )
        super().write_analyzed_data(save_path, data['zonal_data'])

        # ---- pupil (optional) ----
        if data.get('windmill_laser_pupil') is not None:
            append_info = '_windmill_laser_pupil'
            save_path = get_analysed_shot_save_path(
                analysis_dir,
                f'{self.output_diagnostic}{append_info}',
                scan,
                shot_num,
                self.output_file_ext,
            )
            super().write_analyzed_data(save_path, data['windmill_laser_pupil'])

        

    