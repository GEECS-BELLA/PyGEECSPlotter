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

        zernike_phase_statistics = {
            'zernike_phase_rms (um)' : phase.get_statistics().rms,
            'zernike_phase_pv (um)' : phase.get_statistics().pv,
            'zernike_phase_max (um)' : phase.get_statistics().max,
            'zernike_phase_min (um)' : phase.get_statistics().min,
        }
        
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

    @staticmethod
    def intensity_reconstruction(hasoslopes):
        return hasoslopes.get_intensity()

    


