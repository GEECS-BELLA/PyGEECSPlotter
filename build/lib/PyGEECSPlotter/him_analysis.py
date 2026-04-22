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
import imageio.v3 as imio


from PyGEECSPlotter.wavefront_analysis import WavefrontAnalyzer
from PyGEECSPlotter.utils import find_matching_row_index
from PyGEECSPlotter.navigation_utils import get_analysed_shot_save_path

def make_bg_selector(bg_mean_df, bg_dir, match_cols, tolerances=None, default_tol=1e-3, print_path=True):
    """
    Returns a function bg_selector(row_dict) -> bg_filename or None
    """
    def bg_selector(row_dict):
        bg_idx = find_matching_row_index(
            bg_mean_df,
            match_cols,
            row_dict,
            tolerances=tolerances,
            default_tol=default_tol,
        )
        if bg_idx is None or len(bg_idx) == 0:
            return None

        return os.path.join(bg_dir, f"averaged_reference_{int(bg_idx[0]):03d}.has")

    return bg_selector

class HimgAnalyzer(WavefrontAnalyzer):
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
        # Aberration filter selection (same logic as your current code)
        # ------------------------------------------------------------
        if analyzer_dict.get("filter_tilts_and_curv", False):
            phasemap_aberration_filter = [0, 0, 0, 1, 1]
        else:
            phasemap_aberration_filter = [1, 1, 1, 1, 1]
    
        # ------------------------------------------------------------
        # ZONAL reconstruction 
        #    If this fails with known pupil errors, we stop and return None.
        # ------------------------------------------------------------
        try:
            zonal_data_um, zonal_pupil, zonal_phase_statistics = self.zonal_reconstruction(
                hasodata,
                phasemap_aberration_filter=phasemap_aberration_filter[:-2],
                nan_to_zero=analyzer_dict.get("set_nan_to_zero", False),
            )
            zonal_data_rad = 1e-6 * zonal_data_um * 2 * np.pi / analyzer_dict['probe_wl'] 
            data_out['zonal_data_rad'] = zonal_data_rad

        except Exception as e:
            if self._is_wavekit_recoverable_exception(e):
                return None, {}, {}
            raise
    
        if zonal_data_um is None:
            return None, {}, {}

        # ------------------------------------------------------------
        # Reconstruct intensity
        # ------------------------------------------------------------
    
        if analyzer_dict.get("reconstruct_intensity", True):
            try:
                intensity = self.intensity_reconstruction(hasoslopes)
                data_out["intensity"] = intensity
            except Exception:
                pass

        
        return data_out, raw_stats_dict, {}

    def write_analyzed_data(self, data, analysis_dir, scan, shot_num, context_dict=None):

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
        append_info = '_zonal_data_rad'
        save_path = get_analysed_shot_save_path(
            analysis_dir,
            f'{self.output_diagnostic}{append_info}',
            scan,
            shot_num,
            self.output_file_ext,
        )
        super().write_analyzed_data(save_path, data['zonal_data_rad'])

        append_info = '_readable'
        save_path = get_analysed_shot_save_path(
            analysis_dir,
            f'{self.output_diagnostic}{append_info}',
            scan,
            shot_num,
            self.output_file_ext,
        )
        stack = np.stack((data['zonal_data_rad'], data['intensity']), axis=0, dtype=np.float32)
        imio.imwrite(save_path, stack, metadata=context_dict)
    

    def display_data(self, data, display_dict=None, return_dict=None, title=None, fig=None, ax=None):
        if display_dict is None:
            display_dict = self.display_dict

        if fig is None or ax is None:
            fig, (ax_phi, ax_I) = plt.subplots(1, 2, constrained_layout=True, 
                                               figsize=display_dict.get('figsize', (11, 4)))
         
        fig, ax_phi = super().display_data( data['zonal_data_rad'], 
                                          display_dict=display_dict['zonal_data_rad'], 
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

        return fig, (ax_phi, ax_I)
