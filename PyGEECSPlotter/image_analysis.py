# Base class ImageAnalyzer for PyGEECSPlotter
# Author: Alex Picksley
# Version 0.4
# Created: 2024-02-26
# Last Modified: 2025-02-19

import os, sys
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple  
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import medfilt2d
from scipy.ndimage import affine_transform
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import imageio as imio

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

import PyGEECSPlotter.ni_imread as ni_imread
from PyGEECSPlotter.utils import super_gaussian, get_lineout_width, save_lineouts_to_txt
from PyGEECSPlotter.navigation_utils import get_analysed_shot_save_path
from PyGEECSPlotter.diagnostic_analyzer import DiagnosticAnalyzer


class ImageAnalyzer(DiagnosticAnalyzer):
    """
    Base class for image analysis.
    Derive from this class to customize analysis steps for specific image types.
    """

    def __init__(self,
                 diagnostic=None,
                 file_ext=None,
                 analyzer_dict=None,
                 display_dict=None,
                 output_diagnostic=None,
                 output_file_ext=None,
                ):
        super().__init__(
            diagnostic=diagnostic,
            file_ext=file_ext,
            analyzer_dict=analyzer_dict,
            display_dict=display_dict,
            output_diagnostic=output_diagnostic,
            output_file_ext=output_file_ext,
        )

    # -------------------------------------------------------------------
    # Public pipeline method
    # -------------------------------------------------------------------
    def load_data(self, filename):
        if not os.path.exists( filename ):
            return None
        return ni_imread.read_imaq_image(filename).astype('float')
    
    def analyze_data(self, data, bg=None, context=None, analyzer_dict=None):
        """
        Run the analysis pipeline on one image.

        The steps and their order come from ``analyzer_dict['pipeline']``, a
        list of ``(step_name, kwargs)`` tuples, e.g.::

            'pipeline': [
                ('subtract_bg',    {}),
                ('auto_roi',       {'size': 350, 'thresh': 0.8}),
                ('threshold',      {'low': 16, 'units': 'counts'}),
                ('median_filter',  {'N': 5}),
                ('second_moments', {'r_max': 100}),
            ]

        Step names may repeat. Valid names are the keys of ``self.STEPS``;
        each maps to a ``_step_<name>`` method, which wraps a static method
        that can also be called on its own. Settings shared by several steps
        (``dx``, ``dy``, ``spatial_units``, ``round_centroid``) stay at the
        top level of ``analyzer_dict``.

        A dict without ``'pipeline'`` uses the legacy flat keys
        (``bg_file``, ``median_filter``, ``threshold_high``, ...), translated
        to the old fixed order by ``legacy_to_pipeline``, with a warning.
        """
        if analyzer_dict is None:
            analyzer_dict = self.analyzer_dict
        analyzer_dict = analyzer_dict or {}

        if data is None:
            print("Warning: analyze_data() called with None input — skipping analysis.")
            return None, {}, {}

        pipeline = self._resolve_pipeline(analyzer_dict)

        s = {
            'data': np.copy(data).astype(float),
            'bg': bg,
            'ad': analyzer_dict,
            'results': {},
            'lineouts': {},
            'x': None, 'y': None, 'x0': None, 'y0': None,
            'roi_offset': (0, 0),      # (row, col) of the ROI's corner in the full frame
        }
        for name, kwargs in pipeline:
            getattr(self, self.STEPS[name])(s, **(kwargs or {}))

        return s['data'], s['results'], s['lineouts']

    def _resolve_pipeline(self, analyzer_dict):
        if 'pipeline' in analyzer_dict:
            pipeline = [(step, {}) if isinstance(step, str) else tuple(step)
                        for step in analyzer_dict['pipeline']]
        else:
            if not getattr(self, '_legacy_warned', False):
                warnings.warn(
                    "ImageAnalyzer: flat analyzer_dict keys are deprecated; use "
                    "analyzer_dict['pipeline']. ImageAnalyzer.legacy_to_pipeline"
                    "(analyzer_dict) returns the equivalent pipeline.",
                    FutureWarning, stacklevel=3)
                self._legacy_warned = True
            pipeline = self.legacy_to_pipeline(analyzer_dict)

        unknown = [name for name, _ in pipeline if name not in self.STEPS]
        if unknown:
            raise ValueError(f"Unknown pipeline step(s) {unknown}. "
                             f"Valid steps: {sorted(self.STEPS)}")
        return pipeline

    @staticmethod
    def legacy_to_pipeline(analyzer_dict):
        """
        Translate a legacy flat ``analyzer_dict`` into the equivalent
        ``pipeline`` list, reproducing the old fixed order of analyze_data.
        """
        ad = analyzer_dict
        p = []
        if ad.get('return_n_saturated', True):
            p.append(('n_saturated', {'bit_depth': ad.get('bit_depth', 12),
                                      'saturation_threshold': ad.get('saturation_threshold', None)}))
        if ad.get('bg_file', False):
            p.append(('subtract_bg', {}))
        if ad.get('bg_const', 0):
            p.append(('bg_const', {'value': ad['bg_const']}))
        if ad.get('roi', None) is not None:
            p.append(('roi', {'roi': ad['roi']}))
        if ad.get('auto_roi', False):
            p.append(('auto_roi', {'size': ad.get('auto_roi_size', 500),
                                   'thresh': ad.get('auto_roi_thresh', 0.85),
                                   'presmooth': 0}))
        if ad.get('remove_outliers', False):
            p.append(('remove_outliers', {'threshold': ad.get('outlier_threshold', 3),
                                          'mask': ad.get('outlier_mask', None)}))
        # the old code clipped the top 0.5 % by default before computing counts
        threshold_high = ad.get('threshold_high', 0.995)
        if threshold_high > 0:
            p.append(('threshold', {'high': threshold_high}))
        if ad.get('return_data_counts', True):
            p.append(('counts', {}))
        if ad.get('median_filter', False):
            p.append(('median_filter', {'N': ad.get('N_filt', 3)}))
        if ad.get('generate_lineouts', False):
            p.append(('lineouts', {'centroid_method': ad.get('centroid_method', 'center'),
                                   'x0': ad.get('x0', None),
                                   'y0': ad.get('y0', None),
                                   'centroid_thresh': ad.get('centroid_threshold_low', 0.85),
                                   'lineout_method': ad.get('lineout_method', 'center'),
                                   'Nlo': ad.get('Nlo', 2)}))
        if ad.get('fit_super_gaussian', False):
            p.append(('gauss_fit', {'fit_exponent': ad.get('include_exponent', True)}))
        threshold_low = ad.get('threshold_low', -1)
        threshold_high = ad.get('threshold_high', -1)
        if threshold_low >= 0 or threshold_high > 0:
            p.append(('threshold', {'low': threshold_low, 'high': threshold_high}))
        if ad.get('get_fwhm', False):
            p.append(('fwhm', {}))
        if ad.get('measured_misalignment', False):
            p.append(('misalignment', {'target_x0': ad.get('x0', None),
                                       'target_y0': ad.get('y0', None)}))
        if ad.get('fitted_misalignment', False) and ad.get('fit_super_gaussian', False):
            p.append(('misalignment', {'fitted': True}))
        if ad.get('calibrate_to_fluence', False):
            p.append(('fluence', {'pulse_energy': ad.get('pulse_energy', 41.2)}))
        return p

    def display_data(self, data, display_dict=None, return_dict=None, title=None, fig=None, ax=None):
        if display_dict is None:
            display_dict = self.display_dict
            
        if fig is None or ax is None:
            fig, ax = plt.subplots(constrained_layout=True, figsize=display_dict.get('figsize', (6, 5)))

        if return_dict is not None:
            extent = return_dict.get( 'imshow_extent', None )
        else:
            extent = display_dict.get('extent', None)

        im = ax.imshow(data,
                        aspect=display_dict.get('aspect', 'equal'),
                        norm=display_dict.get('norm', None),
                        cmap=display_dict.get('cmap', 'RdBu'),
                        interpolation=display_dict.get('interpolation', None),
                        origin='lower',
                        extent=extent,
                        vmin=display_dict.get('vmin', None),
                        vmax=display_dict.get('vmax', None),
                      )
        if not display_dict.get('cbar_off', False):
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(display_dict.get('cbar_label', 'Counts'))

        spatial_units = display_dict.get('spatial_units', 'pixels')
        xtitle = display_dict.get('xtitle', 'x')
        ytitle = display_dict.get('ytitle', 'y')

        ax.set_xlabel(r'$%s \ (\mathrm{%s})$' %(xtitle, spatial_units))
        ax.set_ylabel(r'$%s \ (\mathrm{%s})$' %(ytitle, spatial_units))

        axlims = display_dict.get('axlims', None)
        tick_dx = display_dict.get('tick_dx', None)
        if tick_dx is not None and axlims is not None:
            ax.set_xticks(np.arange(-axlims, axlims+tick_dx, tick_dx))
            ax.set_yticks(np.arange(-axlims, axlims+tick_dx, tick_dx))

        if axlims is not None:
            ax.set_xlim([-axlims, axlims])
            ax.set_ylim([-axlims, axlims])


        for v_line in display_dict.get('v_lines', []):
            ax.axvline(x=v_line, color='k', linestyle='--')

        for h_line in display_dict.get('h_lines', []):
            ax.axhline(y=h_line, color='k', linestyle='--')

        if title is not None:
            ax.set_title(title)

        return fig, ax

    def display_lineout(self, x, x_lo, display_dict=None, title=None, fig=None, ax=None):
        if display_dict is None:
            display_dict = self.display_dict

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=display_dict.get('figsize', (6, 5)))
            
        ax.plot(x, x_lo, label=display_dict.get('plot_label', None))

        spatial_units = display_dict.get('spatial_units', 'pixels')
        xtitle = display_dict.get('xtitle', 'x')
        ytitle = display_dict.get('ytitle', 'y')

        ax.set_xlabel(r'$%s \ (\mathrm{%s})$' %(xtitle, spatial_units))
        ax.set_ylabel(r'$%s \ (\mathrm{%s})$' %(ytitle, spatial_units))

        axlims = display_dict.get('axlims', None)
        tick_dx = display_dict.get('tick_dx', None)
        if tick_dx is not None and axlims is not None:
            ax.set_xticks(np.arange(0, axlims+tick_dx, tick_dx))

        if axlims is not None:
            ax.set_xlim([0, axlims])

        for v_line in display_dict.get('v_lines', []):
            ax.axvline(x=v_line, color='k', linestyle='--')

        for h_line in display_dict.get('h_lines', []):
            ax.axhline(y=h_line, color='k', linestyle='--')

        title_append = display_dict.get('title_append', '')
        if title is not None:
            ax.set_title('%s - %s' %(title, title_append))

        return fig, ax
    
    def write_displayed_data(self, fig, analysis_dir, scan, shot_num):
        save_path = get_analysed_shot_save_path(
            analysis_dir,
            f'{self.output_diagnostic}_disp',
            scan,
            shot_num,
            '.png', 
            append_info='_disp'
        )

        fig.savefig( save_path, dpi=200 )

    def write_analyzed_lineouts(self, lineouts, analysis_dir, scan, shot_num):
        save_path = get_analysed_shot_save_path(
            analysis_dir,
            f'{self.output_diagnostic}_lineouts', 
            scan,
            shot_num,
            '.txt', 
        )
        save_lineouts_to_txt(lineouts, save_path, cols=None, float_format="%.3f", na_rep="nan")




    # -------------------------------------------------------------------
    # Pipeline steps. Each _step_<name>(s, **kwargs) reads and updates the
    # state dict s ('data', 'bg', 'ad', 'results', 'lineouts', 'x', 'y',
    # 'x0', 'y0', 'roi_offset') and does the numerics through a static
    # method. Subclasses add steps with
    #     STEPS = {**ImageAnalyzer.STEPS, 'my_step': '_step_my_step'}
    # -------------------------------------------------------------------
    STEPS = {
        'n_saturated':      '_step_n_saturated',
        'subtract_bg':      '_step_subtract_bg',
        'bg_const':         '_step_bg_const',
        'roi':              '_step_roi',
        'auto_roi':         '_step_auto_roi',
        'remove_outliers':  '_step_remove_outliers',
        'threshold':        '_step_threshold',
        'median_filter':    '_step_median_filter',
        'radial_mask':      '_step_radial_mask',
        'counts':           '_step_counts',
        'lineouts':         '_step_lineouts',
        'gauss_fit':        '_step_gauss_fit',
        'fwhm':             '_step_fwhm',
        'second_moments':   '_step_second_moments',
        'encircled_energy': '_step_encircled_energy',
        'misalignment':     '_step_misalignment',
        'fluence':          '_step_fluence',
    }

    def _step_n_saturated(self, s, bit_depth=12, saturation_threshold=None):
        s['results']['n_saturated'] = self.compute_n_saturated(
            s['data'], bit_depth=bit_depth, saturation_threshold=saturation_threshold)

    def _step_subtract_bg(self, s, file=None, scale=1.0):
        """Subtract `file` if given, else the bg passed to analyze_data (skipped if None)."""
        bg = self.load_data(file) if file is not None else s['bg']
        if bg is None:
            return
        bg = np.asarray(bg, dtype=float)
        if bg.shape != s['data'].shape:
            raise ValueError(f"subtract_bg: background shape {bg.shape} != image shape "
                             f"{s['data'].shape}; put subtract_bg before any ROI step.")
        s['data'] = s['data'] - scale*bg

    def _step_bg_const(self, s, value=0):
        s['data'] = s['data'] - value

    def _step_roi(self, s, roi=None):
        if roi is None:
            return
        r0, _, c0, _ = self.clip_roi(roi, s['data'].shape)
        s['data'] = self.roi_data(s['data'], roi)
        self._shift_roi_offset(s, r0, c0)

    def _step_auto_roi(self, s, size=500, thresh=0.85, method='centroid', presmooth=3):
        """Square (int size) or rectangular ((ny, nx) size) ROI centred on the beam."""
        x0, y0 = self.find_beam_center(s['data'], method=method, thresh=thresh,
                                       presmooth=presmooth)
        ny, nx = (size, size) if np.isscalar(size) else size
        N, M = s['data'].shape
        # same centre rounding/clipping as extract_subarray
        xi, yi = np.clip(int(x0), 0, M - 1), np.clip(int(y0), 0, N - 1)
        s['data'] = self.extract_subarray(s['data'], x0, y0, nx, ny)
        self._shift_roi_offset(s, int(yi) - ny//2, int(xi) - nx//2)

    def _shift_roi_offset(self, s, dr, dc):
        r, c = s['roi_offset']
        s['roi_offset'] = (r + dr, c + dc)
        # coordinates from earlier steps refer to the old frame
        s['x'] = s['y'] = s['x0'] = s['y0'] = None

    def _step_remove_outliers(self, s, threshold=3, mask=None):
        s['data'] = self.remove_outliers(s['data'], threshold=threshold, mask=mask)

    def _step_threshold(self, s, low=-1, high=-1, units='auto'):
        s['data'] = self.apply_threshold(s['data'], low, high, units=units)

    def _step_median_filter(self, s, N=3):
        s['data'] = self.median_filter(s['data'], N)

    def _step_radial_mask(self, s, r_max=None, x0=None, y0=None, fill_value=0):
        """Set pixels further than r_max pixels from (x0, y0) (default: centroid) to fill_value."""
        if r_max is None:
            return
        if x0 is None or y0 is None:
            x0, y0 = self.centroid(s['data'])
        s['data'] = self.apply_radial_mask(s['data'], x0, y0, r_max, fill_value=fill_value)

    def _step_counts(self, s):
        s['results'].update(self.compute_data_counts(s['data']))

    def _step_lineouts(self, s, centroid_method='center', x0=None, y0=None,
                       centroid_thresh=0.85, lineout_method='center', Nlo=2):
        ad = s['ad']
        x, y, x0, y0 = self.get_spatial_coords(
            s['data'], method=centroid_method, x0=x0, y0=y0,
            dx=ad.get('dx', 1), dy=ad.get('dy', 1),
            centroid_thresh=centroid_thresh,
            round_centroid=ad.get('round_centroid', True))
        x_lo, y_lo = self.get_lineouts(s['data'], x0, y0, method=lineout_method, Nlo=Nlo)
        s.update(x=x, y=y, x0=x0, y0=y0)
        s['lineouts'].update({'x': x, 'y': y, 'x_lo': x_lo, 'y_lo': y_lo})
        res = s['results']
        res['x0'] = x0
        res['y0'] = y0
        res['x0 frame'] = x0 + s['roi_offset'][1]
        res['y0 frame'] = y0 + s['roi_offset'][0]
        res['imshow_extent'] = self.get_imshow_extent(x, y)

    def _ensure_lineouts(self, s):
        """Lineouts from the last 'lineouts' step, or centroid lineouts of the current data."""
        if 'x_lo' not in s['lineouts']:
            self._step_lineouts(s, centroid_method='centroid')
        lo = s['lineouts']
        return lo['x'], lo['y'], lo['x_lo'], lo['y_lo']

    def _step_gauss_fit(self, s, fit_exponent=True):
        x, y, x_lo, y_lo = self._ensure_lineouts(s)
        fit_x = self.fit_super_gaussian(x, x_lo, fit_exponent=fit_exponent)
        fit_y = self.fit_super_gaussian(y, y_lo, fit_exponent=fit_exponent)
        res = s['results']
        res['fit_x'] = fit_x
        res['fit_y'] = fit_y
        keys = ['amplitude', 'center', 'sigma', 'offset'] + (['N'] if fit_exponent else [])
        for ax_name, fit in (('x', fit_x), ('y', fit_y)):
            res.update({f'gauss fit {ax_name} {k}': fit.get(k, np.nan) for k in keys})

    def _step_fwhm(self, s):
        ad = s['ad']
        _, _, x_lo, y_lo = self._ensure_lineouts(s)
        res = s['results']
        res['fwhm x'] = get_lineout_width(x_lo)[0]*ad.get('dx', 1)
        res['fwhm y'] = get_lineout_width(y_lo)[0]*ad.get('dy', 1)
        res['fwhm_av'] = np.nanmean([res['fwhm x'], res['fwhm y']])
        res['fwhm_diff'] = np.abs(res['fwhm x'] - res['fwhm y'])

    def _step_second_moments(self, s, r_max=None, x0=None, y0=None, n_iter=1, prefix='2nd moment'):
        ad = s['ad']
        m = self.compute_second_moments(s['data'], dx=ad.get('dx', 1), dy=ad.get('dy', 1),
                                        x0=x0, y0=y0, r_max=r_max, n_iter=n_iter)
        res = s['results']
        for k in ('x', 'y', 'x45', 'y45', 'major', 'minor'):
            res[f'{prefix} {k}'] = m[k]
        res[f'{prefix} angle'] = m['angle']
        res[f'{prefix} x0'] = m['x0'] + s['roi_offset'][1]    # full-frame pixels
        res[f'{prefix} y0'] = m['y0'] + s['roi_offset'][0]

    def _step_encircled_energy(self, s, radii=(), fractions=(0.5, 0.865), x0=None, y0=None):
        """radii in spatial units (dx); fractions of total energy, 0-1."""
        ad = s['ad']
        res = s['results']
        res.update(self.compute_encircled_energy(s['data'], dx=ad.get('dx', 1), dy=ad.get('dy', 1),
                                                 x0=x0, y0=y0, radii=radii, fractions=fractions))

    def _step_misalignment(self, s, target_x0=None, target_y0=None, fitted=False):
        res = s['results']
        if fitted:
            res.update(self.calculate_misalignment(
                s['data'], 0, 0,
                measured_x0=res['fit_x']['center'], measured_y0=res['fit_y']['center'],
                analyzer_dict={}, offset_type='fitted'))
        else:
            res.update(self.calculate_misalignment(s['data'], target_x0, target_y0,
                                                   analyzer_dict=s['ad']))

    def _step_fluence(self, s, pulse_energy=41.2):
        ad = s['ad']
        s['data'], fluence = self.adjust_data_to_fluence(
            s['data'], pulse_energy, ad.get('dx', 1), ad.get('dy', 1),
            spatial_units=ad.get('spatial_units', 'microns'))
        s['results']['fluence_per_count'] = fluence
        s['results']['max fluence'] = np.nanmax(s['data'])

    # -------------------------------------------------------------------
    # Utility methods (can be overridden or extended in subclasses)
    # -------------------------------------------------------------------

    @staticmethod
    def roi_data(data, roi):
        """
        Crop data to region of interest.
        ROI should be [row_start, row_end, col_start, col_end]; it is
        clipped to the image (see clip_roi), None ends mean the image edge.
        """
        r0, r1, c0, c1 = ImageAnalyzer.clip_roi(roi, data.shape)
        return data[r0:r1, c0:c1]

    @staticmethod
    def compute_data_counts(data):
        """
        Return max, mean, and sum of data ignoring NaNs.
        """
        return {
            'max_counts':  np.nanmax(data),
            'mean_counts': np.nanmean(data),
            'sum_counts':  np.nansum(data)
        }

    @staticmethod
    def calculate_misalignment(data, 
                               target_x, target_y, 
                               measured_x0=None, measured_y0=None, 
                               analyzer_dict={}, 
                               offset_type='measured'):
        
        if measured_x0 is None and measured_y0 is None:
            measured_x0, measured_y0 = ImageAnalyzer._get_centroid(data, 
                                                    thresh_min=analyzer_dict.get('centroid_threshold_low', 0.85)
                                                   )
            offset_type='measured'

        x_offset = (measured_x0 - target_x)*analyzer_dict.get('dx', 1)
        y_offset = (measured_y0 - target_y)*analyzer_dict.get('dy', 1)
        r_offset = np.sqrt(x_offset**2 + y_offset**2)

        return {f'{offset_type} x_offset' : x_offset, 
                f'{offset_type} y_offset' : y_offset, 
                f'{offset_type} r_offset' : r_offset
               }

    @staticmethod
    def compute_n_saturated(data, bit_depth=12, saturation_threshold=None):
        """
        Count the number of saturated pixels in 'data'.
        """
        if saturation_threshold is None:
            saturation_threshold = (2**bit_depth) - 2
        return np.count_nonzero(data >= saturation_threshold)

    @staticmethod
    def apply_threshold(data, threshold_low, threshold_high, units='auto'):
        """
        Zero‐out pixels outside [threshold_low, threshold_high): pixels equal
        to threshold_low are kept (as in the LabVIEW focus analysis).
        A negative threshold_low or non-positive threshold_high is ignored.

        units: 'auto' (legacy) — low > 1 / high >= 1 are counts, otherwise
        a fraction of the max; 'counts' — always counts; 'frac' — always a
        fraction of the max.
        """
        if units not in ('auto', 'counts', 'frac'):
            raise ValueError(f"apply_threshold: unknown units {units!r}")
        data_copy = np.copy(data)

        if units != 'auto':
            peak = np.nanmax(data) if units == 'frac' else 1.0
            if threshold_low >= 0:
                data_copy *= (data >= threshold_low*peak).astype(np.uint8)
            if threshold_high > 0:
                data_copy *= (data < threshold_high*peak).astype(np.uint8)
            return data_copy

        if threshold_low >= 0:
            if threshold_low > 1:
                data_copy *= (data >= threshold_low).astype(np.uint8)
            else:
                data_copy *= (data >= threshold_low * np.nanmax(data)).astype(np.uint8)

        if threshold_high > 0:
            if threshold_high < 1:
                data_copy *= (data < threshold_high * np.nanmax(data)).astype(np.uint8)
            else:
                data_copy *= (data < threshold_high).astype(np.uint8)

        return data_copy

    @staticmethod
    def remove_outliers(data, threshold=3, mask=None):
        """
        Replace outliers (beyond threshold*std) with NaN.
        Optional 'mask' for partial application.
        """
        mean = np.nanmean(data)
        std_dev = np.nanstd(data)

        lower_bound = mean - threshold*std_dev
        upper_bound = mean + threshold*std_dev

        outliers = (data < lower_bound) | (data > upper_bound)
        if mask is not None:
            outliers = np.logical_and(outliers, mask)

        data_with_nans = np.where(outliers, np.nan, data)
        return data_with_nans

    @staticmethod
    def get_spatial_coords(data, method='center', x0=None, y0=None,
                           dx=1, dy=1, centroid_thresh=0.85, round_centroid=True):
        """
        Compute x, y arrays for data, using one of several 'method's to place origin:
          - 'center', 'pixel', 'manual', 'centroid'

        With method='centroid' and round_centroid=True (default), the axis
        origin is the centroid rounded to the nearest pixel, so every shot's
        axis lies on the same pixel lattice and lineouts from different shots
        line up by a whole-pixel shift. The returned x0, y0 stay sub-pixel.
        """
        if method == 'center':
            x0 = 0.5*data.shape[1]
            y0 = 0.5*data.shape[0]
        elif method == 'pixel':
            x0, y0 = 0, 0
            dx, dy = 1, 1
        elif method == 'manual':
            if x0 is None:
                x0 = 0.5*data.shape[1]
            if y0 is None:
                y0 = 0.5*data.shape[0]
        elif method == 'centroid':
            x0, y0 = ImageAnalyzer._get_centroid(data, thresh_min=centroid_thresh)
        else:
            x0, y0 = 0.5*data.shape[1], 0.5*data.shape[0]

        x_origin, y_origin = x0, y0
        if method == 'centroid' and round_centroid:
            x_origin, y_origin = np.round(x0), np.round(y0)
        x = (np.arange(data.shape[1]) - x_origin) * dx
        y = (np.arange(data.shape[0]) - y_origin) * dy
        return x, y, x0, y0

    @staticmethod
    def _get_centroid(data, thresh_min=0.85, thresh_max=4095):
        """
        Helper for centroid calculation. 
        Used by get_spatial_coords(method='centroid').
        """
        data_copy = np.copy(data)
        if thresh_min > 1:
            data_copy *= (data > thresh_min).astype(np.uint8)
        else:
            data_copy *= (data > thresh_min * np.nanmax(data)).astype(np.uint8)

        if thresh_max < 1:
            data_copy *= (data < thresh_max * np.nanmax(data)).astype(np.uint8)
        else:
            data_copy *= (data < thresh_max).astype(np.uint8)

        sum_counts = np.nansum(data_copy)
        x0 = np.nansum(np.arange(data_copy.shape[1]) * data_copy) / sum_counts
        y0 = np.nansum(np.arange(data_copy.shape[0]) * data_copy.T) / sum_counts
        return x0, y0

    @staticmethod
    def get_lineouts(data, x0, y0, method='center', Nlo=2):
        """
        Return x_lo, y_lo lineouts from data around (x0,y0).
        method='center' takes vertical+horizontal strips of width Nlo.
        method='all' averages entire dimension.
        """
        if method == 'center':
            x_lo = np.nanmean(data[int(y0 - 0.5*Nlo):int(y0 + 0.5*Nlo), :], axis=0)
            y_lo = np.nanmean(data[:, int(x0 - 0.5*Nlo):int(x0 + 0.5*Nlo)], axis=1)
        elif method == 'all':
            x_lo = np.nanmean(data, axis=0)
            y_lo = np.nanmean(data, axis=1)
        else:
            x_lo = np.nanmean(data, axis=0)
            y_lo = np.nanmean(data, axis=1)
        return x_lo, y_lo
    
    @staticmethod
    def display_imshow_lineouts(fig, ax, extent, x=None, y=None, x_lo=None, y_lo=None, amp=0.15, color='k', ls='-'):
        if x is not None and x_lo is not None:
            ax.plot(
                x,
                amp * x_lo * (extent[3] - extent[2]) + extent[2] ,
                c=color,
                ls=ls,
            )
        if y is not None and y_lo is not None:
            ax.plot(
                amp * y_lo * (extent[1] - extent[0]) + extent[0],
                y,
                c=color,
                ls=ls,
            )
        return fig, ax

    @staticmethod
    def adjust_data_to_fluence(image, energy, dx, dy, spatial_units='microns'):
        """
        Adjust image data to represent fluence in J/cm² based on total energy and pixel size.

        Parameters:
            image (ndarray): 2D image array.
            energy (float): Total energy of the laser pulse (Joules).
            dx (float): Pixel size in x (microns).
            dy (float): Pixel size in y (microns).

        Returns:
            calibrated_image (ndarray): Image calibrated to J/cm².
            fluence (float): Calibration factor in J/cm² per count.
        """

        if spatial_units == 'microns' or spatial_units == 'um':
            dx_cm = dx / 10000.0
            dy_cm = dy / 10000.0
        else:
            dx_cm = dx / 10.0
            dy_cm = dy / 10.0


        # Compute energy per count
        energy_per_count = energy / np.sum(image)

        # Convert to fluence
        fluence = energy_per_count / (dx_cm * dy_cm)

        # Apply scaling
        calibrated_image = image * fluence

        return calibrated_image, fluence


    @staticmethod
    def get_imshow_extent(x, y):
        return np.array([x[0], x[-1], y[0], y[-1]])

    @staticmethod
    def fit_super_gaussian(xdata, ydata, fit_exponent=False, p0=None, bounds=None,
                           loss='linear', max_nfev=5000):
        """
        Fit a super‐Gaussian function to (xdata,ydata).
        """
        xdata = np.array(xdata, dtype=float)
        ydata = np.array(ydata, dtype=float)

        valid_mask = np.isfinite(xdata) & np.isfinite(ydata)
        xdata = xdata[valid_mask]
        ydata = ydata[valid_mask]
        if xdata.size == 0:
            return {'success': False, 'message': 'No valid data'}

        def estimate_initial_params(xx, yy):
            y_min, y_max = np.nanmin(yy), np.nanmax(yy)
            amp_guess    = y_max - y_min
            idx_max      = np.nanargmax(yy)
            cen_guess    = xx[idx_max]
            off_guess    = y_min

            half_level = y_min + 0.5*amp_guess
            crosses = np.where(np.diff((yy > half_level).astype(int)) != 0)[0]
            if len(crosses) >= 2:
                x_left, x_right = xx[crosses[0]], xx[crosses[-1]]
                sigma_guess = (x_right - x_left)/2.355
                sigma_guess = max(1e-6, np.abs(sigma_guess))
            else:
                sigma_guess = max(1e-6, (xx.max() - xx.min())/10)
            return [amp_guess, cen_guess, sigma_guess, off_guess]

        if p0 is None:
            p0_est = estimate_initial_params(xdata, ydata)
            if fit_exponent:
                p0_est.append(2.0)
            p0 = p0_est

        if bounds is None:
            if fit_exponent:
                bounds = ([0, -np.inf, 1e-12, -np.inf, 1],
                          [np.inf, np.inf, np.inf, np.inf, 20])
            else:
                bounds = ([0, -np.inf, 1e-12, -np.inf],
                          [np.inf, np.inf, np.inf, np.inf])

        def residual(params, xx, yy):
            if fit_exponent:
                amp, cen, sig, off, N = params
            else:
                amp, cen, sig, off = params
                N = 2
            return yy - super_gaussian(xx, amp, cen, sig, off, N)

        res = least_squares(residual, p0, args=(xdata, ydata),
                            bounds=bounds, loss=loss, max_nfev=max_nfev)

        result = {
            'success':  res.success,
            'message':  res.message,
            'cost':     res.cost,
            'params':   res.x
        }
        if fit_exponent:
            amp, cen, sig, off, exponent = res.x
            result.update({'amplitude': amp, 'center': cen, 'sigma': sig,
                           'offset': off, 'N': exponent})
        else:
            amp, cen, sig, off = res.x
            result.update({'amplitude': amp, 'center': cen, 'sigma': sig,
                           'offset': off})
        return result

    @staticmethod
    def clip_roi(roi, shape):
        """
        ROI [row_start, row_end, col_start, col_end] clipped to an image of
        `shape`. None ends mean the image edge; negative values count from
        the end, as in Python slicing.
        """
        N, M = shape[:2]
        r0, r1, c0, c1 = roi
        r0, r1, _ = slice(r0, r1).indices(N)
        c0, c1, _ = slice(c0, c1).indices(M)
        return r0, max(r0, r1), c0, max(c0, c1)

    @staticmethod
    def median_filter(data, N=3):
        """N x N median filter (N odd). NaNs are treated as 0."""
        return medfilt2d(np.nan_to_num(np.asarray(data, dtype=np.float64)), [N, N])

    @staticmethod
    def centroid(data, x0=None, y0=None, r_max=None, n_iter=1):
        """
        Intensity-weighted centroid (x0, y0) in pixels, negative and NaN
        pixels counted as 0. With r_max, the centroid is recomputed n_iter
        times using only pixels within r_max pixels of the previous estimate
        (starting from x0, y0 if given), so isolated far-out pixels can't
        pull it.
        """
        w = np.clip(np.nan_to_num(np.asarray(data, dtype=float)), 0, None)
        N, M = w.shape
        xx, yy = np.arange(M)[None, :], np.arange(N)[:, None]
        total = w.sum()
        if total <= 0:
            return 0.5*M, 0.5*N
        if x0 is None or y0 is None:
            x0, y0 = (w*xx).sum()/total, (w*yy).sum()/total
        if r_max is not None:
            for _ in range(n_iter):
                wm = w*(((xx - x0)**2 + (yy - y0)**2) <= r_max**2)
                total = wm.sum()
                if total <= 0:
                    break
                x0, y0 = (wm*xx).sum()/total, (wm*yy).sum()/total
        return x0, y0

    @staticmethod
    def find_beam_center(data, method='centroid', thresh=0.85, presmooth=3):
        """
        Beam centre (x0, y0) in pixels, for placing an automatic ROI.

        method 'centroid': thresholded centroid (_get_centroid with
        thresh_min=thresh); 'peak': position of the maximum. presmooth > 1
        median-filters a copy first so single hot pixels can't move the
        centre (presmooth=0 reproduces the legacy auto_roi exactly). Falls
        back to the image centre if the result isn't finite.
        """
        d = np.asarray(data, dtype=float)
        if presmooth and presmooth > 1:
            d = ImageAnalyzer.median_filter(d, int(presmooth) | 1)
        N, M = d.shape
        if method == 'peak':
            if not np.any(np.isfinite(d)):
                return 0.5*M, 0.5*N
            y0, x0 = np.unravel_index(np.nanargmax(d), d.shape)
        elif method == 'centroid':
            with np.errstate(invalid='ignore', divide='ignore'):
                x0, y0 = ImageAnalyzer._get_centroid(d, thresh_min=thresh)
        else:
            raise ValueError(f"find_beam_center: unknown method {method!r}")
        if not (np.isfinite(x0) and np.isfinite(y0)):
            return 0.5*M, 0.5*N
        return x0, y0

    @staticmethod
    def apply_radial_mask(data, x0, y0, r_max, fill_value=0):
        """Copy of data with pixels further than r_max pixels from (x0, y0) set to fill_value."""
        N, M = data.shape
        r2 = (np.arange(M)[None, :] - x0)**2 + (np.arange(N)[:, None] - y0)**2
        return np.where(r2 <= r_max**2, data, fill_value)

    @staticmethod
    def compute_second_moments(data, dx=1, dy=1, x0=None, y0=None, r_max=None, n_iter=1):
        """
        Second-moment (ISO 11146) beam widths.

        Negative and NaN pixels count as 0. The centre is the intensity
        centroid unless x0, y0 (pixels) are given. With r_max (pixels), only
        pixels within r_max of the centre are used, and the centroid is
        re-found inside that circle n_iter times.

        Widths are 2*sigma (sigma = rms width), in dx/dy units, matching the
        LabVIEW focus analysis. Returns a dict:
          'x', 'y'          2*sigma along x and y
          'x45', 'y45'      2*sigma along the +45 deg (x=y) and -45 deg
                            diagonals: sigma45^2 = (sx^2 + sy^2)/2 +- sxy
          'major', 'minor'  2*sigma along the principal axes
          'angle'           deg, major axis from +x
          'cov xy'          the xy covariance sxy, in (dx units)^2
          'x0', 'y0'        centre used, pixels
        """
        w = np.clip(np.nan_to_num(np.asarray(data, dtype=float)), 0, None)
        N, M = w.shape
        cx, cy = ImageAnalyzer.centroid(w, x0=x0, y0=y0, r_max=r_max, n_iter=n_iter)
        if x0 is not None and y0 is not None:
            cx, cy = x0, y0
        xx = (np.arange(M)[None, :] - cx)
        yy = (np.arange(N)[:, None] - cy)
        if r_max is not None:
            w = w*((xx**2 + yy**2) <= r_max**2)
        total = w.sum()
        out = {'x0': cx, 'y0': cy}
        if total <= 0:
            for k in ('x', 'y', 'x45', 'y45', 'major', 'minor'):
                out[k] = np.nan
            out.update({'cov xy': np.nan, 'angle': np.nan})
            return out
        X, Y = xx*dx, yy*dy
        sxx = (w*X**2).sum()/total
        syy = (w*Y**2).sum()/total
        sxy = (w*X*Y).sum()/total
        mean = 0.5*(sxx + syy)
        diff = np.sqrt((0.5*(sxx - syy))**2 + sxy**2)
        var = {'x': sxx, 'y': syy, 'x45': mean + sxy, 'y45': mean - sxy,
               'major': mean + diff, 'minor': mean - diff}
        for k, v in var.items():
            out[k] = 2*np.sqrt(max(v, 0))
        out['cov xy'] = sxy
        out['angle'] = np.degrees(0.5*np.arctan2(2*sxy, sxx - syy))
        return out

    @staticmethod
    def compute_encircled_energy(data, dx=1, dy=1, x0=None, y0=None, radii=(), fractions=(0.5, 0.865)):
        """
        Encircled energy about (x0, y0) (pixels; default: centroid).

        Returns 'ee r=<R>' (fraction of the total within radius R, R in dx
        units) for each of `radii`, 'ee radius <p>%' (radius holding a
        fraction p of the total) for each of `fractions`, and
        'peak fraction' (max pixel / total). Negative and NaN pixels count
        as 0.
        """
        w = np.clip(np.nan_to_num(np.asarray(data, dtype=float)), 0, None)
        N, M = w.shape
        if x0 is None or y0 is None:
            x0, y0 = ImageAnalyzer.centroid(w)
        r = np.sqrt(((np.arange(M)[None, :] - x0)*dx)**2 + ((np.arange(N)[:, None] - y0)*dy)**2)
        total = w.sum()
        out = {}
        if total <= 0:
            out.update({f'ee r={R:g}': np.nan for R in radii})
            out.update({f'ee radius {100*p:g}%': np.nan for p in fractions})
            out['peak fraction'] = np.nan
            return out
        order = np.argsort(r, axis=None)
        r_sorted = r.ravel()[order]
        cum = np.cumsum(w.ravel()[order])/total
        for R in radii:
            out[f'ee r={R:g}'] = w[r <= R].sum()/total
        for p in fractions:
            out[f'ee radius {100*p:g}%'] = float(np.interp(p, cum, r_sorted))
        out['peak fraction'] = w.max()/total
        return out

    # -------------------------------------------------------------------
    # Optional / Misc: plotting, rotating, or special calculations
    # -------------------------------------------------------------------

    @staticmethod
    def rotate_around(image, angle, x0, y0):
        """
        Rotate 'image' around (x0, y0) in‐place via affine_transform.
        """
        angle_rad = np.deg2rad(angle)
        cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)

        rot_mat = np.array([[cos_t, sin_t, 0],
                            [-sin_t, cos_t, 0],
                            [0, 0, 1]])
        trans1 = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [-x0, -y0, 1]])
        trans2 = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [x0, y0, 1]])
        transform_matrix = trans1 @ rot_mat @ trans2
        transform_linear = transform_matrix[:2, :2]
        offset = transform_matrix[:2, 2]

        rotated = affine_transform(image, transform_linear, offset=offset,
                                   output_shape=image.shape, order=1)
        return rotated

    @staticmethod
    def rotational_average(image, x0, y0, interpolate_nans=True):
        """
        Compute radial average around (x0, y0).
        """
        Y, X = np.indices(image.shape)
        R = np.sqrt((X - x0)**2 + (Y - y0)**2)
        R = np.round(R).astype(int)

        max_r = np.max(R)
        avg = np.zeros(max_r+1)

        for r in range(max_r+1):
            mask = (R == r)
            avg[r] = np.nanmean(image[mask])

        if interpolate_nans:
            avg = ImageAnalyzer.interpolate_nan_values(avg)
        return avg

    @staticmethod
    def interpolate_nan_values(data):
        """
        Interpolate over 1D array's NaN regions using nearest valid points.
        """
        nans = np.isnan(data)
        notnans = ~nans
        if not np.any(notnans):
            return data  # all NaNs
        indices = np.arange(len(data))
        data[nans] = np.interp(indices[nans], indices[notnans], data[notnans])
        return data
    
    @staticmethod
    def angular_difference_deg(theta, theta0):
        """
        Smallest signed angular difference (theta - theta0) in degrees, wrapped to [-180, 180).
        theta, theta0 can be scalars or arrays.
        """
        return (theta - theta0 + 180.0) % 360.0 - 180.0
    
    @staticmethod
    def angle_deg_from_points(x0, y0, x1, y1):
        """
        Angle from (x0, y0) -> (x1, y1) in degrees with:
        - 0° = up (negative y direction)
        - +angle = clockwise
        Returns angle in [0, 360).
        """
        dx = x1 - x0
        dy = y1 - y0

        # If the point is identical, angle is undefined
        if np.isscalar(dx) and np.isscalar(dy) and dx == 0 and dy == 0:
            return np.nan

        # Start with standard: atan2(dy, dx) gives 0° on +x and CCW positive
        theta = (np.degrees(np.arctan2(dy, dx)) + 90.0) % 360.0  # 0° at up, still CCW-ish
        theta = (-theta) % 360.0  # make clockwise-positive
        return theta

    @staticmethod
    def rotational_average_sector(
        image, x0, y0,
        dtheta=360.0, theta0=0.0,
        interpolate_nans=True,
        r_rounding="round",
        return_counts=False,
    ):
        """
        Radial (rotational) average over a sector (wedge) of angular width dtheta (deg),
        centered at theta0 (deg), with:

        - 0° = up (toward decreasing y)
        - +angle = clockwise

        Example: top-right quadrant -> dtheta=90, theta0=45.
        """
        img = np.asarray(image, dtype=float)

        Y, X = np.indices(img.shape)
        dx = X - x0
        dy = Y - y0

        # Radius
        r = np.sqrt(dx**2 + dy**2)
        if r_rounding == "round":
            R = np.rint(r).astype(int)
        elif r_rounding == "floor":
            R = np.floor(r).astype(int)
        elif r_rounding == "ceil":
            R = np.ceil(r).astype(int)
        else:
            raise ValueError("r_rounding must be 'round', 'floor', or 'ceil'")

        # Angle in degrees, with 0=up and CW positive:
        # Start from arctan2(dy, dx) which is 0 on +x and CCW positive,
        # then convert to (0=up, CW positive) via:
        theta = (np.degrees(np.arctan2(dy, dx)) + 90.0) % 360.0
        theta = (-theta) % 360.0  # flip to CW-positive while keeping 0=up

        # Sector mask
        if dtheta >= 360.0:
            ang_mask = np.ones_like(theta, dtype=bool)
        else:
            d = ImageAnalyzer.angular_difference_deg(theta, theta0)
            ang_mask = np.abs(d) <= (dtheta / 2.0)

        max_r = int(np.nanmax(R))
        avg = np.full(max_r + 1, np.nan)
        counts = np.zeros(max_r + 1, dtype=int)

        for rr in range(max_r + 1):
            mask = (R == rr) & ang_mask
            vals = img[mask]
            if vals.size:
                avg[rr] = np.nanmean(vals)
                counts[rr] = np.count_nonzero(np.isfinite(vals))

        if interpolate_nans:
            avg = ImageAnalyzer.interpolate_nan_values(avg)

        return (avg, counts) if return_counts else avg


    @staticmethod
    def fit_2d_polynomial(data, roi_bounds=None, exclude_nan=False, degree=2):
        """
        Fits a 2D polynomial to the data, either excluding a region of interest (ROI)
        or ignoring NaN values.

        Parameters:
        data (np.ndarray): 2D array of shape (N1, M1).
        roi_bounds (tuple, optional): Bounds of the ROI in the format 
                                      (roi_x_start, roi_x_end, roi_y_start, roi_y_end).
                                      If None, the entire data is used.
        exclude_nan (bool): If True, the function ignores NaN values. Defaults to False.
        degree (int): Degree of the polynomial to fit.

        Returns:
        np.ndarray: 2D array with the 2D polynomial values.
        """
        N1, M1 = data.shape

        if exclude_nan:
            # Create a mask for NaN values
            mask = ~np.isnan(data)
        elif roi_bounds is not None:
            # Create a mask for the ROI
            roi_x_start, roi_x_end, roi_y_start, roi_y_end = roi_bounds
            mask = np.ones((N1, M1), dtype=bool)
            mask[roi_x_start:roi_x_end, roi_y_start:roi_y_end] = False
        else:
            # If no mask condition is applied, fit the entire dataset
            mask = np.ones((N1, M1), dtype=bool)

        # Extract the points based on the mask
        masked_data = data[mask]

        # Generate the coordinates for all points
        x_coords, y_coords = np.meshgrid(np.arange(N1), np.arange(M1), indexing='ij')
        x_coords_flat = x_coords.flatten()
        y_coords_flat = y_coords.flatten()

        # Generate the coordinates for the masked points
        x_coords_masked = x_coords[mask]
        y_coords_masked = y_coords[mask]

        # Fit a 2D polynomial
        input_pts = np.vstack((x_coords_masked, y_coords_masked)).T

        poly = PolynomialFeatures(degree)
        in_features = poly.fit_transform(input_pts)

        model = LinearRegression()
        model.fit(in_features, masked_data.flatten())

        # Predict values on the full grid
        full_grid_coords = np.vstack((x_coords_flat, y_coords_flat)).T
        full_in_features = poly.fit_transform(full_grid_coords)
        predicted_full_grid = model.predict(full_in_features)

        # Reshape the predicted values to the original grid shape
        predicted_data_full = predicted_full_grid.reshape(N1, M1)

        return predicted_data_full
    
    @staticmethod
    def apply_elliptical_mask(array, x0, y0, a, b, fill_value=np.nan, invert=False):
        """
        Apply an elliptical mask to a 2D array, setting values inside or outside the ellipse to 'fill_value'
        based on the invert parameter. Optionally, return only the mask as 1s and 0s.
        
        Parameters:
            array (np.ndarray): The input 2D array to mask.
            x0 (int): The x-coordinate of the ellipse's center.
            y0 (int): The y-coordinate of the ellipse's center.
            a (int): The semi-major axis of the ellipse.
            b (int): The semi-minor axis of the ellipse.
            fill_value (float or np.nan): The value to assign to elements inside or outside the ellipse.
            invert (bool): If True, mask the inside of the ellipse; otherwise, mask the outside.
            return_mask_only (bool): If True, return the mask as a 2D array of 1s and 0s; otherwise, return the masked array.
        
        Returns:
            np.ndarray: The masked array or the mask (1s and 0s) depending on 'return_mask_only'.
        """
        # Create an index grid
        Y, X = np.ogrid[:array.shape[0], :array.shape[1]]
        # Calculate the distance from the center adjusted for the ellipse
        dist_from_center = ((X - x0)**2 / a**2) + ((Y - y0)**2 / b**2)
        # Determine whether to mask inside or outside the ellipse
        if invert:
            mask = dist_from_center <= 1  # Mask inside
        else:
            mask = dist_from_center > 1  # Mask outside

        # Copy the original array to not modify it
        masked_array = np.copy(array).astype(float)
        # Apply the mask
        masked_array[mask] = fill_value
        return masked_array
    
    @staticmethod    
    def extract_subarray(array, x0, y0, n, ny=None):
        """
        n x n (or ny x n) window centred on pixel (x0, y0), zero-padded where
        it runs off the image.
        """
        if ny is None:
            ny = n
        x0, y0 = int(x0), int(y0)
        N, M = array.shape

        # Clip center to nearest valid pixel
        x0 = np.clip(x0, 0, M - 1)
        y0 = np.clip(y0, 0, N - 1)

        row_start = y0 - ny // 2
        row_end   = row_start + ny
        col_start = x0 - n // 2
        col_end   = col_start + n

        row_start_clip = max(0, row_start)
        row_end_clip   = min(N, row_end)
        col_start_clip = max(0, col_start)
        col_end_clip   = min(M, col_end)

        subarray = array[row_start_clip:row_end_clip, col_start_clip:col_end_clip]

        pad_top    = row_start_clip - row_start
        pad_bottom = row_end - row_end_clip
        pad_left   = col_start_clip - col_start
        pad_right  = col_end - col_end_clip

        subarray = np.pad(
            subarray,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0
        )

        return subarray

    @staticmethod
    def add_lineout_to_imshow(ax, x, y, x_lo, y_lo, axlims, normalize=True, style_dict=None):
        """
        Plots a scaled line on the given Axes, using x for the horizontal coordinate
        and a rescaled version of x_lo for the vertical coordinate.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes on which to draw the plot.
        x : array-like
            The x-coordinates to plot.
        y : array-like
            The x-coordinates to plot.
        x_lo : array-like
            Data used to determine the line's vertical extent. It will be rescaled
            based on its max value and the provided axlims.
        y_lo : array-like
            Data used to determine the line's vertical extent. It will be rescaled
            based on its max value and the provided axlims.
        axlims : float
            A scaling factor applied after normalizing x_lo.
        style_dict : dict, optional
            Dictionary of style parameters. Recognized keys include:
            - 'color' (default: 'k')
            - 'linestyle' (default: '-')
            - 'alpha' (default: 0.75)

        Returns
        -------
        lines : list of matplotlib.lines.Line2D
            List of Line2D objects that were plotted (the return value of ax.plot).
        """
        if style_dict is None:
            style_dict = {}
        color = style_dict.get('color', 'k')
        linestyle = style_dict.get('linestyle', '-')
        alpha = style_dict.get('alpha', 0.75)
        
        if normalize:
            max_x = np.nanmax(x_lo)
            max_y = np.nanmax(y_lo)
        else:
            max_x = 1.0
            max_y = 1.0
            
        
        # Plot the lines
        line_x = ax.plot(x, 0.25*axlims*x_lo/max_x-axlims, 
                        color=color, linestyle=linestyle, alpha=alpha)
        line_y = ax.plot(0.25*axlims*y_lo/max_y-axlims, y, 
                        color=color, linestyle=linestyle, alpha=alpha)
        
        return line_x, line_y

    @staticmethod
    def overlay_fitted_curve(ax, return_dict, curve_function, axlims, style_dict=None):
        fitted_x = curve_function(return_dict['x'], *return_dict['fit_x']['params'])
        fitted_y = curve_function(return_dict['y'], *return_dict['fit_y']['params'])

        x = return_dict['x']
        y = return_dict['y']
        x_lo = return_dict['x_lo']
        y_lo = return_dict['y_lo']

        max_x = np.nanmax([x_lo, fitted_x])
        max_y = np.nanmax([y_lo, fitted_y])
        line_x, line_y = ImageAnalyzer.add_lineout_to_imshow(ax, x, y, x_lo/max_x, y_lo/max_y, axlims, normalize=False, style_dict=style_dict)
        line_fitx, line_fity = ImageAnalyzer.add_lineout_to_imshow(ax, x, y, fitted_x/max_x, fitted_y/max_y, axlims, normalize=False, style_dict={'color' : 'r'})
        return line_x, line_y


    
class ScaledImageAnalyzer(ImageAnalyzer):
    """
    Base class for scaled image analysis. 
    Derive from this class to customize analysis steps for specific image types.
    """

    def __init__(
        self,
        diagnostic: Optional[str] = None,
        file_ext: Optional[str] = None,
        analyzer_dict: Optional[Dict] = None,
        display_dict: Optional[Dict] = None,
        output_diagnostic: Optional[str] = None,
        output_file_ext: Optional[str] = None,
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

    def load_data(self, filename):
        file_ext = os.path.splitext(filename)[-1]
        if 'png' in file_ext:
            data = ni_imread.read_imaq_image('%s' % filename)

            with open('%s.txt' % filename[:-4]) as f:
                lines = f.readlines()
            scale_min = float(lines[1].split(' ')[2])
            scale_max = float(lines[2].split(' ')[2])

            return data * (scale_max - scale_min) / (2**16 - 1) + scale_min
        
        elif 'npy' in file_ext:
                return np.load( filename )