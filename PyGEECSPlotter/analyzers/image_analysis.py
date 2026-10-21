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

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

import PyGEECSPlotter.utils.ni_imread as ni_imread
import PyGEECSPlotter.utils.analysis_2d_utils as a2d
import PyGEECSPlotter.utils.analysis_1d_utils as a1d
import PyGEECSPlotter.utils.plotting as gplt
from PyGEECSPlotter.utils.analysis_1d_utils import super_gaussian, get_lineout_width
from PyGEECSPlotter.utils.io_utils import save_lineouts_to_txt, read_scaled_png
from PyGEECSPlotter.utils.navigation_utils import get_analysed_shot_save_path
from PyGEECSPlotter.analyzers.diagnostic_analyzer import DiagnosticAnalyzer


class ImageAnalyzer(DiagnosticAnalyzer):
    """
    Base class for image analysis.
    Derive from this class to customize analysis steps for specific image types.
    """

    # Where coordinate (0, 0) goes when analyzer_dict has no 'origin':
    # 'centroid', 'center', 'pixel' or (x0, y0) (see coords_for_origin).
    DEFAULT_ORIGIN = 'centroid'

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
        self._set_extent(s)

        return s['data'], s['results'], s['lineouts']

    def _set_extent(self, s):
        """Give every analysed image an imshow extent: from the last
        coordinates a step set, else from analyzer_dict 'origin' / dx / dy."""
        if s['x'] is not None and len(s['x']) == s['data'].shape[1]                 and len(s['y']) == s['data'].shape[0]:
            s['results']['imshow_extent'] = self.get_imshow_extent(s['x'], s['y'])
            return
        if np.ndim(s['data']) != 2:
            return
        ad = s['ad']
        x, y, _, _ = self.coords_for_origin(
            s['data'], origin=ad.get('origin', self.DEFAULT_ORIGIN),
            dx=ad.get('dx', 1), dy=ad.get('dy', 1),
            round_centroid=ad.get('round_centroid', True))
        s['results']['imshow_extent'] = self.get_imshow_extent(x, y)

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
        """
        Show one image in physical coordinates.

        The extent comes from, in order: display_dict['extent'];
        return_dict['imshow_extent'] (set by analyze_data); or, for images
        that weren't analysed one by one (per-bin means, direct calls), the
        analyzer_dict 'origin' / 'dx' / 'dy' applied to `data` itself. So
        'axlims' always windows around the same origin the analysis used.

        display_dict keys: figsize, aspect, norm, cmap, interpolation, vmin,
        vmax, cbar_off, cbar_label, spatial_units, xtitle, ytitle,
        axlims (L, (Lx, Ly) or [x0, x1, y0, y1]; see apply_axlims), tick_dx,
        v_lines, h_lines, extent, and for lineout overlays:
        show_lineouts (default False), lineout_method ('center': rows /
        columns through the origin, averaged over Nlo; 'all': mean over the
        image), Nlo (default 2), lineout_amp (default 0.25), lineout_style
        (matplotlib line keywords).
        """
        if display_dict is None:
            display_dict = self.display_dict

        if fig is None or ax is None:
            fig, ax = plt.subplots(constrained_layout=True, figsize=display_dict.get('figsize', (6, 5)))

        extent = self._resolve_extent(data, return_dict, display_dict)
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

        self.apply_axlims(ax, display_dict.get('axlims', None), display_dict.get('tick_dx', None))

        for v_line in display_dict.get('v_lines', []):
            ax.axvline(x=v_line, color='k', linestyle='--')

        for h_line in display_dict.get('h_lines', []):
            ax.axhline(y=h_line, color='k', linestyle='--')

        if display_dict.get('show_lineouts', False):
            self._overlay_image_lineouts(ax, data, extent, display_dict)

        if title is not None:
            ax.set_title(title)

        return fig, ax

    def _resolve_extent(self, data, return_dict=None, display_dict=None):
        """The imshow extent display_data uses (see its docstring)."""
        display_dict = display_dict if display_dict is not None else (self.display_dict or {})
        if display_dict.get('extent') is not None:
            return np.asarray(display_dict['extent'], float)
        if return_dict is not None and return_dict.get('imshow_extent') is not None:
            return np.asarray(return_dict['imshow_extent'], float)
        ad = self.analyzer_dict or {}
        x, y, _, _ = self.coords_for_origin(
            data, origin=ad.get('origin', self.DEFAULT_ORIGIN), dx=ad.get('dx', 1), dy=ad.get('dy', 1),
            round_centroid=ad.get('round_centroid', True))
        return self.get_imshow_extent(x, y)

    def _overlay_image_lineouts(self, ax, data, extent, display_dict):
        """Lineouts of `data` through coordinate (0, 0) (or the centroid if
        that lies outside the image, e.g. origin='pixel'), drawn inside the
        current window."""
        data = np.asarray(data, float)
        ny, nx = data.shape
        x = np.linspace(extent[0], extent[1], nx + 1)
        y = np.linspace(extent[2], extent[3], ny + 1)
        x, y = 0.5*(x[1:] + x[:-1]), 0.5*(y[1:] + y[:-1])     # pixel centres
        origin = (self.analyzer_dict or {}).get('origin', self.DEFAULT_ORIGIN)
        if origin == 'pixel' or not (min(x) <= 0 <= max(x) and min(y) <= 0 <= max(y)):
            with np.errstate(invalid='ignore', divide='ignore'):
                i0, j0 = self._get_centroid(data)
            if not (np.isfinite(i0) and np.isfinite(j0)):
                i0, j0 = 0.5*nx, 0.5*ny
        else:
            i0, j0 = np.argmin(np.abs(x)) + 0.5, np.argmin(np.abs(y)) + 0.5
        x_lo, y_lo = self.get_lineouts(data, i0, j0,
                                       method=display_dict.get('lineout_method', 'center'),
                                       Nlo=display_dict.get('Nlo', 2))
        self.overlay_lineouts(ax, x, y, x_lo, y_lo,
                              amp=display_dict.get('lineout_amp', 0.25),
                              style_dict=display_dict.get('lineout_style', None))

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
        s['data'] = a2d.subtract_background(s['data'], bg, scale)

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

    def _step_lineouts(self, s, centroid_method=None, x0=None, y0=None,
                       centroid_thresh=0.85, lineout_method='center', Nlo=2):
        """Coordinates + lineouts. centroid_method (get_spatial_coords
        'center' / 'pixel' / 'manual' / 'centroid') defaults to the
        analyzer_dict 'origin'."""
        ad = s['ad']
        if centroid_method is None:
            x, y, x0, y0 = self.coords_for_origin(
                s['data'], origin=ad.get('origin', self.DEFAULT_ORIGIN),
                dx=ad.get('dx', 1), dy=ad.get('dy', 1), centroid_thresh=centroid_thresh,
                round_centroid=ad.get('round_centroid', True))
        else:
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
    # Analysis functions live in utils.analysis_2d_utils / analysis_1d_utils
    # and plotting helpers in utils.plotting; these aliases keep the old
    # ImageAnalyzer.<name> calls working.
    # -------------------------------------------------------------------

    roi_data = staticmethod(a2d.roi_data)
    compute_data_counts = staticmethod(a2d.compute_data_counts)
    calculate_misalignment = staticmethod(a2d.calculate_misalignment)
    compute_n_saturated = staticmethod(a2d.compute_n_saturated)
    apply_threshold = staticmethod(a2d.apply_threshold)
    remove_outliers = staticmethod(a2d.remove_outliers)
    get_spatial_coords = staticmethod(a2d.get_spatial_coords)
    _get_centroid = staticmethod(a2d.thresholded_centroid)
    get_lineouts = staticmethod(a2d.get_lineouts)
    display_imshow_lineouts = staticmethod(gplt.display_imshow_lineouts)
    adjust_data_to_fluence = staticmethod(a2d.adjust_data_to_fluence)
    get_imshow_extent = staticmethod(a2d.get_imshow_extent)
    coords_for_origin = staticmethod(a2d.coords_for_origin)
    apply_axlims = staticmethod(gplt.apply_axlims)
    overlay_lineouts = staticmethod(gplt.overlay_lineouts)
    fit_super_gaussian = staticmethod(a1d.fit_super_gaussian)
    clip_roi = staticmethod(a2d.clip_roi)
    median_filter = staticmethod(a2d.median_filter)
    centroid = staticmethod(a2d.centroid)
    find_beam_center = staticmethod(a2d.find_beam_center)
    apply_radial_mask = staticmethod(a2d.apply_radial_mask)
    compute_second_moments = staticmethod(a2d.compute_second_moments)
    compute_encircled_energy = staticmethod(a2d.compute_encircled_energy)
    rotate_around = staticmethod(a2d.rotate_around)
    rotational_average = staticmethod(a2d.rotational_average)
    interpolate_nan_values = staticmethod(a1d.interpolate_nan_values)
    angular_difference_deg = staticmethod(a1d.angular_difference_deg)
    angle_deg_from_points = staticmethod(a2d.angle_deg_from_points)
    rotational_average_sector = staticmethod(a2d.rotational_average_sector)
    fit_2d_polynomial = staticmethod(a2d.fit_2d_polynomial)
    apply_elliptical_mask = staticmethod(a2d.apply_elliptical_mask)
    extract_subarray = staticmethod(a2d.extract_subarray)
    add_lineout_to_imshow = staticmethod(gplt.add_lineout_to_imshow)
    overlay_fitted_curve = staticmethod(gplt.overlay_fitted_curve)


    
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
            return read_scaled_png(filename)
        
        elif 'npy' in file_ext:
                return np.load( filename )