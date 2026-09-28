# BELLA transverse e-beam profile (phosphor screen, CAM-TEA-EBeam_Profile).
# Port of the e-beam part of Kei Nakamura's live quickE script
# (bellaLiveMagspc3.m); numerics in PyGEECSPlotter.magspec.ebeam_profile.
# Standalone: run it on its own, or alongside MagSpecAllEAnalyzer, whose
# infoE figure shows its saved outputs.

import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from PyGEECSPlotter.diagnostic_analyzer import DiagnosticAnalyzer
from PyGEECSPlotter.magspec.ebeam_profile import (EBeamProfileCalibration, analyze_profile,
                                                  filter_position_from)
from PyGEECSPlotter.magspec.io import open_12bit_png, write_int_ac_png, write_table
from PyGEECSPlotter.navigation_utils import get_analysed_shot_save_path


class EBeamProfileAnalyzer(DiagnosticAnalyzer):
    """
    Charge-calibrated transverse e-beam profile from the phosphor screen.

    Per shot: 12-bit PNG, analysis ROI, Tony background subtraction,
    hot-pixel filter, rotation, screen mask and damage holes, filter-wheel
    and lanex counts-to-charge calibration. Then the MATLAB ``EBeamPrf``
    scalars.

    Parameters
    ----------
    calib_dir : str
        The ``Calibrations/ESMCalib`` directory.
    day : str
        Experiment day, e.g. ``'26_0521'`` (selects the dated camCalib file).
    bg_dir : str, optional
        Directory holding ``Scan###CAM-TEA-EBeam_Profile_averaged.png``
        (usually ``<day>/analysis``); used when ``analyze_data`` gets no
        ``bg``. As in MATLAB, the first ``Scan###`` with one is used.
    diagnostic : str
        Camera name (default ``'CAM-TEA-EBeam_Profile'``).
    output_diagnostic : str
        Output folder / column prefix. The default
        ``'CAM-TEA-EBeam_ProfileA'`` is the name MATLAB uses, so its saved
        outputs and these are interchangeable (``MagSpecAllEAnalyzer`` reads
        them for infoE).
    analyzer_dict : dict, optional
        ``filter_position``: filter-wheel position to use instead of the
        sfile's ``Phosphor-FW`` column (a missing column raises rather
        than guessing). ``hole_radius`` [mrad] and
        ``cap_length`` [m] as in bellaLiveMagspc3 (defaults 1.15, 0.03).
    display_dict : dict, optional
        ``figsize``, ``cmap``, ``vmax``.

    ``analyze_data`` returns ``(image [pC per pixel], results, aux)``:

    - ``results``: ``charge [pC]``, ``charge in hole [pC]``, peak / mean
      angle, fwhm / std divergence (x and y, mrad), ``mx fluence
      [pC/mrad2]``, ``saturation``. Same names as the MATLAB ``EBeamPrf``
      columns.
    - ``aux``: ``'x'`` / ``'x_lo'`` and ``'y'`` / ``'y_lo'`` (angle axes
      [mrad] and projected charge [pC/mrad], lineout family).
    """

    def __init__(self, calib_dir, day, bg_dir=None, diagnostic='CAM-TEA-EBeam_Profile',
                 output_diagnostic='CAM-TEA-EBeam_ProfileA', analyzer_dict=None,
                 display_dict=None):
        super().__init__(diagnostic=diagnostic, file_ext='.png', analyzer_dict=analyzer_dict,
                         display_dict=display_dict, output_diagnostic=output_diagnostic,
                         output_file_ext='.png')
        ad = self.analyzer_dict
        kwargs = {k: ad[k] for k in ('hole_radius', 'cap_length') if k in ad}
        self.calibration = EBeamProfileCalibration(calib_dir, day, **kwargs)
        self.bg_dir = bg_dir
        self._default_bg = None

    def load_data(self, filename):
        if not isinstance(filename, str) or not os.path.exists(filename):
            return None
        return open_12bit_png(filename)

    def default_background(self):
        """``Scan###<diagnostic>_averaged.png`` from ``bg_dir``."""
        if self._default_bg is None:
            if self.bg_dir is None:
                raise ValueError('no bg given and no bg_dir set')
            found = sorted(glob.glob(os.path.join(self.bg_dir, f'Scan*{self.diagnostic}_averaged.png')))
            if not found:
                raise FileNotFoundError(f'no Scan*{self.diagnostic}_averaged.png in {self.bg_dir}')
            self._default_bg = open_12bit_png(found[0])
        return self._default_bg

    def analyze_data(self, data, bg=None, context=None, analyzer_dict=None):
        if data is None:
            return None, {}, {}
        ad = {**self.analyzer_dict, **(analyzer_dict or {})}
        context = context or {}
        position = ad.get('filter_position')
        if position is None:
            position = filter_position_from(context)
        if bg is None:
            bg = self.default_background()
        r = analyze_profile(data, bg, self.calibration, position)
        img = np.nan_to_num(r.image)
        cal = self.calibration
        aux = {'x': r.x_mrad, 'x_lo': img.sum(axis=0) / cal.dmrad,
               'y': r.y_mrad, 'y_lo': img.sum(axis=1) / cal.dmrad}
        return img, dict(r.scalars), aux

    def display_data(self, data, return_dict=None, title=None, fig=None, ax=None):
        """Fluence [pC/mrad^2] on angle axes, with the 1" hole outline."""
        if data is None:
            return None, None
        dd = self.display_dict
        cal = self.calibration
        if fig is None or ax is None:
            fig, ax = plt.subplots(constrained_layout=True, figsize=dd.get('figsize', (5, 5)))
        m = ax.pcolormesh(cal.x_mrad, cal.y_mrad, data / cal.dmrad ** 2, shading='auto',
                          cmap=dd.get('cmap', 'jet'), vmin=0, vmax=dd.get('vmax'))
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(cal.hole_radius * np.cos(t), cal.hole_radius * np.sin(t), 'w--', lw=1)
        fig.colorbar(m, ax=ax, label='pC/mrad$^2$')
        ax.set_aspect('equal')
        ax.set_xlabel('x [mrad]')
        ax.set_ylabel('y [mrad]')
        label = title or ''
        if return_dict:
            label = f"{label}  {return_dict.get('charge in hole [pC]', np.nan):.3g} pC in hole".strip()
        ax.set_title(label, fontsize=9)
        return fig, ax

    def write_analyzed_data(self, data, analysis_dir, scan, shot_num, context=None):
        """``fBellaPhosSv`` outputs: integer-aC PNG (``'N aC/count'``) and
        the X / Y projection tables (mm, mrad, pC, pC/mrad) -- the same
        files MATLAB writes to ``CAM-TEA-EBeam_ProfileA``."""
        if data is None:
            return
        cal = self.calibration
        diag = self.output_diagnostic
        write_int_ac_png(get_analysed_shot_save_path(analysis_dir, diag, scan, shot_num, '.png'),
                         1e6 * data)
        for axis, mm, mrad, proj in (('X', cal.x_mm, cal.x_mrad, data.sum(axis=0)),
                                     ('Y', cal.y_mm, cal.y_mrad, data.sum(axis=1))):
            path = get_analysed_shot_save_path(analysis_dir, diag, scan, shot_num, '.txt', axis)
            write_table(path, ['mm', 'mrad', 'pC', 'pC/mrad'],
                        [mm, mrad, proj, proj / cal.dmrad], digits=5)

    def write_displayed_data(self, fig, analysis_dir, scan, shot_num):
        if fig is None:
            return
        path = get_analysed_shot_save_path(analysis_dir, self.output_diagnostic, scan, shot_num,
                                           '.png', '_disp')
        fig.savefig(path, dpi=150)
