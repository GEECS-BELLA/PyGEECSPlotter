# BELLA triangle-chamber magnetic spectrometer: full-range "allE" analysis.
# Python port of Kei Nakamura's MATLAB chain bellaMagspcTri.m ->
# bellaMagspecViewTri.m / fBellaSShotTri.m. The numerics live in
# PyGEECSPlotter.magspec; this module wires them into the
# MultiDiagnosticAnalyzer contract.

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PyGEECSPlotter.diagnostic_analyzer import DiagnosticAnalyzer
from PyGEECSPlotter.magspec.calibration import MAGSPEC_CAMERAS, MagSpecCalibration
from PyGEECSPlotter.magspec.infoe import draw_infoe
from PyGEECSPlotter.magspec.io import open_12bit_png, read_int_ac_png, write_int_ac_png, write_table
from PyGEECSPlotter.magspec.matlab_compat import interp1
from PyGEECSPlotter.magspec.pipeline import run_alle
from PyGEECSPlotter.magspec.stage1 import ebeam_y_angle
from PyGEECSPlotter.multi_diagnostic_analyzer import MultiDiagnosticAnalyzer
from PyGEECSPlotter.navigation_utils import get_analysed_shot_save_path

FIELD_COLUMN = 'HALLPROBE-TEA-MAGSPEC Field'
ICT_COLUMN = 'TurboICT charge [pc]'


class _MagSpecCamera(DiagnosticAnalyzer):
    """Loader for one magspec camera: 12-bit PNG, MATLAB-style bit shift."""

    def load_data(self, filename):
        if not isinstance(filename, str) or not os.path.exists(filename):
            return None
        return open_12bit_png(filename)


class MagSpecAllEAnalyzer(MultiDiagnosticAnalyzer):
    """
    Full-range ("allE") electron spectrum from the ten BELLA magspec cameras.

    Per shot: background subtraction, rotation, hot-pixel removal, lanex
    charge calibration and vignetting / incidence corrections for each
    camera; resampling onto uniform angle and momentum axes for the front
    (cams A-C) and side (cams D, G-L) screens; x-ray background removal
    from the camera-A front lineout; stitching into one 256 x 2048 image.

    Parameters
    ----------
    calib_dir : str
        The ``Calibrations/ESMCalib`` directory (cam, lanex and trajectory
        calibration files).
    day : str
        Experiment day, e.g. ``'26_0521'``; picks the newest dated
        calibration files on or before this day.
    bg_dir : str, optional
        Directory holding ``Scan###<camera>_averaged.png`` backgrounds
        (usually ``<day>/analysis``). Used when ``analyze_data`` gets no
        ``bg``. As in MATLAB, the first ``Scan###`` with a MagSpecB
        background is used for all cameras.
    analyzer_dict : dict, optional
        ``roi`` : momentum window [GeV/c], default ``(0.01, 5.0)``.
        ``emulate_quantization`` : reproduce MATLAB's file round-trip
        between the two stages (default True, for bit-level parity).
        ``angle_cuts`` : (max angle, max div, max |mean-peak|) [mrad] for
        the EBeam-profile input-angle estimate, default (1.8, 1.32, 0.35).
        ``ey_angle`` : force the input angle [mrad] instead.
        ``momentum_grid`` : fixed momentum axis [GeV/c] for ``aux['p']``
        (default 1024 points across ``roi``).
    calibration_kwargs : dict, optional
        Extra arguments for :class:`MagSpecCalibration` (``crrnt``,
        ``mgs_chg``, ``lanex_file``).
    ebeam_diagnostic : str or None
        Folder under ``analysis/ScanNNN/`` holding saved EBeam-profile
        outputs, for the left panel of the infoE figure. The default
        ``'CAM-TEA-EBeam_ProfileA'`` is where both MATLAB and
        ``EBeamProfileAnalyzer`` write them. None, or missing files, leaves
        that panel blank.

    ``display_data`` draws the MATLAB infoE summary figure (e-beam profile,
    allE charge density and spectrum, front-screen x-ray, info text).
    ``display_dict={'info': False}``, or passing ``fig``/``ax`` (as the
    image-grid displayers do), gives the simple allE + spectrum view.
    ``write_displayed_data`` saves it as ``ScanNNN_<diag>infoE_SSS.png``.

    ``analyze_data`` returns ``(allE, results, aux)``:

    - ``allE``: 256 x 2048 image [aC] on ``aux['momentum']`` x ``aux['angle']``.
    - ``results``: charge, energy, momentum and angle statistics, x-ray
      base, per-camera charge and saturation, input angle.
    - ``aux``: ``'allESpec'`` / ``'allEDiv'`` DataFrames (the MATLAB text
      files), the axes, and the stage-1 ``'highE'`` / ``'lowE'`` images.
      Also lineout-family pairs for the scan displayers: ``'p'`` /
      ``'p_lo'`` (spectrum [pC/GeV] on the fixed momentum grid) and
      ``'angle'`` / ``'angle_lo'`` (divergence [fC/mrad]).

    The sfile row (``context``) must contain ``HALLPROBE-TEA-MAGSPEC Field``
    and the ``EBeamPrf ...`` angle / divergence columns.
    """

    def __init__(self, calib_dir, day, bg_dir=None, analyzer_dict=None,
                 display_dict=None, output_diagnostic='MagSpecAllE',
                 calibration_kwargs=None, ebeam_diagnostic='CAM-TEA-EBeam_ProfileA'):
        cams = {name: _MagSpecCamera(diagnostic=name, file_ext='.png') for name in MAGSPEC_CAMERAS}
        super().__init__(
            inputs=[(name, '.png') for name in MAGSPEC_CAMERAS],
            sub_analyzers=cams,
            output_diagnostic=output_diagnostic,
            output_file_ext='.png',
            analyzer_dict=analyzer_dict,
            display_dict=display_dict,
        )
        self.calibration = MagSpecCalibration(calib_dir, day, **(calibration_kwargs or {}))
        self.bg_dir = bg_dir
        self.ebeam_diagnostic = ebeam_diagnostic
        self._default_bg = None
        # analyze_scan does not hand aux to display/write; keep the last shot's
        self._last_aux = None

    def register_with_scan(self, scan, remove_missing_files=False):
        # a missing camera is handled inside the analysis, as in MATLAB
        super().register_with_scan(scan, remove_missing_files=remove_missing_files)

    def default_background(self):
        """Backgrounds from ``bg_dir`` (``fBellaBgV03`` lookup rule)."""
        if self._default_bg is None:
            if self.bg_dir is None:
                raise ValueError('no bg given and no bg_dir set')
            found = sorted(glob.glob(os.path.join(self.bg_dir, 'Scan*MagSpecB_averaged.png')))
            if not found:
                raise FileNotFoundError(f'no Scan*MagSpecB_averaged.png in {self.bg_dir}')
            scan_s = os.path.basename(found[0])[4:7]
            self._default_bg = {
                name: open_12bit_png(os.path.join(self.bg_dir, f'Scan{scan_s}{name}_averaged.png'))
                for name in MAGSPEC_CAMERAS
            }
        return self._default_bg

    def momentum_grid(self, analyzer_dict=None):
        """Common momentum grid [GeV/c] for ``aux['p']``: ``analyzer_dict
        ['momentum_grid']`` if given, else 1024 points across ``roi``."""
        ad = analyzer_dict if analyzer_dict is not None else self.analyzer_dict
        if ad.get('momentum_grid') is not None:
            return np.asarray(ad['momentum_grid'], float)
        roi = ad.get('roi', (0.01, 5.0))
        return np.linspace(roi[0], roi[1], 1024)

    def analyze_data(self, data, bg=None, context=None, analyzer_dict=None):
        ad = {**self.analyzer_dict, **(analyzer_dict or {})}
        context = context or {}
        if all(data.get(name) is None for name in MAGSPEC_CAMERAS):
            return None, {}, {}
        bg_all = self.default_background() if bg is None else None
        bgs = []
        for name in MAGSPEC_CAMERAS:
            b = self._bg_for(bg, name) if bg is not None else bg_all[name]
            if b is None:
                b = self.default_background()[name]
            bgs.append(np.asarray(b, float))
        raw = [data.get(name) for name in MAGSPEC_CAMERAS]

        field_T = 1e-3 * float(context[FIELD_COLUMN])
        if ad.get('ey_angle') is not None:
            ey = float(ad['ey_angle'])
        else:
            ey = ebeam_y_angle(context, *ad.get('angle_cuts', (1.8, 1.32, 0.35)))

        s1, s2 = run_alle(raw, bgs, self.calibration, field_T, ey,
                          roi=tuple(ad.get('roi', (0.01, 5.0))),
                          emulate_quantization=ad.get('emulate_quantization', True))

        results = dict(s2.scalars)
        results['eYAngle_mrad'] = ey
        results['ImgMissed'] = s1.missing
        for name, chg, sat in zip(MAGSPEC_CAMERAS, s1.cam_charge_pC, s1.saturated):
            cam = name[-1]
            results[f'cam{cam}_pC'] = chg
            results[f'cam{cam} sat'] = sat
        # The allE momentum axis follows each shot's field and input angle,
        # so also give the spectrum on a fixed grid (lineout-family keys) for
        # scan-level waterfalls.
        grid = self.momentum_grid(ad)
        p_lo = interp1(s2.spec['Momentum_GeV/c'].to_numpy(), s2.spec['ChargeDen_pC/GeV/c'].to_numpy(), grid)
        aux = {
            'allESpec': s2.spec, 'allEDiv': s2.div,
            'momentum': s2.alle_mmt, 'angle': s2.angle,
            'highE': s1.high, 'lowE': s1.low,
            'p': grid, 'p_lo': np.nan_to_num(p_lo, nan=0.0),
            'angle_lo': s2.div['ChargeDen_fC/mrad'].to_numpy(),
            # everything the infoE figure needs
            'infoE': {
                'mmt': s2.mmtR, 'ya': s2.angle, 'density': s2.density, 'accp': s2.accpR,
                'spectrum': s2.spectrum, 'gap': s2.gap, 'scalars': dict(s2.scalars),
                'xray_img': s1.front_img, 'xray_x_mm': s1.front_x[0], 'xray_y_mm': s1.front_y[0],
                'ey_angle': ey, 'ict_pC': float(context.get(ICT_COLUMN, np.nan)),
                'scan': context.get('scan'), 'shot': context.get('Shotnumber'),
                # the e-beam files are only read when infoE is drawn
                'ebeam_context': {k: context.get(k) for k in
                                  ['scan', 'Shotnumber'] + [f'{n} file_list' for n in MAGSPEC_CAMERAS]},
                'roi': tuple(ad.get('roi', (0.01, 5.0))),
            },
        }
        self._last_aux = aux
        return s2.alle, results, aux

    def _load_ebeam(self, context):
        """Saved EBeam-profile image [pC] and angle axes for this shot, or None.

        The day's top directory is taken from a camera ``file_list`` path
        (``<top>/scans/ScanNNN/<camera>/...``), else from ``bg_dir``
        (``<top>/analysis``)."""
        if not self.ebeam_diagnostic or 'scan' not in context or 'Shotnumber' not in context:
            return None
        cam_path = next((context.get(f'{n} file_list') for n in MAGSPEC_CAMERAS
                         if isinstance(context.get(f'{n} file_list'), str)), None)
        if cam_path is not None:
            top = cam_path
            for _ in range(4):
                top = os.path.dirname(top)
        elif self.bg_dir:
            top = os.path.dirname(os.path.normpath(self.bg_dir))
        else:
            return None
        scan, shot, diag = int(context['scan']), int(context['Shotnumber']), self.ebeam_diagnostic
        folder = os.path.join(top, 'analysis', f'Scan{scan:03d}', diag)
        png = os.path.join(folder, f'Scan{scan:03d}_{diag}_{shot:03d}.png')
        tables = [os.path.join(folder, f'Scan{scan:03d}_{diag}{ax}_{shot:03d}.txt') for ax in 'XY']
        if not all(os.path.exists(p) for p in [png] + tables):
            return None
        x, y = (pd.read_csv(p, sep='\t')['mrad'].to_numpy() for p in tables)
        return {'image': 1e-6 * read_int_ac_png(png), 'x': x, 'y': y}   # aC -> pC

    def display_data(self, data, return_dict=None, title=None, fig=None, ax=None):
        """infoE summary figure by default; the simple allE + spectrum view
        with ``display_dict={'info': False}`` or when ``fig``/``ax`` are given."""
        if data is None:
            return None, None
        info = (self._last_aux or {}).get('infoE')
        if self.display_dict.get('info', True) and info is not None and fig is None and ax is None:
            info = dict(info)
            info['ebeam'] = self._load_ebeam(info.pop('ebeam_context'))
            return draw_infoe(**info, fontsize=self.display_dict.get('fontsize', 10),
                              figsize=self.display_dict.get('figsize', (20, 6.67)))
        return self._display_simple(data, return_dict=return_dict, title=title, fig=fig, ax=ax)

    def _display_simple(self, data, return_dict=None, title=None, fig=None, ax=None):
        """allE charge density [pC/mrad/(GeV/c)] with the spectrum below."""
        dd = self.display_dict
        n_ang, n_mmt = data.shape
        last = self._last_aux or {}
        mmt = np.asarray(last['momentum']) if len(last.get('momentum', ())) == n_mmt             else self._last_momentum(n_mmt)
        ang = np.asarray(last['angle']) if len(last.get('angle', ())) == n_ang             else np.linspace(-1.3, 1.3, n_ang)
        dens = 1e-6 * data / (mmt[1] - mmt[0]) / (ang[1] - ang[0])
        if fig is None or ax is None:
            fig, ax = plt.subplots(2, 1, figsize=dd.get('figsize', (10, 5)), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})
        ax = np.atleast_1d(ax)
        m = ax[0].pcolormesh(mmt, ang, dens, cmap=dd.get('cmap', 'jet'), shading='auto',
                             vmin=dd.get('vmin', 0), vmax=dd.get('vmax'))
        # attach to all axes so the shared momentum axis stays aligned
        fig.colorbar(m, ax=list(ax), label='pC/mrad/(GeV/c)', location='right', shrink=0.9)
        ax[0].set_ylabel('mrad')
        if title:
            ax[0].set_title(title)
        if len(ax) > 1:
            ax[1].semilogy(mmt, 1e-6 * data.sum(axis=0) / (mmt[1] - mmt[0]), 'r-')
            ax[1].set_xlabel('GeV/c')
            ax[1].set_ylabel('pC/GeV')
            if return_dict:
                ax[1].set_title(f"Q = {return_dict.get('charge_pC', np.nan):.3g} pC, "
                                f"peak = {return_dict.get('peakMomentum_GeV/c', np.nan):.3g} GeV/c, "
                                f"max = {return_dict.get('maxMomentum_GeV/c', np.nan):.3g} GeV/c",
                                fontsize=9)
        return fig, ax

    def _last_momentum(self, n):
        roi = self.analyzer_dict.get('roi', (0.01, 5.0))
        return np.linspace(roi[0], roi[1], n)

    def write_analyzed_data(self, data, analysis_dir, scan, shot_num, context=None, aux=None):
        """allE png (integer aC, 'N aC/count' comment) + allESpec/allEDiv
        tables, in the MATLAB formats."""
        if data is None:
            return
        diag = self.output_diagnostic
        write_int_ac_png(get_analysed_shot_save_path(analysis_dir, diag, scan, shot_num, '.png'), data)
        aux = aux or self._last_aux
        if aux:
            spec, div = aux['allESpec'], aux['allEDiv']
            write_table(get_analysed_shot_save_path(analysis_dir, diag, scan, shot_num, '.txt', 'Spec'),
                        list(spec.columns), [spec[c] for c in spec.columns])
            write_table(get_analysed_shot_save_path(analysis_dir, diag, scan, shot_num, '.txt', 'Div'),
                        list(div.columns), [div[c] for c in div.columns])

    def write_displayed_data(self, fig, analysis_dir, scan, shot_num):
        """Save the displayed figure (infoE by default) as
        ``<output_diagnostic>/ScanNNN_<diag>infoE_SSS.png``."""
        if fig is None:
            return
        path = get_analysed_shot_save_path(analysis_dir, self.output_diagnostic, scan, shot_num,
                                           '.png', 'infoE')
        fig.savefig(path, dpi=self.display_dict.get('dpi', 120))
