# E-beam profile and magspec allE in one pass per shot.
#
# MagSpecAllEAnalyzer needs the e-beam vertical angle of the same shot
# (it picks the trajectory calibration). In batch that is two analyze_scan
# passes: EBeamProfileAnalyzer writes its columns, then MagSpecAllEAnalyzer
# reads them through analyzer_dict['ebeam_columns']. Here both run on each
# shot in turn, with the same calls, so the outputs are the same; this is
# what live (per-shot) analysis needs, and it opens each shot once.

import numpy as np
import matplotlib.pyplot as plt

from PyGEECSPlotter.analyzers.multi_diagnostic_analyzer import MultiDiagnosticAnalyzer
from PyGEECSPlotter.analyzers.magspec.ebeam_profile_analysis import EBeamProfileAnalyzer
from PyGEECSPlotter.analyzers.magspec.magspec_alle_analysis import MagSpecAllEAnalyzer, MAGSPEC_CAMERAS

# the e-beam scalars MagSpecAllEAnalyzer reads (see MagSpecAllEAnalyzer._ebeam_columns)
EBEAM_ANGLE_STATS = [f'{stat} {ax} [mrad]' for stat in ('peak angle', 'mean angle', 'std div', 'fwhm div')
                     for ax in 'xy']


class MagSpecEBeamAnalyzer(MultiDiagnosticAnalyzer):
    """
    ``EBeamProfileAnalyzer`` then ``MagSpecAllEAnalyzer`` on each shot.

    Parameters
    ----------
    calib_dir, day, bg_dir :
        As for both analyzers (``...\\Calibrations\\ESMCalib``, the
        calibration day, the day's ``analysis`` folder with the averaged
        backgrounds).
    analysis_label : str
        The label the result columns get (e.g. ``'online'``); the magspec
        reads the e-beam angles under the same names as in batch,
        ``'<ebeam_diagnostic> peak angle y [mrad] <label>'``.
    ebeam_diagnostic, magspec_diagnostic, xray_diagnostic : str
        Output names (folders under ``analysis/ScanNNN/`` and column
        prefixes), as in the batch pipeline.
    ebeam_camera : str
        The e-beam profile camera.
    ebeam_dict, magspec_dict : dict, optional
        ``analyzer_dict`` of each analyzer (``ebeam_columns`` is set here).
    display_dict : dict, optional
        For the per-shot view: ``cmap``, ``vmax``, ``xlims``, ``figsize``,
        ``mmt_bin`` (momentum columns averaged per displayed pixel, default
        4), ``info`` (True: MATLAB infoE figure instead, ~0.8 s).

    ``results`` are named ``'<ebeam_diagnostic> <name>'`` and
    ``'<magspec_diagnostic> <name>'``: use ``column_prefix=''`` (LiveJob /
    analyze_scan) so the columns match a batch run. ``data`` is the allE
    image; ``aux`` is the magspec ``aux`` plus ``'ebeam_image'`` and
    ``'ebeam_aux'``. Writes both analyzers' per-shot files.
    """

    def __init__(self, calib_dir, day, bg_dir=None, analysis_label='', ebeam_diagnostic='py-ebeam',
                 magspec_diagnostic='py-magspec', xray_diagnostic=None,
                 ebeam_camera='CAM-TEA-EBeam_Profile', ebeam_dict=None, magspec_dict=None,
                 display_dict=None):
        self.ebeam = EBeamProfileAnalyzer(calib_dir, day, bg_dir=bg_dir, diagnostic=ebeam_camera,
                                          output_diagnostic=ebeam_diagnostic, analyzer_dict=ebeam_dict)
        self.alle = MagSpecAllEAnalyzer(
            calib_dir, day, bg_dir=bg_dir, output_diagnostic=magspec_diagnostic,
            xray_diagnostic=xray_diagnostic or f'{magspec_diagnostic}-xray',
            ebeam_diagnostic=ebeam_diagnostic,
            analyzer_dict={**(magspec_dict or {}), 'ebeam_columns': (ebeam_diagnostic, analysis_label)})
        subs = dict(self.alle.sub_analyzers)
        subs[ebeam_camera] = self.ebeam
        super().__init__(subs, output_diagnostic=magspec_diagnostic, output_file_ext='.png',
                         display_dict=display_dict)
        self.analysis_label = analysis_label
        self.ebeam_diagnostic = ebeam_diagnostic
        self.magspec_diagnostic = magspec_diagnostic
        self.ebeam_camera = ebeam_camera
        self.thread_safe = True

    def check_backgrounds(self):
        """Load both analyzers' default backgrounds now (from ``bg_dir``), so a
        missing file fails here, with its name, instead of on every shot.
        Raises FileNotFoundError / ValueError."""
        errors = []
        for what, analyzer in (('magspec', self.alle), ('e-beam profile', self.ebeam)):
            try:
                analyzer.default_background()
            except (FileNotFoundError, ValueError, OSError) as err:
                errors.append(f'{what}: {err}')
        if errors:
            raise FileNotFoundError('Backgrounds missing (run the background scans first):\n  '
                                    + '\n  '.join(errors))

    def live_payload(self, scan, shot, data, results, aux, mmt_bin=4):
        """A compact, picklable summary of one shot for a live display in
        another process: allE density [pC/mrad/(GeV/c)] averaged over
        ``mmt_bin`` momentum columns (float32), its extent, the shot's
        spectrum [pC/GeV] on its own momentum axis and on the fixed grid
        (``aux['p']``, for waterfalls), and the scalars (``results``)."""
        if data is None:
            return {'kind': 'shot', 'scan': scan, 'shot': shot, 'image': None,
                    'results': dict(results or {})}
        aux = aux or {}
        n_ang, n_mmt = data.shape
        mmt = np.asarray(aux.get('momentum', np.linspace(0.01, 5.0, n_mmt)), float)
        ang = np.asarray(aux.get('angle', np.linspace(-1.3, 1.3, n_ang)), float)
        dm, da = mmt[1] - mmt[0], ang[1] - ang[0]
        k = max(1, int(mmt_bin))
        n = n_mmt // k * k
        dens = (1e-6 * data[:, :n].reshape(n_ang, n // k, k).mean(axis=2) / dm / da).astype(np.float32)
        return {
            'kind': 'shot', 'scan': scan, 'shot': shot, 'image': dens,
            'extent': (mmt[0] - dm / 2, mmt[n - 1] + dm / 2, ang[0] - da / 2, ang[-1] + da / 2),
            'mmt': mmt.astype(np.float32), 'spectrum': (1e-6 * data.sum(axis=0) / dm).astype(np.float32),
            'grid': np.asarray(aux.get('p', []), np.float32),
            'grid_spectrum': np.asarray(aux.get('p_lo', []), np.float32),
            'results': {k_: float(v) for k_, v in (results or {}).items()
                        if isinstance(v, (int, float, np.integer, np.floating))},
        }

    def register_with_scan(self, scan, remove_missing_files=False):
        # missing magspec cameras are handled in the analysis (as in MATLAB)
        super().register_with_scan(scan, remove_missing_files=remove_missing_files)

    def _ebeam_name(self, name):
        return ' '.join(s for s in (self.ebeam_diagnostic, name, self.analysis_label) if s)

    def analyze_data(self, data, bg=None, context=None):
        context = dict(context or {})
        e_img, e_res, e_aux = self.ebeam.analyze_data(data.get(self.ebeam_camera), context=context)
        # the e-beam columns as batch has them in the sfile: NaN for a shot
        # without an e-beam image (the merge leaves them empty)
        for name in EBEAM_ANGLE_STATS:
            context[self._ebeam_name(name)] = e_res.get(name, np.nan)
        for name, value in e_res.items():
            context[self._ebeam_name(name)] = value
        m_data = {name: data.get(name) for name in MAGSPEC_CAMERAS}
        m_img, m_res, m_aux = self.alle.analyze_data(m_data, bg=bg, context=context)
        results = {f'{self.ebeam_diagnostic} {k}': v for k, v in e_res.items()}
        results.update({f'{self.magspec_diagnostic} {k}': v for k, v in m_res.items()})
        aux = dict(m_aux or {})
        aux['ebeam_image'], aux['ebeam_aux'] = e_img, e_aux
        return m_img, results, aux

    def write_analyzed_data(self, data, analysis_dir, scan, shot_num, context=None, aux=None):
        """Both analyzers' files (``<ebeam_diagnostic>/...``, ``<magspec_diagnostic>[-interpSpec,
        -interpDiv, -xray]/...``), as the two batch passes write them."""
        aux = aux or {}
        if aux.get('ebeam_image') is not None:
            self.ebeam.write_analyzed_data(aux['ebeam_image'], analysis_dir, scan, shot_num,
                                           context=context)
        self.alle.write_analyzed_data(data, analysis_dir, scan, shot_num, context=context, aux=aux)

    def display_data(self, data, return_dict=None, title=None, fig=None, ax=None, aux=None):
        """allE charge density and the spectrum below it (fast: ``imshow`` of the
        image averaged over ``mmt_bin`` momentum columns). ``display_dict['info']``:
        the MATLAB infoE figure instead."""
        if data is None:
            return None, None
        dd = self.display_dict
        if dd.get('info'):
            res = {k.split(' ', 1)[1]: v for k, v in (return_dict or {}).items()
                   if k.startswith(self.magspec_diagnostic + ' ')}
            return self.alle.display_data(data, return_dict=res, title=title, aux=aux)
        aux = aux or {}
        n_ang, n_mmt = data.shape
        mmt = np.asarray(aux.get('momentum', np.linspace(0.01, 5.0, n_mmt)), float)
        ang = np.asarray(aux.get('angle', np.linspace(-1.3, 1.3, n_ang)), float)
        dm, da = mmt[1] - mmt[0], ang[1] - ang[0]
        k = max(1, int(dd.get('mmt_bin', 4)))
        n = n_mmt // k * k
        dens = 1e-6 * data[:, :n].reshape(n_ang, n // k, k).mean(axis=2) / dm / da
        extent = [mmt[0] - dm / 2, mmt[n - 1] + dm / 2, ang[0] - da / 2, ang[-1] + da / 2]
        spec = 1e-6 * data.sum(axis=0) / dm
        spec = np.where(spec > 0, spec, np.nan)
        r = return_dict or {}
        m = self.magspec_diagnostic
        summary = (f"Q = {r.get(f'{m} charge_pC', np.nan):.3g} pC,  "
                   f"peak = {r.get(f'{m} peakMomentum_GeV/c', np.nan):.3g} GeV/c,  "
                   f"max = {r.get(f'{m} maxMomentum_GeV/c', np.nan):.3g} GeV/c,  "
                   f"eY = {r.get(f'{m} eYAngle_mrad', np.nan):.2f} mrad")
        vmax = dd.get('vmax') or np.nanmax(dens) or 1.0
        if fig is None:
            fig = plt.figure(figsize=dd.get('figsize', (9, 5.5)), constrained_layout=True)
        view = getattr(fig, '_magspec_view', None)
        if view is not None and view['shape'] == dens.shape and all(a in fig.axes for a in view['axes']):
            # same figure as last shot: update in place (redrawing it all costs ~0.5 s)
            ax_img, ax_sp = view['axes']
            if fig.get_layout_engine() is not None:
                fig.set_layout_engine('none')   # keep the layout of the first draw
            view['image'].set_data(dens)
            view['image'].set_extent(extent)
            view['image'].set_clim(0, vmax)
            view['line'].set_data(mmt, spec)
            ax_sp.relim()
            ax_sp.autoscale_view(scalex=not dd.get('xlims'))
        else:
            fig.clf()
            ax_img, ax_sp = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            im = ax_img.imshow(dens, aspect='auto', origin='lower', interpolation='nearest',
                               extent=extent, cmap=dd.get('cmap', 'jet'), vmin=0, vmax=vmax)
            fig.colorbar(im, ax=ax_img, label='pC/mrad/(GeV/c)')
            ax_img.set_ylabel('mrad')
            line, = ax_sp.semilogy(mmt, spec, 'r-', lw=1)
            ax_sp.set_xlabel('GeV/c')
            ax_sp.set_ylabel('pC/GeV')
            if dd.get('xlims'):
                ax_sp.set_xlim(*dd['xlims'])
            fig._magspec_view = {'shape': dens.shape, 'axes': (ax_img, ax_sp), 'image': im,
                                 'line': line}
        ax_img.set_title(title or '', fontsize=10)
        ax_sp.set_title(summary, fontsize=9)
        return fig, (ax_img, ax_sp)

    # LiveShotWatcher: the shot window is updated in place, not cleared per shot
    live_update = True
