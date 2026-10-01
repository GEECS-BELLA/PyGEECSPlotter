# Reader for allE outputs already written by MagSpecAllEAnalyzer
# (write_analyzed=True). Scan-level displayers can run on it instead of on
# the full ten-camera analysis: no calibration, no image processing, just
# file reads -- and only the files a given view needs.

import os
import threading
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PyGEECSPlotter.diagnostic_analyzer import DiagnosticAnalyzer
try:
    import pw_py_magspec  # noqa: F401  (private BellaCenter package)
except ImportError as err:
    raise ImportError(
        "magspec_alle_reader needs the private 'pw-py-magspec' package (BellaCenter/PW-py-magspec). "
        "Install it with: pip install git+https://github.com/BellaCenter/PW-py-magspec.git"
    ) from err

from pw_py_magspec.infoe import draw_infoe
from pw_py_magspec.io import read_int_ac_png
from pw_py_magspec.matlab_compat import interp1
from PyGEECSPlotter.navigation_utils import get_analysed_shot_save_path

# which saved tables each `load` mode reads, besides (or instead of) the image
_MODES = {
    'image': (True, ()),
    'spec': (False, ('Spec',)),
    'div': (False, ('Div',)),
    'all': (True, ('Spec', 'Div')),
    'infoE': (True, ('Spec', 'Div')),   # plus the optional Info / x-ray / e-beam files
}

# sfile scalars (as written by MagSpecAllEAnalyzer) the infoE text panels show
_INFOE_SCALARS = ('charge_pC', 'peakMomentum_GeV/c', 'fwhmMomentum_GeV/c', 'mmtRes_%',
                  'peakAngle_mrad', 'fwhmAngle_mrad', 'maxMomentum_GeV/c',
                  'energyPeakMmt_GeV/c')
ICT_COLUMN = 'TurboICT charge [pc]'


def load_ebeam_profile(top, scan, shot, diag):
    """Saved EBeam-profile image [pC] and angle axes of one shot, or None.

    ``top`` is the day directory (holding ``analysis/``); ``diag`` the folder
    under ``analysis/ScanNNN/`` (``EBeamProfileAnalyzer`` or MATLAB output)."""
    if not diag:
        return None
    folder = os.path.join(top, 'analysis', f'Scan{scan:03d}', diag)
    png = os.path.join(folder, f'Scan{scan:03d}_{diag}_{shot:03d}.png')
    tables = [os.path.join(folder, f'Scan{scan:03d}_{diag}{ax}_{shot:03d}.txt') for ax in 'XY']
    if not all(os.path.exists(p) for p in [png] + tables):
        return None
    x, y = (pd.read_csv(p, sep='\t')['mrad'].to_numpy() for p in tables)
    return {'image': 1e-6 * read_int_ac_png(png), 'x': x, 'y': y}   # aC -> pC


class MagSpecAllEReader(DiagnosticAnalyzer):
    """
    Load saved ``MagSpecAllEAnalyzer`` outputs instead of recomputing them.

    ``MagSpecAllEAnalyzer`` with ``write_analyzed=True`` writes, per shot,
    under ``analysis/ScanNNN/<output_diagnostic>/``::

        ScanNNN_<diag>_SSS.png       allE image (integer aC, 'N aC/count')
        ScanNNN_<diag>Spec_SSS.txt   allESpec table
        ScanNNN_<diag>Div_SSS.txt    allEDiv table

    Set ``load`` to read only what the view needs:

    - ``'image'`` (default): ``data`` = allE image [aC]. For the image-grid
      displayers (``MeanImagePerBin``, ``RepresentativeImagePerBin``,
      ``SampledImages``, ...).
    - ``'spec'``: ``aux`` = ``'allESpec'`` and ``'p'`` / ``'p_lo'`` (spectrum
      [pC/GeV] on the fixed momentum grid). For ``MagSpecAllEWaterfall`` and
      the lineout family on ``axis='p'``.
    - ``'div'``: ``aux`` = ``'allEDiv'`` and ``'angle'`` / ``'angle_lo'``
      (divergence [fC/mrad]). For the lineout family on ``axis='angle'``.
    - ``'all'``: all three.
    - ``'infoE'``: everything ``MagSpecAllEAnalyzer`` shows in its infoE
      summary figure, so ``analyze_scan(display_data=True,
      write_displayed=True)`` redraws it without re-analysing. ``aux['infoE']``
      holds the ``draw_infoe`` inputs, built from the saved allE files, the
      shot's sfile scalars (columns named
      ``'<diagnostic> <name> <analysis_label>'``) and, if present, the saved
      x-ray files (``xray_diagnostic``) and EBeam profile
      (``ebeam_diagnostic``). Without the x-ray files the x-ray panel and
      the acceptance / gap masks are blank (warned once). The density is
      drawn on the saved uniform momentum axis, not the analysis' ROI axis.
      ``write_displayed_data`` saves into ``output_diagnostic`` (default
      ``'<diagnostic>-infoE'``).

    ``results`` holds nothing: the scalars are already sfile columns if the
    analysis was run with ``write_columns_to_sfile=True``, or can be re-merged
    from its ``...Summary.txt``. Rank-by-scalar views such as
    ``RepresentativeImagePerBin(mode='max', parameter=...)`` therefore work
    on those existing columns and read only the chosen shots' files.

    Every mode lists the PNGs to find each shot's files, so shots without
    saved outputs are masked; the tables are read from beside the PNG.

    Parameters
    ----------
    load : str
        ``'image'``, ``'spec'``, ``'div'`` or ``'all'``.
    diagnostic : str
        The analyzer's ``output_diagnostic`` (default ``'MagSpecAllE'``).
        ``'allE'`` reads the MATLAB outputs, which use the same naming.
    analyzer_dict : dict, optional
        ``roi`` / ``momentum_grid`` as for ``MagSpecAllEAnalyzer`` -- defines
        the fixed grid for ``aux['p']``. Keep them the same as the analysis
        run for identical waterfalls.
    display_dict : dict, optional
        As for ``MagSpecAllEAnalyzer.display_data`` (``figsize``, ``cmap``,
        ``vmin``, ``vmax``; ``fontsize``, ``dpi`` for infoE).
    output_diagnostic : str, optional
        Folder for ``write_displayed_data``. Default ``diagnostic``, or
        ``'<diagnostic>-infoE'`` for ``load='infoE'``.
    analysis_label : str
        The ``analysis_label`` the scalars were written to the sfile with
        (``load='infoE'``).
    xray_diagnostic, ebeam_diagnostic : str or None
        Folders under ``analysis/ScanNNN/`` of the saved x-ray files and
        EBeam profile (``load='infoE'``). See ``MagSpecAllEAnalyzer``.

    Notes
    -----
    The image axes are not saved with the PNG. The momentum axis is
    recovered from the Spec table when it is loaded (``load='all'``), and
    otherwise approximated as ``linspace(roi)`` -- which ignores the exact
    ROI edge snapping, so use ``'all'`` if the image axis must be exact.
    The angle axis is the fixed 256-point [-1.3, 1.3] mrad axis.
    """

    def __init__(self, load='image', diagnostic='MagSpecAllE', analyzer_dict=None,
                 display_dict=None, output_diagnostic=None, analysis_label='',
                 xray_diagnostic=None, ebeam_diagnostic='CAM-TEA-EBeam_ProfileA'):
        if load not in _MODES:
            raise ValueError(f"load must be one of {sorted(_MODES)}, got {load!r}")
        if output_diagnostic is None:
            # infoE figures go to their own folder, not into the allE one
            output_diagnostic = f'{diagnostic}-infoE' if load == 'infoE' else diagnostic
        # file_list always points at the PNG: listing its folder is cheap and
        # masks shots with no outputs; the tables sit beside it
        super().__init__(diagnostic=diagnostic, file_ext='.png',
                         analyzer_dict=analyzer_dict, display_dict=display_dict,
                         output_diagnostic=output_diagnostic)
        self.load = load
        self.analysis_label = analysis_label
        self.xray_diagnostic = xray_diagnostic
        self.ebeam_diagnostic = ebeam_diagnostic
        self._warned = set()
        self._warn_lock = threading.Lock()

    def _warn_once(self, key, msg):
        """Thread-safe: warn once per analyzer per ``key``."""
        with self._warn_lock:
            if key in self._warned:
                return
            self._warned.add(key)
        warnings.warn(msg, stacklevel=3)

    def _load_infoe_files(self, filename):
        """Files beside the allE PNG that infoE adds: the x-ray image and
        ``Info`` table (``xray_diagnostic``) and the saved EBeam profile.
        Each is None if missing. Locations follow from ``filename`` =
        ``<top>/analysis/ScanNNN/<diag>/ScanNNN_<diag>_SSS.png``."""
        analysis_dir = os.path.dirname(os.path.dirname(os.path.dirname(filename)))
        scan_name = os.path.basename(os.path.dirname(os.path.dirname(filename)))   # 'ScanNNN'
        scan = int(scan_name[4:])
        shot = int(os.path.splitext(filename)[0].rsplit('_', 1)[1])
        out = {'xray': None, 'info': None,
               'ebeam': load_ebeam_profile(os.path.dirname(analysis_dir), scan, shot,
                                           self.ebeam_diagnostic)}
        xr = self.xray_diagnostic
        if xr:
            folder = os.path.join(analysis_dir, scan_name, xr)
            png = os.path.join(folder, f'{scan_name}_{xr}_{shot:03d}.png')
            info = os.path.join(folder, f'{scan_name}_{xr}Info_{shot:03d}.txt')
            if os.path.exists(info):
                out['info'] = pd.read_csv(info, sep='\t')
            if os.path.exists(png):
                out['xray'] = 1e-3 * read_int_ac_png(png)   # aC -> fC
        return out

    @staticmethod
    def table_path(png_path, kind):
        """``.../ScanNNN_<diag>_SSS.png`` -> ``.../ScanNNN_<diag><kind>_SSS.txt``."""
        folder, name = os.path.split(png_path)
        stem, shot = os.path.splitext(name)[0].rsplit('_', 1)
        return os.path.join(folder, f'{stem}{kind}_{shot}.txt')

    def load_data(self, filename):
        if not isinstance(filename, str):
            return None
        want_image, tables = _MODES[self.load]
        out = {}
        if want_image:
            if not os.path.exists(filename):
                return None
            out['image'] = read_int_ac_png(filename)
        for kind in tables:
            path = self.table_path(filename, kind)
            if not os.path.exists(path):
                return None
            out[kind.lower()] = pd.read_csv(path, sep='	')
        if self.load == 'infoE':
            out.update(self._load_infoe_files(filename))
        return out

    def momentum_grid(self):
        """Same fixed grid as ``MagSpecAllEAnalyzer.momentum_grid``."""
        ad = self.analyzer_dict
        if ad.get('momentum_grid') is not None:
            return np.asarray(ad['momentum_grid'], float)
        roi = ad.get('roi', (0.01, 5.0))
        return np.linspace(roi[0], roi[1], 1024)

    def analyze_data(self, data, bg=None, context=None, analyzer_dict=None):
        if data is None:
            return None, {}, {}
        aux = {}
        if 'spec' in data:
            spec = data['spec']
            grid = self.momentum_grid()
            p_lo = interp1(spec['Momentum_GeV/c'].to_numpy(), spec['ChargeDen_pC/GeV/c'].to_numpy(), grid)
            aux.update({'allESpec': spec, 'p': grid, 'p_lo': np.nan_to_num(p_lo, nan=0.0),
                        'momentum': spec['Momentum_GeV/c'].to_numpy()})
        if 'div' in data:
            div = data['div']
            aux.update({'allEDiv': div, 'angle': div['Angle_mrad'].to_numpy(),
                        'angle_lo': div['ChargeDen_fC/mrad'].to_numpy()})
        if self.load == 'infoE':
            aux['infoE'] = self._infoe_inputs(data, context or {})
        return data.get('image'), {}, aux

    def _sfile_value(self, context, name):
        """This analysis' sfile column ``name`` (``'<diag> <name> <label>'``)."""
        col = ' '.join(s for s in (self.diagnostic, name, self.analysis_label) if s)
        return context.get(col)

    def _infoe_inputs(self, data, context):
        """Keyword arguments for ``draw_infoe`` from the saved files and the
        shot's sfile scalars (``context``)."""
        spec, div, image = data['spec'], data['div'], data['image']
        mmt = spec['Momentum_GeV/c'].to_numpy()
        ya = div['Angle_mrad'].to_numpy()
        # the saved allE image is on the uniform momentum axis, so the density
        # is drawn there rather than on the analysis' non-uniform ROI axis
        density = 1e-6 * image / (mmt[1] - mmt[0]) / (ya[1] - ya[0])
        scalars = {}
        for name in _INFOE_SCALARS:
            v = self._sfile_value(context, name)
            if v is not None and np.isfinite(v):
                scalars[name] = float(v)
        if 'charge_pC' not in scalars:
            self._warn_once('scalars', f"no '{self.diagnostic} ... {self.analysis_label}' scalar "
                            "columns in the sfile (check analysis_label, and that the analysis "
                            "wrote them): the infoE text panels will be mostly empty.")
        accp, gap = np.full(mmt.size, ya[-1]), (mmt[0], mmt[0])    # nothing masked
        info = data.get('info')
        if info is not None:
            accp = np.interp(mmt, info['mmt_GeV/c'].dropna().to_numpy(),
                             info['accp_mrad'].dropna().to_numpy())
            g = info['gap_GeV/c'].dropna().to_numpy()
            gap = (g[0], g[1]) if g.size >= 2 else gap
        xray = data.get('xray')
        if info is not None and xray is not None:
            xray_x, xray_y = (info[c].dropna().to_numpy() for c in ('xray_x_mm', 'xray_y_mm'))
        else:
            self._warn_once('xray', "no saved x-ray files (re-run MagSpecAllEAnalyzer with "
                            "xray_diagnostic=... and write_analyzed=True): infoE has no x-ray "
                            "panel or acceptance mask.")
            xray, xray_x, xray_y = np.zeros((2, 2)), np.array([0.0, 1.0]), np.array([0.0, 1.0])
        ict = context.get(ICT_COLUMN, np.nan)
        ey = self._sfile_value(context, 'eYAngle_mrad')
        return {
            'mmt': mmt, 'ya': ya, 'density': density, 'accp': accp,
            'spectrum': spec['ChargeDen_pC/GeV/c'].to_numpy(), 'gap': gap, 'scalars': scalars,
            'xray_img': xray, 'xray_x_mm': xray_x, 'xray_y_mm': xray_y,
            'ebeam': data.get('ebeam'),
            'ict_pC': float(ict) if ict is not None else np.nan,
            'ey_angle': float(ey) if ey is not None else np.nan,
            'scan': context.get('scan'), 'shot': context.get('Shotnumber'),
            'roi': (mmt[0], mmt[-1]),
        }

    def write_displayed_data(self, fig, analysis_dir, scan, shot_num):
        """Save the displayed figure as
        ``<output_diagnostic>/ScanNNN_<output_diagnostic>infoE_SSS.png``."""
        if fig is None:
            return
        path = get_analysed_shot_save_path(analysis_dir, self.output_diagnostic, scan, shot_num,
                                           '.png', 'infoE')
        fig.savefig(path, dpi=self.display_dict.get('dpi', 120))

    def display_data(self, data, return_dict=None, title=None, fig=None, ax=None, aux=None):
        """allE charge density [pC/mrad/(GeV/c)] -- same view as
        ``MagSpecAllEAnalyzer.display_data`` (image panel only when the
        caller supplies a single axis). With ``load='infoE'`` (and no
        ``fig``/``ax``) it draws the full infoE summary figure instead."""
        if data is None:
            return None, None
        dd = self.display_dict
        info = (aux or {}).get('infoE')
        if info is not None and fig is None and ax is None:
            return draw_infoe(**info, fontsize=dd.get('fontsize', 10),
                              figsize=dd.get('figsize', (20, 6.67)))
        n_ang, n_mmt = data.shape
        last = aux or {}
        mmt = np.asarray(last['momentum']) if len(last.get('momentum', ())) == n_mmt else \
            np.linspace(*self.analyzer_dict.get('roi', (0.01, 5.0)), n_mmt)
        ang = np.asarray(last['angle']) if len(last.get('angle', ())) == n_ang else \
            np.linspace(-1.3, 1.3, n_ang)
        dens = 1e-6 * data / (mmt[1] - mmt[0]) / (ang[1] - ang[0])
        if fig is None or ax is None:
            fig, ax = plt.subplots(2, 1, figsize=dd.get('figsize', (10, 5)), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})
        axes = np.atleast_1d(ax)
        m = axes[0].pcolormesh(mmt, ang, dens, cmap=dd.get('cmap', 'jet'), shading='auto',
                               vmin=dd.get('vmin', 0), vmax=dd.get('vmax'))
        fig.colorbar(m, ax=list(axes), label='pC/mrad/(GeV/c)', location='right', shrink=0.9)
        axes[0].set_ylabel('mrad')
        if title:
            axes[0].set_title(title)
        if len(axes) > 1:
            axes[1].semilogy(mmt, 1e-6 * data.sum(axis=0) / (mmt[1] - mmt[0]), 'r-')
            axes[1].set_xlabel('GeV/c')
            axes[1].set_ylabel('pC/GeV')
        return fig, ax
