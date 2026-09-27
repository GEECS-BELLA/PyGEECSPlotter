# Reader for allE outputs already written by MagSpecAllEAnalyzer
# (write_analyzed=True). Scan-level displayers can run on it instead of on
# the full ten-camera analysis: no calibration, no image processing, just
# file reads -- and only the files a given view needs.

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PyGEECSPlotter.diagnostic_analyzer import DiagnosticAnalyzer
from PyGEECSPlotter.magspec.io import read_int_ac_png
from PyGEECSPlotter.magspec.matlab_compat import interp1

# which saved tables each `load` mode reads, besides (or instead of) the image
_MODES = {
    'image': (True, ()),
    'spec': (False, ('Spec',)),
    'div': (False, ('Div',)),
    'all': (True, ('Spec', 'Div')),
}


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
        ``vmin``, ``vmax``).

    Notes
    -----
    The image axes are not saved with the PNG. The momentum axis is
    recovered from the Spec table when it is loaded (``load='all'``), and
    otherwise approximated as ``linspace(roi)`` -- which ignores the exact
    ROI edge snapping, so use ``'all'`` if the image axis must be exact.
    The angle axis is the fixed 256-point [-1.3, 1.3] mrad axis.
    """

    def __init__(self, load='image', diagnostic='MagSpecAllE', analyzer_dict=None,
                 display_dict=None):
        if load not in _MODES:
            raise ValueError(f"load must be one of {sorted(_MODES)}, got {load!r}")
        # file_list always points at the PNG: listing its folder is cheap and
        # masks shots with no outputs; the tables sit beside it
        super().__init__(diagnostic=diagnostic, file_ext='.png',
                         analyzer_dict=analyzer_dict, display_dict=display_dict,
                         output_diagnostic=diagnostic)
        self.load = load
        self._last_aux = None

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
        self._last_aux = aux
        return data.get('image'), {}, aux

    def display_data(self, data, return_dict=None, title=None, fig=None, ax=None):
        """allE charge density [pC/mrad/(GeV/c)] -- same view as
        ``MagSpecAllEAnalyzer.display_data`` (image panel only when the
        caller supplies a single axis)."""
        if data is None:
            return None, None
        dd = self.display_dict
        n_ang, n_mmt = data.shape
        last = self._last_aux or {}
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
