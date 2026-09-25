from typing import Optional, Dict, Any

import numpy as np
from matplotlib.colors import LogNorm

from PyGEECSPlotter.displayers.lineout_waterfall import LineoutWaterfall


class MagSpecAllEWaterfall(LineoutWaterfall):
    """
    Every shot's allE electron spectrum stacked into one image: one row per
    shot, x = momentum [GeV/c], colour = charge density [pC/GeV].

    A ``LineoutWaterfall`` on the ``'p'`` / ``'p_lo'`` pair that
    ``MagSpecAllEAnalyzer`` puts in ``aux``: each shot's spectrum resampled
    onto the analyzer's fixed momentum grid (its own allE axis moves with
    the magnet field and the e-beam input angle). Set the grid with
    ``analyzer_dict['momentum_grid']``; the default is 1024 points across
    ``roi``.

    Parameters
    ----------
    analyzer : MagSpecAllEAnalyzer
    bg : optional
        Background spec forwarded to the per-shot pipeline.
    y_column : str, optional
        As for ``LineoutWaterfall``: row order / y label; defaults to the
        scan parameter, ``False`` keeps acquisition order.
    display_dict : dict, optional
        ``LineoutWaterfall`` keys, plus ``log`` (bool, default True): log
        colour scale, with ``vmin`` defaulting to ``vmax / 1e3``.
    """

    def __init__(
        self,
        analyzer,
        bg=None,
        y_column=None,
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        dd = {'cmap': 'jet', 'xlabel': 'Momentum [GeV/c]', 'cbar_label': 'pC/GeV'}
        dd.update(display_dict or {})
        super().__init__(analyzer, axis='p', bg=bg, y_column=y_column, display_dict=dd)
        diag = analyzer.output_diagnostic or analyzer.diagnostic
        self.name = f'{diag}_spectrum_waterfall'

    def display(self, scan, fig=None, ax=None):
        fig, ax = super().display(scan, fig=fig, ax=ax)
        im = ax.images[-1]
        if self.display_dict.get('log', True):
            stack = self.last_export['stack']
            vmax = self.display_dict.get('vmax') or np.nanmax(stack)
            vmin = self.display_dict.get('vmin') or vmax / 1e3
            im.set_norm(LogNorm(vmin=vmin, vmax=vmax))
            # zero-charge bins (e.g. the gap between the two screens) would
            # be masked white on a log scale; show them as the lowest colour
            cmap = im.get_cmap().copy()
            cmap.set_bad(cmap(0.0))
            cmap.set_under(cmap(0.0))
            im.set_cmap(cmap)
        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        ax.set_title(scan.scan_data_title(f'{diag} spectrum waterfall'))
        return fig, ax
