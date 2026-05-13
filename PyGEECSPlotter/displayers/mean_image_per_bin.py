from typing import Optional, Dict, Any, Iterable

import numpy as np
import matplotlib.pyplot as plt

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer


class MeanImagePerBin(ScanDisplayer):
    """
    Grid of per-bin mean images, computed via the diagnostic analyzer's pipeline.

    Replaces the old ``ScanDataOverview.analyze_scan`` workflow:
    ``scan.display_scan(MeanImagePerBin(analyzer, ...))``.

    Parameters
    ----------
    analyzer : DiagnosticAnalyzer
        Per-shot analyzer used to load + process each shot.
    bg : optional
        Background spec forwarded to ``aggregate_per_bin``.
    bins : iterable of int, optional
        Bin numbers to render. Defaults to all unique bins in ``active_data``.
    ncols : int, optional
        Number of columns in the figure grid.
    use_analyzer_display : bool, optional
        If True, render each panel with ``analyzer.display_data`` (preserves
        colormap / extent / lineouts settings). If False, plain ``imshow``.
    display_dict : dict, optional
        Style overrides: ``figsize``, ``cmap``.

    Notes
    -----
    This displayer creates its own figure; ``fig`` / ``ax`` arguments are
    ignored.
    """

    def __init__(
        self,
        analyzer,
        bg=None,
        bins: Optional[Iterable[int]] = None,
        ncols: int = 4,
        use_analyzer_display: bool = True,
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        name = f"{analyzer.output_diagnostic or analyzer.diagnostic}_mean_per_bin"
        super().__init__(name=name, display_dict=display_dict)
        self.analyzer = analyzer
        self.bg = bg
        self.bins = bins
        self.ncols = ncols
        self.use_analyzer_display = use_analyzer_display

    def display(self, scan, fig=None, ax=None):
        bins, mean_per_bin, _ = scan.aggregate_per_bin(
            self.analyzer, bg=self.bg, bins=self.bins
        )
        if mean_per_bin is None:
            raise RuntimeError("No bins produced data.")

        n_bins = len(bins)
        ncols = min(self.ncols, n_bins)
        nrows = int(np.ceil(n_bins / ncols))

        figsize = self.display_dict.get('figsize', (3 * ncols, 3 * nrows))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize,
            constrained_layout=True,
            squeeze=False,
        )

        for k, b in enumerate(bins):
            a = axes.flat[k]
            data = mean_per_bin[k]
            if data is None or np.all(np.isnan(data)):
                a.set_visible(False)
                continue
            if self.use_analyzer_display:
                self.analyzer.display_data(data, fig=fig, ax=a, title=f'bin {int(b)}')
            else:
                a.imshow(
                    data,
                    origin='lower',
                    cmap=self.display_dict.get('cmap', 'viridis'),
                )
                a.set_title(f'bin {int(b)}')

        for k in range(n_bins, nrows * ncols):
            axes.flat[k].set_visible(False)

        fig.suptitle(scan.scan_data_title(f'{self.analyzer.diagnostic} mean per bin'))
        return fig, axes
