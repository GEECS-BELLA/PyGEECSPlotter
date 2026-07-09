from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer


class SampledImages(ScanDisplayer):
    """
    Grid of individual shot images sampled evenly across ``active_data``.

    Per-shot analogue of ``MeanImagePerBin``: instead of averaging each
    bin, pick ``n_samples`` shots equally spaced across
    ``scan.active_data`` and render each through the analyzer's pipeline.
    The first and last active shots are always included; intermediate
    samples are evenly spaced between them.

    Parameters
    ----------
    analyzer : DiagnosticAnalyzer
        Per-shot analyzer used to load + process each sampled shot.
    bg : optional
        Background spec forwarded to the per-shot pipeline
        (``ScanDataAnalyzer._iter_shots``).
    n_samples : int, optional
        Number of shots to display. Defaults to ``len(scan.active_data)``
        (every active shot). Clamped to the number of active shots. When
        rounding to integer row positions would collide (``n_samples``
        close to the shot count), fewer unique panels may result.
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
    ignored. Only the sampled shots are loaded and analyzed, so this stays
    cheap even for scans with thousands of shots.
    """

    def __init__(
        self,
        analyzer,
        bg=None,
        n_samples: Optional[int] = None,
        ncols: int = 4,
        use_analyzer_display: bool = True,
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        name = f"{analyzer.output_diagnostic or analyzer.diagnostic}_sampled"
        super().__init__(name=name, display_dict=display_dict)
        self.analyzer = analyzer
        self.bg = bg
        self.n_samples = n_samples
        self.ncols = ncols
        self.use_analyzer_display = use_analyzer_display

    def _sample_indices(self, n_total: int) -> np.ndarray:
        """Evenly-spaced, unique row positions across ``n_total`` shots."""
        n = self.n_samples if self.n_samples is not None else n_total
        n = max(1, min(int(n), n_total))
        return np.unique(np.linspace(0, n_total - 1, n).round().astype(int))

    def display(self, scan, fig=None, ax=None):
        active = scan.active_data
        n_total = len(active)
        if n_total == 0:
            raise RuntimeError("No active shots to display.")

        idxs = self._sample_indices(n_total)
        subset = active.iloc[idxs]

        n_panels = len(subset)
        ncols = min(self.ncols, n_panels)
        nrows = int(np.ceil(n_panels / ncols))

        figsize = self.display_dict.get('figsize', (3 * ncols, 3 * nrows))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize,
            constrained_layout=True,
            squeeze=False,
        )

        # Iterate only the sampled rows. Panel index advances even when a
        # shot is missing, so each panel keeps its position in scan order.
        panel = 0
        for context, data, results, _ in scan._iter_shots(
            self.analyzer, bg=self.bg, show_progress=False, rows=subset,
        ):
            a = axes.flat[panel]
            shot = int(context.get('Shotnumber', idxs[panel]))
            if data is None:
                a.set_visible(False)
                panel += 1
                continue
            if self.use_analyzer_display:
                self.analyzer.display_data(
                    data, return_dict=results, fig=fig, ax=a, title=f'shot {shot}'
                )
            else:
                a.imshow(
                    np.asarray(data),
                    origin='lower',
                    cmap=self.display_dict.get('cmap', 'viridis'),
                )
                a.set_title(f'shot {shot}')
            panel += 1

        for k in range(n_panels, nrows * ncols):
            axes.flat[k].set_visible(False)

        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        fig.suptitle(scan.scan_data_title(f'{diag} sampled shots'))
        return fig, axes
