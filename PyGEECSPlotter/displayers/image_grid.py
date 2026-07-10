from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer


class ImageGridDisplayer(ScanDisplayer):
    """
    Base class for displayers that render a grid of per-shot images.

    Owns everything visual — figure/grid layout, the render loop, the
    ``use_analyzer_display`` vs plain ``imshow`` choice, axis-label
    suppression, blank-panel handling, and the suptitle. Subclasses only
    answer *which* images go in the panels by implementing
    ``_collect_panels``.

    Parameters
    ----------
    analyzer : DiagnosticAnalyzer
        Per-shot analyzer whose ``display_data`` renders a single panel.
    ncols : int, optional
        Number of columns in the figure grid.
    use_analyzer_display : bool, optional
        If True, render each panel with ``analyzer.display_data`` (preserves
        colormap / extent / lineouts settings). If False, plain ``imshow``.
    suppress_labels : bool, optional
        If True (default), strip per-panel axis labels and tick labels for
        a cleaner thumbnail grid. Pass False to keep the analyzer's axes.
    display_dict : dict, optional
        Style overrides: ``figsize``, ``cmap``.

    Notes
    -----
    This displayer creates its own figure; ``fig`` / ``ax`` arguments to
    ``display`` are ignored.
    """

    def __init__(
        self,
        analyzer,
        ncols: int = 4,
        use_analyzer_display: bool = True,
        suppress_labels: bool = True,
        display_dict: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        if name is None:
            name = f"{analyzer.output_diagnostic or analyzer.diagnostic}_image_grid"
        super().__init__(name=name, display_dict=display_dict)
        self.analyzer = analyzer
        self.ncols = ncols
        self.use_analyzer_display = use_analyzer_display
        self.suppress_labels = suppress_labels

    # ------------------------------------------------------------------
    # Subclasses override this.
    # ------------------------------------------------------------------
    def _collect_panels(self, scan) -> List[Tuple[str, Any, Optional[Dict[str, Any]]]]:
        """
        Return the panels to render, in order.

        Each panel is ``(label, data, return_dict)``:
          - ``label``: panel title (str).
          - ``data``: image array, or ``None`` to leave the panel blank.
          - ``return_dict``: optional dict passed to ``display_data`` (e.g.
            for imshow extent); may be ``None``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _collect_panels(scan)."
        )

    def _suptitle(self, scan) -> str:
        """Figure suptitle. Subclasses may override."""
        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        return scan.scan_data_title(f'{diag} image grid')

    # ------------------------------------------------------------------
    # Shared render loop
    # ------------------------------------------------------------------
    def display(self, scan, fig=None, ax=None):
        panels = self._collect_panels(scan)
        if not panels:
            raise RuntimeError(f"{type(self).__name__} produced no panels.")

        n_panels = len(panels)
        ncols = min(self.ncols, n_panels)
        nrows = int(np.ceil(n_panels / ncols))

        figsize = self.display_dict.get('figsize', (3 * ncols, 3 * nrows))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize,
            constrained_layout=True,
            squeeze=False,
        )

        for k, (label, data, return_dict) in enumerate(panels):
            a = axes.flat[k]
            if data is None:
                a.set_visible(False)
                continue
            self._render_panel(fig, a, data, return_dict, label)

        for k in range(n_panels, nrows * ncols):
            axes.flat[k].set_visible(False)

        fig.suptitle(self._suptitle(scan))
        return fig, axes

    def _render_panel(self, fig, a, data, return_dict, label):
        """Draw one panel and apply the shared label/tick treatment."""
        if self.use_analyzer_display:
            self.analyzer.display_data(
                data, return_dict=return_dict, fig=fig, ax=a, title=label
            )
        else:
            a.imshow(
                np.asarray(data),
                origin='lower',
                cmap=self.display_dict.get('cmap', 'viridis'),
            )
        # Always own the title so it's consistent regardless of branch.
        a.set_title(label)
        if self.suppress_labels:
            a.set_xlabel(None)
            a.set_ylabel(None)
            a.set_xticklabels([])
            a.set_yticklabels([])
