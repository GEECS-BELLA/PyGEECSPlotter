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
        Style overrides: ``cmap``; ``panel_aspect`` (plot-area width /
        height, default 1 for square panels, e.g. 4 for 4:1; None leaves
        each panel its own shape); ``panel_width`` (inches per panel before
        shrinking to fit, default 2.5); ``cbar_span`` (panels a shared top
        colour bar spans, default 2); ``max_figsize`` (cap on the figure,
        default ``MAX_FIG_SIZE``: a letter page wide, 0.7 of one tall;
        panels shrink to fit); ``figsize`` (used as given, ignoring the cap).

    Notes
    -----
    This displayer creates its own figure; ``fig`` / ``ax`` arguments to
    ``display`` are ignored.
    """

    # Subclasses override for a different default panel shape.
    default_panel_aspect = 1
    default_panel_width = 2.5

    def __init__(
        self,
        analyzer,
        ncols: int = 4,
        use_analyzer_display: bool = True,
        suppress_labels: bool = True,
        display_dict: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        output_subdir: Optional[str] = None,
        timestamp_files: bool = True,
    ):
        if name is None:
            name = f"{analyzer.output_diagnostic or analyzer.diagnostic}_image_grid"
        super().__init__(name=name, display_dict=display_dict,
                          output_subdir=output_subdir, timestamp_files=timestamp_files)
        self.display_dict.setdefault('panel_aspect', self.default_panel_aspect)
        self.display_dict.setdefault('panel_width', self.default_panel_width)
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

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=self._grid_figsize(ncols, nrows, pad=self._panel_pad()),
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

        self._apply_panel_aspect(axes)
        fig.suptitle(self._suptitle(scan))
        self.last_export = self._build_export(panels)
        return fig, axes

    def _build_export(self, panels):
        """
        Stack panel images into one array (labels alongside), so the grid
        can be reloaded without recomputing anything. Panels are stacked as
        a plain ``(n_panels, ...)`` array when every image shares one shape;
        otherwise stored as an object array, one entry per panel.

        The pixel scale goes alongside, so the axes can be rebuilt on
        reload: ``dx``, ``dy`` and ``spatial_units`` from the analyzer
        (1, 1, ``'pixels'`` if unset), and ``extents``, each panel's
        ``imshow_extent`` ``[x0, x1, y0, y1]`` where the analyzer gave one
        (NaN otherwise, e.g. for per-bin means, which are in pixels from
        the image corner: ``x = arange(nx) * dx``).
        """
        labels = np.asarray([label for label, _, _ in panels], dtype=object)
        images = [data for _, data, _ in panels]

        analyzer_dict = getattr(self.analyzer, 'analyzer_dict', None) or {}
        analyzer_display = getattr(self.analyzer, 'display_dict', None) or {}
        units = analyzer_display.get('spatial_units',
                                     analyzer_dict.get('spatial_units', 'pixels'))
        extents = np.full((len(panels), 4), np.nan)
        for k, (_, _, return_dict) in enumerate(panels):
            ext = (return_dict or {}).get('imshow_extent')
            if ext is not None:
                extents[k] = np.asarray(ext, float)

        shapes = {np.asarray(im).shape for im in images if im is not None}
        if len(shapes) == 1 and len(images) == sum(im is not None for im in images):
            stack = np.stack([np.asarray(im) for im in images], axis=0)
        else:
            stack = np.empty(len(images), dtype=object)
            for k, im in enumerate(images):
                stack[k] = None if im is None else np.asarray(im)

        return {'labels': labels, 'images': stack,
                'dx': float(analyzer_dict.get('dx', 1)),
                'dy': float(analyzer_dict.get('dy', 1)),
                'spatial_units': str(units),
                'extents': extents}

    def _panel_pad(self):
        """Margin per panel, inches (width, height), for ``_grid_figsize``.
        Labels sit inside the panels, so only tick labels (when kept) and
        per-panel colour bars need room."""
        pad_w, pad_h = (0.15, 0.05) if self.suppress_labels else (0.6, 0.35)
        analyzer_display = getattr(self.analyzer, 'display_dict', None) or {}
        if self.use_analyzer_display and not analyzer_display.get('cbar_off', False):
            pad_w += 0.6
        return pad_w, pad_h

    @staticmethod
    def _corner_label(a, label):
        """Write ``label`` in the panel's top-left corner, on a translucent
        dark box so it reads on any colour map, and clear any axes title."""
        a.set_title('')
        a.text(0.03, 0.97, label, transform=a.transAxes, ha='left', va='top',
               fontsize='small', color='w',
               bbox=dict(facecolor='k', alpha=0.5, edgecolor='none', pad=1.5))

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
                aspect='equal',
                cmap=self.display_dict.get('cmap', 'viridis'),
            )
        # Label inside the panel rather than as a title, so rows pack tight.
        self._corner_label(a, label)
        if self.suppress_labels:
            a.set_xlabel(None)
            a.set_ylabel(None)
            a.set_xticklabels([])
            a.set_yticklabels([])
