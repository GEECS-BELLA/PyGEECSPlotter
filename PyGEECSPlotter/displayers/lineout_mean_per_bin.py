from typing import Optional, Dict, Any, Iterable, List

import numpy as np
import matplotlib.pyplot as plt

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer
from PyGEECSPlotter.displayers._lineout_binning import mean_lineouts_per_bin, on_grid
from PyGEECSPlotter.displayers._trace_binning import bin_labels

# Distinct colour/style per axis so an overlay of several lineouts on one
# panel stays legible without a per-axis colormap.
_AXIS_STYLE = {
    'x': dict(color='tab:blue'),
    'y': dict(color='tab:orange'),
    'r': dict(color='tab:green'),
}


class LineoutMeanPerBin(ScanDisplayer):
    """
    Grid of per-bin mean lineouts — the 1-D analogue of ``MeanImagePerBin``.

    One panel per bin; each panel overlays the mean lineout for every axis in
    ``axes`` (e.g. ``x`` and ``y``), one line per axis, colour/style-coded
    with a shared legend. Companion to ``TraceMeanPerBin``, but for the flat
    ``aux`` lineout dict an ``ImageAnalyzer``-style analyzer returns
    (``{'x': x, 'y': y, 'x_lo': x_lo, 'y_lo': y_lo}``) rather than a dict of
    trace DataFrames.

    Parameters
    ----------
    analyzer : DiagnosticAnalyzer
        Per-shot analyzer returning a lineout-style ``aux`` dict.
    axes : iterable of str, optional
        Which coordinates to overlay per panel, e.g. ``['x', 'y']``.
        Defaults to every coordinate present in the aux dict.
    bg : optional
        Background spec forwarded to the per-shot pipeline.
    bins : iterable of int, optional
        Bin numbers to render. Defaults to all unique bins in ``active_data``.
    ncols : int, optional
        Number of columns in the grid.
    show_std : bool, optional
        Shade +/- 1 sigma across the bin's shots around each axis's mean.
    label_column : str, optional
        Scan-data column to label each bin with, instead of the raw bin
        number — e.g. the scan parameter's value at that bin. Defaults to
        ``scan.scan_parameter``, summarised per bin with
        ``scan.compute_bin_summary(mode='mean')``. Pass ``False`` to fall
        back to plain ``'Bin {n}'`` labels.
    label_fmt : str, optional
        Format string for the label column's value, e.g. ``'{:.3g} mm'``.
    suppress_labels : bool, optional
        Strip inner axis labels for a cleaner grid (default True).
    align : {'snap', 'interp'}, optional
        How shots whose coordinate axes are shifted (e.g. ImageAnalyzer
        ``centroid_method='centroid'``) are put on one axis: ``'snap'``
        (default) shifts by whole pixels, exact, needing ImageAnalyzer's
        ``round_centroid=True`` (its default); ``'interp'`` interpolates,
        for sub-pixel origins.
    display_dict : dict, optional
        Style overrides: ``figsize``, ``std_alpha``, ``panel_aspect``
        (plot-area width / height; default None) and ``panel_width`` (inches,
        with ``panel_aspect``).
    """

    def __init__(
        self,
        analyzer,
        axes: Optional[Iterable[str]] = None,
        bg=None,
        bins: Optional[Iterable[int]] = None,
        ncols: int = 4,
        show_std: bool = True,
        label_column=None,
        label_fmt: str = '{:.4g}',
        suppress_labels: bool = True,
        align: str = 'snap',
        display_dict: Optional[Dict[str, Any]] = None,
        output_subdir: Optional[str] = None,
        timestamp_files: bool = True,
    ):
        diag = analyzer.output_diagnostic or analyzer.diagnostic
        super().__init__(name=f'{diag}_lineout_mean_per_bin', display_dict=display_dict,
                          output_subdir=output_subdir, timestamp_files=timestamp_files)
        self.analyzer = analyzer
        self.axes = list(axes) if axes is not None else None
        self.bg = bg
        self.bins = bins
        self.ncols = ncols
        self.show_std = show_std
        self.label_column = label_column
        self.label_fmt = label_fmt
        self.align = align
        self.suppress_labels = suppress_labels

    # ------------------------------------------------------------------
    def display(self, scan, fig=None, ax=None):
        bins, per_bin = mean_lineouts_per_bin(
            scan, self.analyzer, bg=self.bg, bins=self.bins, axes=self.axes,
            align=self.align,
        )
        if all(p is None for p in per_bin):
            raise RuntimeError(f"{type(self).__name__}: no bins produced data.")

        labels = bin_labels(scan, bins, label_column=self.label_column, label_fmt=self.label_fmt)
        axes_present: List[str] = self.axes if self.axes is not None else sorted(
            {axis for entry in per_bin if entry is not None for axis in entry}
        )

        self.last_export = self._build_export(bins, per_bin, labels, axes_present)

        n_panels = len(bins)
        ncols = min(self.ncols, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes_arr = plt.subplots(
            nrows, ncols, figsize=self._grid_figsize(ncols, nrows, panel=(3.5, 3)),
            constrained_layout=True, squeeze=False,
        )

        legend_done = False
        for k, (entry, label) in enumerate(zip(per_bin, labels)):
            a = axes_arr.flat[k]
            if entry is None:
                a.set_visible(False)
                continue
            for axis in axes_present:
                if axis not in entry:
                    continue
                coord, mean, std = entry[axis]
                style = _AXIS_STYLE.get(axis, {})
                a.plot(coord, mean, label=axis, **style)
                if self.show_std:
                    a.fill_between(
                        coord, mean - std, mean + std,
                        alpha=self.display_dict.get('std_alpha', 0.25),
                        color=style.get('color'), lw=0,
                    )
            a.set_title(label)
            # one legend, on the first drawn panel: the axis styles are the same in every panel
            if len(axes_present) > 1 and not legend_done:
                a.legend(fontsize='small')
                legend_done = True
            if self.suppress_labels:
                a.set_xlabel(None)
                a.set_ylabel(None)
                a.set_xticklabels([])
                a.set_yticklabels([])

        for k in range(n_panels, nrows * ncols):
            axes_arr.flat[k].set_visible(False)

        self._apply_panel_aspect(axes_arr)
        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        fig.suptitle(scan.scan_data_title(f'{diag} lineout mean per bin'))
        return fig, axes_arr

    def _build_export(self, bins, per_bin, labels, axes_present):
        """
        Stack each bin's mean/std lineout into one 2D array per axis, so the
        whole grid can be reloaded without recomputing anything.
        """
        export = {'bins': np.asarray(bins, dtype=float), 'labels': np.asarray(labels, dtype=object)}
        for axis in axes_present:
            coord_ref = next(
                (entry[axis][0] for entry in per_bin if entry is not None and axis in entry), None
            )
            if coord_ref is None:
                continue
            mean_stack = np.full((len(bins), len(coord_ref)), np.nan)
            std_stack = np.full((len(bins), len(coord_ref)), np.nan)
            for k, entry in enumerate(per_bin):
                if entry is None or axis not in entry:
                    continue
                c, mean, std = entry[axis]
                mean_stack[k] = on_grid(np.asarray(c, float), np.asarray(mean, float), coord_ref, self.align)
                std_stack[k] = on_grid(np.asarray(c, float), np.asarray(std, float), coord_ref, self.align)
            export[f'{axis}_coord'] = coord_ref
            export[f'{axis}_mean'] = mean_stack
            export[f'{axis}_std'] = std_stack

        return export
