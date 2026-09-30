from typing import Optional, Dict, Any, Iterable

import numpy as np

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer
from PyGEECSPlotter.displayers._lineout_binning import mean_lineouts_per_bin
from PyGEECSPlotter.displayers._trace_binning import bin_labels


class LineoutMeanWaterfall(ScanDisplayer):
    """
    ``LineoutMeanPerBin``'s per-bin mean lineouts, stacked as a waterfall
    image — the per-bin analogue of ``LineoutWaterfall``, and the lineout
    counterpart of ``TraceMeanWaterfall``.

    Each row is one bin's mean lineout for a single coordinate (``axis``).
    Rows are in bin order, labelled on the y-axis by ``label_column``
    (default ``scan.scan_parameter``, summarised per bin) rather than a raw
    bin index.

    Parameters
    ----------
    analyzer : DiagnosticAnalyzer
        Per-shot analyzer returning a lineout-style ``aux`` dict.
    axis : str, optional
        Which coordinate to stack. Default ``'x'``.
    bg : optional
        Background spec forwarded to the per-shot pipeline.
    bins : iterable of int, optional
        Bin numbers to render. Defaults to all unique bins in ``active_data``.
    label_column : str, optional
        Scan-data column to label each row with, instead of the raw bin
        number. Defaults to ``scan.scan_parameter``, summarised per bin.
        Pass ``False`` to fall back to plain ``'Bin {n}'`` labels.
    label_fmt : str, optional
        Format string for the label column's value, e.g. ``'{:.3g} mm'``.
    display_dict : dict, optional
        Style overrides: ``figsize``, ``cmap``, ``vmin``, ``vmax``, ``xlims``,
        ``normalise_rows``, ``n_yticks``.
    """

    def __init__(
        self,
        analyzer,
        axis: str = 'x',
        bg=None,
        bins: Optional[Iterable[int]] = None,
        label_column=None,
        label_fmt: str = '{:.4g}',
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        diag = analyzer.output_diagnostic or analyzer.diagnostic
        super().__init__(name=f'{diag}_{axis}_lineout_mean_waterfall', display_dict=display_dict)
        self.analyzer = analyzer
        self.axis = axis
        self.bg = bg
        self.bins = bins
        self.label_column = label_column
        self.label_fmt = label_fmt

    # ------------------------------------------------------------------
    def display(self, scan, fig=None, ax=None):
        bins, per_bin = mean_lineouts_per_bin(
            scan, self.analyzer, bg=self.bg, bins=self.bins, axes=[self.axis]
        )
        if all(p is None for p in per_bin):
            raise RuntimeError(f"{type(self).__name__}: no bins produced data.")

        labels = bin_labels(scan, bins, label_column=self.label_column, label_fmt=self.label_fmt)

        coord = next(
            (entry[self.axis][0] for entry in per_bin if entry is not None and self.axis in entry),
            None,
        )
        if coord is None:
            raise RuntimeError(
                f"{type(self).__name__}: no bin had axis {self.axis!r} in aux."
            )

        rows = []
        for entry in per_bin:
            if entry is None or self.axis not in entry:
                rows.append(np.full_like(coord, np.nan, dtype=float))
                continue
            _, mean, _ = entry[self.axis]
            rows.append(np.asarray(mean, dtype=float))
        stack = np.vstack(rows)

        if self.display_dict.get('normalise_rows', False):
            peaks = np.nanmax(stack, axis=1, keepdims=True)
            with np.errstate(invalid='ignore', divide='ignore'):
                stack = np.where(peaks > 0, stack / peaks, stack)

        fig, ax = self._new_fig(fig, ax, figsize=(7, 6))

        y = np.arange(len(bins))
        im = ax.imshow(
            stack,
            aspect='auto',
            origin='lower',
            cmap=self.display_dict.get('cmap', 'viridis'),
            vmin=self.display_dict.get('vmin', None),
            vmax=self.display_dict.get('vmax', None),
            extent=[coord[0], coord[-1], y[0] - 0.5, y[-1] + 0.5],
            interpolation='nearest',
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(self.display_dict.get('cbar_label', 'Amplitude'))

        ax.set_xlabel(self.display_dict.get('xlabel', self.axis))
        ylabel = self.label_column if isinstance(self.label_column, str) else (
            'Bin' if self.label_column is False else scan.scan_parameter
        )
        ax.set_ylabel(self.display_dict.get('ylabel', ylabel))
        n_ticks = min(self.display_dict.get('n_yticks', 10), len(y))
        tick_idx = np.linspace(0, len(y) - 1, n_ticks).round().astype(int)
        ax.set_yticks(y[tick_idx])
        ax.set_yticklabels([labels[i] for i in tick_idx])

        xlims = self.display_dict.get('xlims', None)
        if xlims is not None:
            ax.set_xlim(xlims)

        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        ax.set_title(scan.scan_data_title(f'{diag} {self.axis} lineout mean-per-bin waterfall'))

        self.last_export = {
            'stack': stack,
            'coord': coord,
            'bins': np.asarray(bins, dtype=float),
            'labels': np.asarray(labels, dtype=object),
        }

        return fig, ax
