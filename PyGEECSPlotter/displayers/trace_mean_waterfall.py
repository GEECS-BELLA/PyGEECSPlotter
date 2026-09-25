from typing import Optional, Dict, Any, Iterable

import numpy as np
import matplotlib.pyplot as plt

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer
from PyGEECSPlotter.displayers.trace_waterfall import COMPONENTS
from PyGEECSPlotter.displayers._trace_binning import mean_traces_per_bin, bin_labels


class TraceMeanWaterfall(ScanDisplayer):
    """
    ``TraceMeanPerBin``'s per-bin mean traces, stacked as a waterfall image —
    the per-bin analogue of ``TraceWaterfall``.

    Each row is one bin's mean amplitude trace (intensity only; there is no
    single phase to show once shots are averaged together, so unlike
    ``FrogAnalyzer.display_data`` this never draws phase). Rows are in bin
    order, labelled on the y-axis by ``label_column`` (default
    ``scan.scan_parameter``, summarised per bin) rather than a raw bin index.

    Requires shots to share an axis — give the analyzer a common grid (for
    ``FrogAnalyzer``, ``t_grid`` / ``wl_grid`` in ``analyzer_dict``).

    Parameters
    ----------
    analyzer : DiagnosticAnalyzer
        Per-shot analyzer returning a dict of trace DataFrames.
    bg : optional
        Background spec forwarded to the per-shot pipeline.
    bins : iterable of int, optional
        Bin numbers to render. Defaults to all unique bins in ``active_data``.
    component : {'temporal', 'spectral'}, optional
        Which trace to stack.
    label_column : str, optional
        Scan-data column to label each row with, instead of the raw bin
        number — e.g. the scan parameter's value at that bin. Defaults to
        ``scan.scan_parameter``, summarised per bin with
        ``scan.compute_bin_summary(mode='mean')``. Pass ``False`` to fall
        back to plain ``'Bin {n}'`` labels.
    label_fmt : str, optional
        Format string for the label column's value, e.g. ``'{:.3g} mm'``.
    display_dict : dict, optional
        Style overrides: ``figsize``, ``cmap``, ``vmin``, ``vmax``, ``xlims``,
        ``normalise_rows``, ``n_yticks``.
    """

    def __init__(
        self,
        analyzer,
        bg=None,
        bins: Optional[Iterable[int]] = None,
        component: str = 'temporal',
        label_column=None,
        label_fmt: str = '{:.4g}',
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        if component not in COMPONENTS:
            raise ValueError(
                f"component must be one of {sorted(COMPONENTS)}, got {component!r}."
            )
        diag = analyzer.output_diagnostic or analyzer.diagnostic
        super().__init__(name=f'{diag}_{component}_mean_waterfall', display_dict=display_dict)
        self.analyzer = analyzer
        self.bg = bg
        self.bins = bins
        self.component = component
        self.label_column = label_column
        self.label_fmt = label_fmt

    # ------------------------------------------------------------------
    def display(self, scan, fig=None, ax=None):
        key, axis_col, amp_col, axis_label = COMPONENTS[self.component]

        bins, per_bin = mean_traces_per_bin(scan, self.analyzer, bg=self.bg, bins=self.bins)
        if all(p is None for p in per_bin):
            raise RuntimeError(f"{type(self).__name__}: no bins produced data.")

        labels = bin_labels(scan, bins, label_column=self.label_column, label_fmt=self.label_fmt)

        axis = next(
            (entry[0][key][axis_col].values for entry in per_bin if entry is not None), None
        )
        rows = []
        for entry in per_bin:
            if entry is None:
                rows.append(np.full_like(axis, np.nan, dtype=float))
                continue
            mean_data, _ = entry
            rows.append(np.asarray(mean_data[key][amp_col].values, dtype=float))
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
            extent=[axis[0], axis[-1], y[0] - 0.5, y[-1] + 0.5],
            interpolation='nearest',
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(self.display_dict.get('cbar_label', f'{self.component.capitalize()} Amp (norm.)'))

        ax.set_xlabel(axis_label)
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
        ax.set_title(scan.scan_data_title(f'{diag} {self.component} mean-per-bin waterfall'))

        self.last_export = {
            'stack': stack,
            'axis': axis,
            'bins': np.asarray(bins, dtype=float),
            'labels': np.asarray(labels, dtype=object),
        }

        return fig, ax
