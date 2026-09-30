from typing import Optional, Dict, Any, Iterable

import numpy as np
import matplotlib.pyplot as plt

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer
from PyGEECSPlotter.displayers.trace_waterfall import COMPONENTS
from PyGEECSPlotter.displayers._trace_binning import mean_traces_per_bin, bin_labels


class TraceMeanPerBin(ScanDisplayer):
    """
    Grid of per-bin mean traces — the 1-D analogue of ``MeanImagePerBin``.

    One panel per bin, each the mean over that bin's shots, rendered through
    the analyzer's own ``display_data`` so the panels match the per-shot view.

    Averaging is done here rather than via ``scan.aggregate_per_bin`` because
    that helper expects array-like per-shot data, while a trace analyzer
    returns a dict of DataFrames. The bin iteration uses the same
    ``save_mask`` / ``filter_scan_data`` / ``restore_mask`` bracket as
    ``aggregate_per_bin``, so the scan's filters are left untouched.

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
    component : {'temporal', 'spectral', 'both'}, optional
        Which trace to draw in each panel. ``'both'`` defers to the
        analyzer's own default panel layout (only valid with ``overlay=False``).
    overlay : bool, optional
        Draw every bin's mean trace on one shared axis instead of a grid, one
        line per bin, colour-mapped and legended by ``label_column``. Needs
        ``component`` to be a single trace (not ``'both'``).
    ncols : int, optional
        Number of columns in the grid (ignored when ``overlay=True``).
    show_std : bool, optional
        Shade +/- 1 sigma across the bin's shots around the mean.
    label_column : str, optional
        Scan-data column to label each bin with, instead of the raw bin
        number — e.g. the scan parameter's value at that bin. Defaults to
        ``scan.scan_parameter``, summarised per bin with
        ``scan.compute_bin_summary(mode='mean')``. Pass ``False`` to fall
        back to plain ``'Bin {n}'`` labels.
    label_fmt : str, optional
        Format string for the label column's value, e.g. ``'{:.3g} mm'``.
    suppress_labels : bool, optional
        Strip inner axis labels for a cleaner grid (default True, grid mode
        only).
    display_dict : dict, optional
        Style overrides, forwarded to the analyzer's ``display_data``.
    """

    def __init__(
        self,
        analyzer,
        bg=None,
        bins: Optional[Iterable[int]] = None,
        component: str = 'temporal',
        overlay: bool = False,
        ncols: int = 4,
        show_std: bool = True,
        label_column=None,
        label_fmt: str = '{:.4g}',
        suppress_labels: bool = True,
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        if component not in COMPONENTS and component != 'both':
            raise ValueError(
                f"component must be 'both' or one of {sorted(COMPONENTS)}, "
                f"got {component!r}."
            )
        if overlay and component == 'both':
            raise ValueError("overlay=True needs a single component, not 'both'.")
        diag = analyzer.output_diagnostic or analyzer.diagnostic
        super().__init__(name=f'{diag}_{component}_mean_per_bin', display_dict=display_dict)
        self.analyzer = analyzer
        self.bg = bg
        self.bins = bins
        self.component = component
        self.overlay = overlay
        self.ncols = ncols
        self.show_std = show_std
        self.label_column = label_column
        self.label_fmt = label_fmt
        self.suppress_labels = suppress_labels

    # ------------------------------------------------------------------
    def display(self, scan, fig=None, ax=None):
        bins, per_bin = mean_traces_per_bin(scan, self.analyzer, bg=self.bg, bins=self.bins)
        if all(p is None for p in per_bin):
            raise RuntimeError(f"{type(self).__name__}: no bins produced data.")

        labels = bin_labels(scan, bins, label_column=self.label_column, label_fmt=self.label_fmt)
        self.last_export = self._build_export(bins, per_bin, labels)

        if self.overlay:
            return self._display_overlay(scan, bins, per_bin, labels, fig=fig, ax=ax)
        return self._display_grid(scan, bins, per_bin, labels)

    def _build_export(self, bins, per_bin, labels):
        """
        Stack each bin's mean/std trace into one 2D array per component, so
        the whole grid/overlay can be reloaded without recomputing anything.
        """
        components = COMPONENTS if self.component == 'both' else \
            {self.component: COMPONENTS[self.component]}

        export = {'bins': np.asarray(bins, dtype=float), 'labels': np.asarray(labels, dtype=object)}
        for comp, (key, axis_col, amp_col, _) in components.items():
            axis_ref = next(
                (entry[0][key][axis_col].values for entry in per_bin if entry is not None), None
            )
            if axis_ref is None:
                continue
            mean_stack = np.full((len(bins), len(axis_ref)), np.nan)
            std_stack = np.full((len(bins), len(axis_ref)), np.nan)
            for k, entry in enumerate(per_bin):
                if entry is None:
                    continue
                mean_data, std_data = entry
                mean_stack[k] = mean_data[key][amp_col].values
                std_stack[k] = std_data[key][amp_col].values
            export[f'{comp}_axis'] = axis_ref
            export[f'{comp}_mean'] = mean_stack
            export[f'{comp}_std'] = std_stack

        return export

    def _display_overlay(self, scan, bins, per_bin, labels, fig=None, ax=None):
        """Every bin's mean trace on one shared axis, colour-mapped and legended."""
        key, axis_col, amp_col, axis_label = COMPONENTS[self.component]
        fig, ax = self._new_fig(fig, ax, figsize=(7, 5))

        cmap = plt.get_cmap(self.display_dict.get('cmap', 'viridis'))
        n = len(bins)
        for k, (entry, label) in enumerate(zip(per_bin, labels)):
            if entry is None:
                continue
            mean_data, std_data = entry
            df = mean_data[key]
            axis, amp = df[axis_col].values, df[amp_col].values
            color = cmap(k / max(n - 1, 1))
            ax.plot(axis, amp, color=color, label=label)
            if self.show_std:
                std = std_data[key][amp_col].values
                ax.fill_between(axis, amp - std, amp + std, color=color,
                                 alpha=self.display_dict.get('std_alpha', 0.15), lw=0)

        ax.set_xlabel(self.display_dict.get('xlabel', axis_label))
        ax.set_ylabel(self.display_dict.get('ylabel', f'{self.component.capitalize()} Amp (norm.)'))
        xlims = self.display_dict.get(self._xlims_key(), None)
        if xlims is not None:
            ax.set_xlim(xlims)

        legend_title = self.display_dict.get(
            'legend_title',
            self.label_column if isinstance(self.label_column, str) else (
                'Bin' if self.label_column is False else scan.scan_parameter
            ),
        )
        ax.legend(title=legend_title, fontsize='small', ncol=max(1, n // 12))

        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        ax.set_title(scan.scan_data_title(f'{diag} {self.component} mean per bin'))
        return fig, ax

    def _display_grid(self, scan, bins, per_bin, labels):
        n_panels = len(bins)
        ncols = min(self.ncols, n_panels)
        nrows = int(np.ceil(n_panels / ncols))

        figsize = self.display_dict.get('figsize', (3.5 * ncols, 3 * nrows))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, constrained_layout=True, squeeze=False,
        )

        panel_dict = dict(self.display_dict)
        panel_dict['panel'] = self.component

        # Let every panel share one x range, otherwise each bin auto-zooms to
        # its own support and the panels can't be compared by eye.
        if self.component in COMPONENTS and self._xlims_key() not in panel_dict:
            shared = self._shared_xlims(per_bin)
            if shared is not None:
                panel_dict[self._xlims_key()] = shared

        for k, (b, entry, label) in enumerate(zip(bins, per_bin, labels)):
            a = axes.flat[k]
            if entry is None:
                a.set_visible(False)
                continue
            mean_data, std_data = entry
            self.analyzer.display_data(
                mean_data, display_dict=panel_dict, fig=fig, ax=a,
                title=label,
            )
            if self.show_std and self.component in COMPONENTS:
                self._shade_std(a, mean_data, std_data)
            a.set_title(label)
            if self.suppress_labels:
                a.set_xlabel(None)
                a.set_ylabel(None)
                # display_data twins a second axis for the phase; its label
                # repeats on every panel otherwise.
                for twin in fig.axes:
                    if twin is not a and twin.bbox.bounds == a.bbox.bounds:
                        twin.set_ylabel(None)

        for k in range(n_panels, nrows * ncols):
            axes.flat[k].set_visible(False)

        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        fig.suptitle(scan.scan_data_title(f'{diag} {self.component} mean per bin'))
        return fig, axes

    def _xlims_key(self):
        """The display_dict key the analyzer reads for this component's x range."""
        return 't_lims' if self.component == 'temporal' else 'wl_lims'

    def _shared_xlims(self, per_bin):
        """Union of every bin's signal support, so all panels share an x range."""
        key, axis_col, amp_col, _ = COMPONENTS[self.component]
        frac = self.display_dict.get('support_threshold', 0.01)

        lo, hi = np.inf, -np.inf
        for entry in per_bin:
            if entry is None:
                continue
            df = entry[0][key]
            axis = df[axis_col].values
            amp = df[amp_col].values
            peak = np.nanmax(amp) if np.any(np.isfinite(amp)) else np.nan
            if not np.isfinite(peak) or peak <= 0:
                continue
            idx = np.where(np.isfinite(amp) & (amp > frac * peak))[0]
            if idx.size:
                lo, hi = min(lo, axis[idx[0]]), max(hi, axis[idx[-1]])

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None

        pad = self.display_dict.get('support_pad', 0.25) * (hi - lo)
        return [lo - pad, hi + pad]

    def _shade_std(self, a, mean_data, std_data):
        key, axis_col, amp_col, _ = COMPONENTS[self.component]
        axis = mean_data[key][axis_col].values
        mean = mean_data[key][amp_col].values
        std = std_data[key][amp_col].values
        a.fill_between(
            axis, mean - std, mean + std,
            alpha=self.display_dict.get('std_alpha', 0.25),
            color=self.display_dict.get('std_color', None),
            lw=0,
        )
