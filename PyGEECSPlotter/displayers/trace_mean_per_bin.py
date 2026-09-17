from typing import Optional, Dict, Any, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer
from PyGEECSPlotter.displayers.trace_waterfall import COMPONENTS


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
        analyzer's own default panel layout.
    ncols : int, optional
        Number of columns in the grid.
    show_std : bool, optional
        Shade +/- 1 sigma across the bin's shots around the mean.
    suppress_labels : bool, optional
        Strip inner axis labels for a cleaner grid (default True).
    display_dict : dict, optional
        Style overrides, forwarded to the analyzer's ``display_data``.
    """

    def __init__(
        self,
        analyzer,
        bg=None,
        bins: Optional[Iterable[int]] = None,
        component: str = 'temporal',
        ncols: int = 4,
        show_std: bool = True,
        suppress_labels: bool = True,
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        if component not in COMPONENTS and component != 'both':
            raise ValueError(
                f"component must be 'both' or one of {sorted(COMPONENTS)}, "
                f"got {component!r}."
            )
        diag = analyzer.output_diagnostic or analyzer.diagnostic
        super().__init__(name=f'{diag}_{component}_mean_per_bin', display_dict=display_dict)
        self.analyzer = analyzer
        self.bg = bg
        self.bins = bins
        self.component = component
        self.ncols = ncols
        self.show_std = show_std
        self.suppress_labels = suppress_labels

    # ------------------------------------------------------------------
    def _mean_traces(self, scan):
        """Mean (and std) trace dict for each bin, in bin order."""
        if self.bins is None:
            bins = np.unique(scan.active_data['temp Bin number'])
        else:
            bins = np.asarray(list(self.bins))

        per_bin = []
        saved = scan.save_mask()
        try:
            for b in bins:
                scan.restore_mask(saved)
                scan.filter_scan_data('temp Bin number', b - 0.1, b + 0.1)
                per_bin.append(self._mean_over_active(scan))
        finally:
            scan.restore_mask(saved)

        return bins, per_bin

    def _mean_over_active(self, scan):
        """Stack the currently-active shots and average them column-wise."""
        stacks: Dict[str, List[pd.DataFrame]] = {}

        for _, data, _, _ in scan._iter_shots(self.analyzer, bg=self.bg, show_progress=False):
            if data is None:
                continue
            for key, df in data.items():
                stacks.setdefault(key, []).append(df)

        if not stacks:
            return None

        mean_data, std_data = {}, {}
        for key, dfs in stacks.items():
            shapes = {df.shape for df in dfs}
            if len(shapes) > 1:
                raise ValueError(
                    f"{type(self).__name__} needs every shot on the same axis, "
                    f"but the '{key}' traces have shapes {sorted(shapes)}. Give "
                    f"the analyzer a common grid (e.g. analyzer_dict "
                    f"{{'t_grid': (lo, hi, n)}})."
                )
            arr = np.stack([df.values.astype(float) for df in dfs], axis=0)
            cols = dfs[0].columns
            mean_data[key] = pd.DataFrame(np.nanmean(arr, axis=0), columns=cols)
            std_data[key] = pd.DataFrame(np.nanstd(arr, axis=0), columns=cols)

        return mean_data, std_data

    def display(self, scan, fig=None, ax=None):
        bins, per_bin = self._mean_traces(scan)
        if all(p is None for p in per_bin):
            raise RuntimeError(f"{type(self).__name__}: no bins produced data.")

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

        for k, (b, entry) in enumerate(zip(bins, per_bin)):
            a = axes.flat[k]
            if entry is None:
                a.set_visible(False)
                continue
            mean_data, std_data = entry
            self.analyzer.display_data(
                mean_data, display_dict=panel_dict, fig=fig, ax=a,
                title=f'Bin {int(b)}',
            )
            if self.show_std and self.component in COMPONENTS:
                self._shade_std(a, mean_data, std_data)
            a.set_title(f'Bin {int(b)}')
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
