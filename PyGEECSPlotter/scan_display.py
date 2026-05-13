# Base class for scan-level displays, plus a small library of concrete displayers.
# A ScanDisplayer takes a ScanDataAnalyzer and produces a (fig, ax) summary of
# the scan. Multiple displayers can be applied to the same scan; each one is
# composable and savable.

import os
from typing import Optional, Dict, Any, Iterable

import numpy as np
import matplotlib.pyplot as plt


class ScanDisplayer:
    """
    Base class for scan-level displays.

    Subclasses implement ``display(scan, *, fig=None, ax=None)`` which reads
    ``scan.active_data`` (and any other scan state) and returns ``(fig, ax)``.

    Use:

        scan.display_scan(MyDisplayer(...), save=True)
    """

    def __init__(
        self,
        name: str = "scan",
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.display_dict = dict(display_dict) if display_dict else {}

    # ------------------------------------------------------------------
    # Subclasses override this.
    # ------------------------------------------------------------------
    def display(self, scan, *, fig=None, ax=None):
        raise NotImplementedError(
            f"{type(self).__name__} must implement display(scan, *, fig, ax)."
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _new_fig(self, fig=None, ax=None, **defaults):
        """Make a new (fig, ax) if one wasn't supplied, using display_dict for figsize."""
        if fig is not None and ax is not None:
            return fig, ax
        figsize = self.display_dict.get('figsize', defaults.get('figsize', (6, 5)))
        return plt.subplots(constrained_layout=True, figsize=figsize)

    def save(self, fig, scan, suffix: str = "", dpi: int = 200):
        """Save ``fig`` under the scan's analysis directory."""
        analysis_dir = scan.get_scan_data_analysis_dir(make_dir=True)
        fname = f"Scan{int(scan.scan):03d}_{self.name}{suffix}.png"
        path = os.path.join(analysis_dir, fname)
        fig.savefig(path, dpi=dpi)
        return path


# ----------------------------------------------------------------------
# Concrete displayers
# ----------------------------------------------------------------------
class ScalarVsParameter(ScanDisplayer):
    """
    Scatter / errorbar plot of one scalar column vs the scan parameter
    (or any other column).

    Parameters
    ----------
    y_col : str
        Column to plot on the y-axis. Must exist in ``scan.active_data``.
    x_col : str, optional
        Column to plot on the x-axis. Defaults to ``scan.scan_parameter``.
    bin_summary : {'none', 'mean', 'median', 'std_err'}, optional
        If not ``'none'``, also overlays a per-bin summary using
        ``scan.compute_bin_summary``.
    display_dict : dict, optional
        Style overrides: ``color``, ``alpha``, ``marker``, ``bin_color``,
        ``figsize``, ``ylabel``, ``xlabel``.
    """

    def __init__(
        self,
        y_col: str,
        x_col: Optional[str] = None,
        bin_summary: str = 'none',
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=f"{y_col}_vs_param", display_dict=display_dict)
        self.y_col = y_col
        self.x_col = x_col
        self.bin_summary = bin_summary

    def display(self, scan, *, fig=None, ax=None):
        fig, ax = self._new_fig(fig, ax)
        df = scan.active_data
        x_col = self.x_col or scan.scan_parameter

        if self.y_col not in df.columns:
            raise KeyError(f"Column '{self.y_col}' not in scan.active_data.")
        if x_col not in df.columns:
            raise KeyError(f"Column '{x_col}' not in scan.active_data.")

        ax.scatter(
            df[x_col],
            df[self.y_col],
            color=self.display_dict.get('color', 'k'),
            marker=self.display_dict.get('marker', 'o'),
            alpha=self.display_dict.get('alpha', 0.5),
            label='shots',
        )

        if self.bin_summary != 'none':
            center, spread = scan.compute_bin_summary(mode=self.bin_summary)
            if x_col in center.columns and self.y_col in center.columns:
                ax.errorbar(
                    center[x_col],
                    center[self.y_col],
                    yerr=spread[self.y_col],
                    fmt='o',
                    color=self.display_dict.get('bin_color', 'C3'),
                    capsize=3,
                    label=f'per-bin {self.bin_summary}',
                )
                ax.legend()

        ax.set_xlabel(self.display_dict.get('xlabel', x_col))
        ax.set_ylabel(self.display_dict.get('ylabel', self.y_col))
        ax.set_title(scan.scan_data_title())
        return fig, ax


class CorrelationHeatmap(ScanDisplayer):
    """
    Heatmap of pairwise correlations between selected scalar columns.

    Parameters
    ----------
    columns : iterable of str
        Columns to include. Missing columns are silently dropped.
    method : {'pearson', 'spearman', 'kendall'}, optional
        Correlation method (forwarded to ``DataFrame.corr``).
    annotate : bool, optional
        If True, write the numeric correlation in each cell.
    display_dict : dict, optional
        Style overrides: ``cmap``, ``figsize``, ``vmin``, ``vmax``.
    """

    def __init__(
        self,
        columns: Iterable[str],
        method: str = 'pearson',
        annotate: bool = True,
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=f"corr_{method}", display_dict=display_dict)
        self.columns = list(columns)
        self.method = method
        self.annotate = annotate

    def display(self, scan, *, fig=None, ax=None):
        n = max(len(self.columns), 4)
        fig, ax = self._new_fig(fig, ax, figsize=(0.6 * n + 2, 0.6 * n + 2))

        cols = [c for c in self.columns if c in scan.active_data.columns]
        if not cols:
            raise ValueError("None of the requested columns are present in scan.active_data.")

        corr = scan.active_data[cols].corr(method=self.method)
        vmin = self.display_dict.get('vmin', -1)
        vmax = self.display_dict.get('vmax', 1)
        cmap = self.display_dict.get('cmap', 'RdBu_r')

        im = ax.imshow(corr.values, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols)

        if self.annotate:
            for i in range(len(cols)):
                for j in range(len(cols)):
                    val = corr.values[i, j]
                    if np.isfinite(val):
                        ax.text(
                            j, i, f"{val:.2f}",
                            ha='center', va='center',
                            color='white' if abs(val) > 0.5 else 'black',
                            fontsize=8,
                        )

        fig.colorbar(im, ax=ax, label=f'{self.method} correlation')
        ax.set_title(scan.scan_data_title('Correlation'))
        return fig, ax


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

    def display(self, scan, *, fig=None, ax=None):
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
