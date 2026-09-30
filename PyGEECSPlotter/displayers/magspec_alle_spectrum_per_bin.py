from typing import Optional, Dict, Any, Iterable

import numpy as np
from matplotlib.colors import LogNorm

from PyGEECSPlotter.displayers.lineout_mean_per_bin import LineoutMeanPerBin
from PyGEECSPlotter.displayers.lineout_mean_waterfall import LineoutMeanWaterfall


class MagSpecAllESpectrumPerBin(LineoutMeanPerBin):
    """
    Mean allE electron spectrum per bin: one panel per bin, mean charge
    density [pC/GeV] vs momentum [GeV/c] with a +/- 1 sigma band.

    A ``LineoutMeanPerBin`` on the ``'p'`` / ``'p_lo'`` pair (the spectrum
    on the analyzer's fixed momentum grid), with physical labels, a log
    y-axis and a shared y range so bins compare directly. Use
    ``MagSpecAllEReader(load='spec')`` to average the saved spectra instead
    of re-running the analysis.

    Parameters
    ----------
    analyzer : MagSpecAllEAnalyzer or MagSpecAllEReader
    bg : optional
        Background spec forwarded to the per-shot pipeline.
    bins : iterable of int, optional
        Bins to show; default all active bins.
    ncols : int, optional
        Grid columns (default 4).
    show_std : bool, optional
        Shade +/- 1 sigma across the bin's shots (default True).
    label_column, label_fmt :
        As for ``LineoutMeanPerBin`` (default: the scan parameter's mean
        value per bin).
    display_dict : dict, optional
        ``figsize``, ``std_alpha``, ``log`` (default True), ``xlims``,
        ``ylims`` (default: shared, from 1e-3 of the largest mean up to it).
    """

    def __init__(
        self,
        analyzer,
        bg=None,
        bins: Optional[Iterable[int]] = None,
        ncols: int = 4,
        show_std: bool = True,
        label_column=None,
        label_fmt: str = '{:.4g}',
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(analyzer, axes=['p'], bg=bg, bins=bins, ncols=ncols,
                         show_std=show_std, label_column=label_column,
                         label_fmt=label_fmt, suppress_labels=False,
                         display_dict=display_dict)
        diag = analyzer.output_diagnostic or analyzer.diagnostic
        self.name = f'{diag}_spectrum_mean_per_bin'

    def display(self, scan, fig=None, ax=None):
        fig, axes = super().display(scan, fig=fig, ax=ax)
        dd = self.display_dict
        log = dd.get('log', True)
        means = self.last_export.get('p_mean')
        ylims = dd.get('ylims')
        if ylims is None and means is not None and np.any(np.isfinite(means)):
            top = np.nanmax(means) * 1.5
            ylims = (top / 1e3, top) if log else (0, top)
        visible = [a for a in axes.flat if a.get_visible()]
        for a in visible:
            if log:
                a.set_yscale('log')
            if ylims is not None:
                a.set_ylim(ylims)
            if dd.get('xlims') is not None:
                a.set_xlim(dd['xlims'])
            a.grid(alpha=0.25, lw=0.5)
            spec = a.get_subplotspec()
            if spec.is_last_row():
                a.set_xlabel('Momentum [GeV/c]')
            else:
                a.set_xticklabels([])
            if spec.is_first_col():
                a.set_ylabel('pC/GeV')
            else:
                a.set_yticklabels([], minor=True)
                a.set_yticklabels([])
        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        fig.suptitle(scan.scan_data_title(f'{diag} mean spectrum per bin'))
        return fig, axes


class MagSpecAllESpectrumMeanWaterfall(LineoutMeanWaterfall):
    """
    Each bin's mean allE spectrum stacked as one row of an image: momentum
    [GeV/c] across, bin (labelled by the scan parameter) up, charge density
    [pC/GeV] as colour on a log scale. The per-bin counterpart of
    ``MagSpecAllEWaterfall``; most useful for real scans with many bins.

    Parameters
    ----------
    analyzer : MagSpecAllEAnalyzer or MagSpecAllEReader
    bg, bins, label_column, label_fmt :
        As for ``LineoutMeanWaterfall``.
    display_dict : dict, optional
        ``LineoutMeanWaterfall`` keys, plus ``log`` (default True; ``vmin``
        defaults to ``vmax / 1e3``).
    """

    def __init__(
        self,
        analyzer,
        bg=None,
        bins: Optional[Iterable[int]] = None,
        label_column=None,
        label_fmt: str = '{:.4g}',
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        dd = {'cmap': 'jet', 'xlabel': 'Momentum [GeV/c]', 'cbar_label': 'pC/GeV'}
        dd.update(display_dict or {})
        super().__init__(analyzer, axis='p', bg=bg, bins=bins, label_column=label_column,
                         label_fmt=label_fmt, display_dict=dd)
        diag = analyzer.output_diagnostic or analyzer.diagnostic
        self.name = f'{diag}_spectrum_mean_waterfall'

    def display(self, scan, fig=None, ax=None):
        fig, ax = super().display(scan, fig=fig, ax=ax)
        if self.display_dict.get('log', True):
            im = ax.images[-1]
            stack = self.last_export['stack']
            vmax = self.display_dict.get('vmax') or np.nanmax(stack)
            vmin = self.display_dict.get('vmin') or vmax / 1e3
            im.set_norm(LogNorm(vmin=vmin, vmax=vmax))
            # zero-charge bins would be masked white on a log scale
            cmap = im.get_cmap().copy()
            cmap.set_bad(cmap(0.0))
            cmap.set_under(cmap(0.0))
            im.set_cmap(cmap)
        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        ax.set_title(scan.scan_data_title(f'{diag} mean spectrum per bin'))
        return fig, ax
