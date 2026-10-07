from typing import Optional, Dict, Any, Iterable

import numpy as np

from PyGEECSPlotter.displayers.lineout_waterfall import LineoutWaterfall
from PyGEECSPlotter.displayers.lineout_mean_per_bin import LineoutMeanPerBin
from PyGEECSPlotter.displayers.lineout_mean_waterfall import LineoutMeanWaterfall

# Legend names for the aux keys OpticalSpectrumAnalyzer / CombinedVisNIRSpectrum return
_SPECTRUM_LABELS = {'wl': 'spectrum', 'vis_wl': 'VIS', 'nir_wl': 'NIR (scaled)'}


def _spectrum_defaults(display_dict):
    dd = {'cmap': 'viridis', 'xlabel': 'Wavelength [nm]', 'cbar_label': 'Counts'}
    dd.update(display_dict or {})
    return dd


class OpticalSpectrumWaterfall(LineoutWaterfall):
    """
    Every shot's optical spectrum stacked into one image: one row per shot,
    x = wavelength [nm], colour = counts.

    A ``LineoutWaterfall`` on the ``'wl'`` / ``'wl_lo'`` pair that
    ``OpticalSpectrumAnalyzer`` and ``CombinedVisNIRSpectrum`` put in
    ``aux``. Every shot must be on the same wavelength axis (raw spectra
    from one spectrometer are; ``wl_lin`` guarantees it).

    Parameters
    ----------
    analyzer : OpticalSpectrumAnalyzer or CombinedVisNIRSpectrum
    axis : str, optional
        ``'wl'`` (default; the combined spectrum for ``CombinedVisNIRSpectrum``),
        or ``'vis_wl'`` / ``'nir_wl'`` for one spectrometer of the combination.
    bg, y_column, overlay_y_values :
        As for ``LineoutWaterfall``.
    display_dict : dict, optional
        ``LineoutWaterfall`` keys (``cmap``, ``vmin``, ``vmax``, ``xlims``,
        ``normalise_rows``, ``n_yticks``, ...).
    output_subdir, timestamp_files :
        As for ``ScanDisplayer``.
    """

    def __init__(
        self,
        analyzer,
        axis: str = 'wl',
        bg=None,
        y_column=None,
        overlay_y_values: bool = False,
        display_dict: Optional[Dict[str, Any]] = None,
        output_subdir: Optional[str] = None,
        timestamp_files: bool = True,
    ):
        super().__init__(analyzer, axis=axis, bg=bg, y_column=y_column,
                         overlay_y_values=overlay_y_values,
                         display_dict=_spectrum_defaults(display_dict),
                         output_subdir=output_subdir, timestamp_files=timestamp_files)
        diag = analyzer.output_diagnostic or analyzer.diagnostic
        suffix = '' if axis == 'wl' else f'_{axis}'
        self.name = f'{diag}{suffix}_spectrum_waterfall'

    def display(self, scan, fig=None, ax=None):
        fig, ax = super().display(scan, fig=fig, ax=ax)
        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        what = '' if self.axis == 'wl' else f' {_SPECTRUM_LABELS.get(self.axis, self.axis)}'
        ax.set_title(scan.scan_data_title(f'{diag}{what} spectrum waterfall'))
        return fig, ax


class OpticalSpectrumMeanPerBin(LineoutMeanPerBin):
    """
    Mean optical spectrum per bin: one panel per bin, mean counts vs
    wavelength [nm] with a +/- 1 sigma band, on a shared y range so bins
    compare directly.

    A ``LineoutMeanPerBin`` on ``axes=['wl']``. For ``CombinedVisNIRSpectrum``
    pass ``axes=['wl', 'vis_wl', 'nir_wl']`` to overlay the combined spectrum
    on the two spectrometers it was stitched from.

    Parameters
    ----------
    analyzer : OpticalSpectrumAnalyzer or CombinedVisNIRSpectrum
    axes : iterable of str, optional
        Default ``['wl']``.
    bg, bins, ncols, show_std, label_column, label_fmt :
        As for ``LineoutMeanPerBin``.
    display_dict : dict, optional
        ``figsize``, ``panel_aspect`` (default 2.5; None for the plain
        shape), ``panel_width`` (inches, default 4.5), ``std_alpha``,
        ``xlims``, ``ylims`` (default shared: 0 to 1.1 x the largest mean),
        ``log`` (default False).
    output_subdir, timestamp_files :
        As for ``ScanDisplayer``.
    """

    def __init__(
        self,
        analyzer,
        axes: Optional[Iterable[str]] = None,
        bg=None,
        bins: Optional[Iterable[int]] = None,
        ncols: int = 3,
        show_std: bool = True,
        label_column=None,
        label_fmt: str = '{:.4g}',
        display_dict: Optional[Dict[str, Any]] = None,
        output_subdir: Optional[str] = None,
        timestamp_files: bool = True,
    ):
        super().__init__(analyzer, axes=list(axes) if axes is not None else ['wl'], bg=bg,
                         bins=bins, ncols=ncols, show_std=show_std,
                         label_column=label_column, label_fmt=label_fmt,
                         suppress_labels=False, display_dict=display_dict,
                         output_subdir=output_subdir, timestamp_files=timestamp_files)
        self.display_dict.setdefault('panel_aspect', 2.5)
        self.display_dict.setdefault('panel_width', 4.5)
        diag = analyzer.output_diagnostic or analyzer.diagnostic
        self.name = f'{diag}_spectrum_mean_per_bin'

    def display(self, scan, fig=None, ax=None):
        fig, axes = super().display(scan, fig=fig, ax=ax)
        dd = self.display_dict
        log = dd.get('log', False)
        ylims = dd.get('ylims')
        if ylims is None:
            tops = [np.nanmax(self.last_export[f'{a}_mean']) for a in self.axes
                    if f'{a}_mean' in self.last_export
                    and np.any(np.isfinite(self.last_export[f'{a}_mean']))]
            if tops:
                top = max(tops) * 1.1
                ylims = (top / 1e3, top) if log else (0, top)
        for a in axes.flat:
            if not a.get_visible():
                continue
            if log:
                a.set_yscale('log')
            if ylims is not None:
                a.set_ylim(ylims)
            if dd.get('xlims') is not None:
                a.set_xlim(dd['xlims'])
            a.grid(alpha=0.25, lw=0.5)
            spec = a.get_subplotspec()
            if spec.is_last_row():
                a.set_xlabel('Wavelength [nm]')
            else:
                a.set_xticklabels([])
            if spec.is_first_col():
                a.set_ylabel('Counts')
            else:
                a.set_yticklabels([])
            legend = a.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_text(_SPECTRUM_LABELS.get(text.get_text(), text.get_text()))
        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        fig.suptitle(scan.scan_data_title(f'{diag} mean spectrum per bin'))
        return fig, axes


class OpticalSpectrumMeanWaterfall(LineoutMeanWaterfall):
    """
    Each bin's mean optical spectrum as one row of an image: wavelength
    [nm] across, bin (labelled by the scan parameter) up, mean counts as
    colour. The per-bin counterpart of ``OpticalSpectrumWaterfall``; set
    ``display_dict['normalise_rows']`` to compare spectral shape across bins.

    Parameters
    ----------
    analyzer : OpticalSpectrumAnalyzer or CombinedVisNIRSpectrum
    axis : str, optional
        ``'wl'`` (default), ``'vis_wl'`` or ``'nir_wl'``.
    bg, bins, label_column, label_fmt, overlay_y_values :
        As for ``LineoutMeanWaterfall``.
    display_dict : dict, optional
        ``LineoutMeanWaterfall`` keys.
    output_subdir, timestamp_files :
        As for ``ScanDisplayer``.
    """

    def __init__(
        self,
        analyzer,
        axis: str = 'wl',
        bg=None,
        bins: Optional[Iterable[int]] = None,
        label_column=None,
        label_fmt: str = '{:.4g}',
        overlay_y_values: bool = False,
        display_dict: Optional[Dict[str, Any]] = None,
        output_subdir: Optional[str] = None,
        timestamp_files: bool = True,
    ):
        super().__init__(analyzer, axis=axis, bg=bg, bins=bins, label_column=label_column,
                         label_fmt=label_fmt, overlay_y_values=overlay_y_values,
                         display_dict=_spectrum_defaults(display_dict),
                         output_subdir=output_subdir, timestamp_files=timestamp_files)
        diag = analyzer.output_diagnostic or analyzer.diagnostic
        suffix = '' if axis == 'wl' else f'_{axis}'
        self.name = f'{diag}{suffix}_spectrum_mean_waterfall'

    def display(self, scan, fig=None, ax=None):
        fig, ax = super().display(scan, fig=fig, ax=ax)
        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        what = '' if self.axis == 'wl' else f' {_SPECTRUM_LABELS.get(self.axis, self.axis)}'
        ax.set_title(scan.scan_data_title(f'{diag}{what} mean spectrum per bin'))
        return fig, ax
