from typing import Optional, Dict, Any, Iterable, List

import numpy as np
import matplotlib.pyplot as plt

from PyGEECSPlotter.displayers.multi_scan_displayer import MultiScanDisplayer
from PyGEECSPlotter.displayers._lineout_binning import (mean_lineouts_per_bin,
                                                       mean_lineouts_over_active, on_grid)
from PyGEECSPlotter.displayers._trace_binning import bin_values

# Line style per lineout axis; colour is per scan.
_AXIS_LINESTYLE = ['-', '--', ':', '-.']


class MultiScanLineoutMean(MultiScanDisplayer):
    """
    Mean lineouts of two or more scans overlaid, one colour per scan: the
    multi-scan counterpart of ``LineoutMeanPerBin``.

    ``per_bin=True`` (default): one panel per bin number, each overlaying
    every scan's mean lineout in that bin. ``per_bin=False``: one panel per
    lineout axis (e.g. ``x``, ``y``), each with one line per scan — its mean
    over all its active shots (no binning; suits no-scans).

    Parameters
    ----------
    analyzer : DiagnosticAnalyzer
        Per-shot analyzer returning a lineout-style ``aux`` dict
        (``{'x': x, 'x_lo': ..., ...}``), applied to every scan.
    axes : iterable of str, optional
        Lineout coordinates to show, e.g. ``['x']``. Default: every one
        found. With ``per_bin=True`` several axes share each panel (line
        style per axis).
    per_bin : bool, optional
        Per-bin panels (default) or one whole-scan mean per scan.
    bins : iterable of int, optional
        Bin numbers to show (``per_bin``). Default: every bin of any scan;
        a scan without a bin leaves it out of that panel.
    ncols : int, optional
        Grid columns for ``per_bin`` (default 4).
    show_std : bool, optional
        Shade +/- 1 sigma across shots.
    label_by, label_fmt : optional
        Legend entry per scan; see ``scan_label``.
    bg : optional
        One background for every scan, or a list with one per scan.
    align : {'snap', 'interp'}, optional
        How shifted lineout axes are combined; see ``on_grid``.
    output_dir : str, optional
        Where files go; default the first scan's analysis directory.
    display_dict : dict, optional
        ``figsize``, ``cmap`` (scan colours), ``std_alpha``,
        ``panel_aspect`` / ``panel_width``, ``xlabel``, ``ylabel``,
        ``bin_label_fmt`` (panel title value format, default ``'{:.4g}'``).

    Notes
    -----
    Bins are matched by bin number, not by parameter value; each panel's
    title gives every scan's mean scan-parameter value in that bin when
    they differ. Lineout axes must not move from shot to shot within a scan
    (no ``auto_roi`` for ``align='snap'`` unless centred), as for the
    other lineout displayers.

    ``last_export`` has ``labels`` and, per scan index ``k`` and axis,
    ``{axis}_coord_{k}`` and ``{axis}_mean_{k}`` / ``{axis}_std_{k}`` —
    shape (n_bins, n) with ``bins`` for ``per_bin``, else (n,).
    """

    def __init__(
        self,
        analyzer,
        axes: Optional[Iterable[str]] = None,
        per_bin: bool = True,
        bins: Optional[Iterable[int]] = None,
        ncols: int = 4,
        show_std: bool = True,
        label_by='scan',
        label_fmt: str = '{:.3g}',
        bg=None,
        align: str = 'snap',
        display_dict: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        output_subdir: Optional[str] = None,
        timestamp_files: bool = True,
    ):
        diag = analyzer.output_diagnostic or analyzer.diagnostic
        suffix = 'lineout_mean_per_bin' if per_bin else 'lineout_mean'
        super().__init__(name=f'{diag}_{suffix}', label_by=label_by, label_fmt=label_fmt,
                         display_dict=display_dict, output_dir=output_dir,
                         output_subdir=output_subdir, timestamp_files=timestamp_files)
        self.analyzer = analyzer
        self.axes = list(axes) if axes is not None else None
        self.per_bin = per_bin
        self.bins = bins
        self.ncols = ncols
        self.show_std = show_std
        self.bg = bg
        self.align = align

    def _bg(self, k):
        return self.bg[k] if isinstance(self.bg, (list, tuple)) else self.bg

    def display(self, scans, fig=None, ax=None):
        scans = self._check_scans(scans)
        labels = self._labels(scans)
        colors = self._colors(len(scans))
        self.last_export = {'labels': np.asarray(labels, dtype=object)}
        if self.per_bin:
            return self._display_per_bin(scans, labels, colors)
        return self._display_whole(scans, labels, colors)

    # ------------------------------------------------------------------
    def _display_whole(self, scans, labels, colors):
        means = [mean_lineouts_over_active(s, self.analyzer, bg=self._bg(k), axes=self.axes,
                                           align=self.align)
                 for k, s in enumerate(scans)]
        if all(m is None for m in means):
            raise RuntimeError(f"{type(self).__name__}: no scan produced a lineout.")
        axes_present = self._axes_present(means)

        n = len(axes_present)
        fig, axes_arr = plt.subplots(1, n, figsize=self._grid_figsize(n, 1, panel=(4, 3)),
                                     constrained_layout=True, squeeze=False)
        for a, axis in zip(axes_arr.flat, axes_present):
            for k, (entry, label, color) in enumerate(zip(means, labels, colors)):
                if entry is None or axis not in entry:
                    continue
                coord, mean, std = entry[axis]
                self._plot_line(a, coord, mean, std, color, '-', label)
                self.last_export[f'{axis}_coord_{k}'] = coord
                self.last_export[f'{axis}_mean_{k}'] = mean
                self.last_export[f'{axis}_std_{k}'] = std
            a.set_title(axis)
            a.set_xlabel(self.display_dict.get('xlabel', axis))
        axes_arr.flat[0].set_ylabel(self.display_dict.get('ylabel', 'mean lineout'))
        axes_arr.flat[0].legend(fontsize='small')
        self._apply_panel_aspect(axes_arr)
        fig.suptitle(self._title(scans, f'{self._diag()} mean lineouts'))
        return fig, axes_arr

    # ------------------------------------------------------------------
    def _display_per_bin(self, scans, labels, colors):
        if self.bins is not None:
            bins = np.asarray(list(self.bins))
        else:
            bins = np.unique(np.concatenate(
                [np.unique(s.active_data['temp Bin number']) for s in scans]))

        # per scan: {bin: {axis: (coord, mean, std)}}
        per_scan = []
        for k, s in enumerate(scans):
            scan_bins = [b for b in bins if np.any(s.active_data['temp Bin number'] == b)]
            _, entries = mean_lineouts_per_bin(s, self.analyzer, bg=self._bg(k), bins=scan_bins,
                                               axes=self.axes, align=self.align)
            per_scan.append(dict(zip(scan_bins, entries)))
        if all(e is None for d in per_scan for e in d.values()):
            raise RuntimeError(f"{type(self).__name__}: no bin of any scan produced a lineout.")
        axes_present = self._axes_present([e for d in per_scan for e in d.values()])
        styles = {axis: _AXIS_LINESTYLE[i % len(_AXIS_LINESTYLE)]
                  for i, axis in enumerate(axes_present)}

        n_panels = len(bins)
        ncols = min(self.ncols, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes_arr = plt.subplots(nrows, ncols,
                                     figsize=self._grid_figsize(ncols, nrows, panel=(3.5, 3)),
                                     constrained_layout=True, squeeze=False, sharex=True, sharey=True)
        titles = self._bin_titles(scans, bins)
        for i, (b, title) in enumerate(zip(bins, titles)):
            a = axes_arr.flat[i]
            for k, (by_bin, color) in enumerate(zip(per_scan, colors)):
                entry = by_bin.get(b)
                if entry is None:
                    continue
                for axis in axes_present:
                    if axis in entry:
                        coord, mean, std = entry[axis]
                        self._plot_line(a, coord, mean, std, color, styles[axis], None)
            a.set_title(title, fontsize='small')
        for i in range(n_panels, nrows * ncols):
            axes_arr.flat[i].set_visible(False)

        self._shared_legend(fig, axes_arr, labels, colors, axes_present, styles)
        self._apply_panel_aspect(axes_arr)
        fig.suptitle(self._title(scans, f'{self._diag()} lineout mean per bin'))
        self.last_export.update(self._per_bin_export(bins, per_scan, axes_present))
        return fig, axes_arr

    def _bin_titles(self, scans, bins):
        """'Bin n: value' per bin; every scan's value, '/'-separated, if they differ."""
        fmt = self.display_dict.get('bin_label_fmt', '{:.4g}')
        values = np.stack([bin_values(s, bins, s.scan_parameter) for s in scans])
        titles = []
        for i, b in enumerate(bins):
            col = values[:, i]
            finite = col[np.isfinite(col)]
            if not len(finite):
                titles.append(f'Bin {int(b)}')
            elif np.allclose(finite, finite[0], rtol=1e-3):
                titles.append(fmt.format(finite[0]))
            else:
                titles.append(' / '.join(fmt.format(v) if np.isfinite(v) else '—' for v in col))
        return titles

    def _shared_legend(self, fig, axes_arr, labels, colors, axes_present, styles):
        handles = [plt.Line2D([], [], color=c, label=l) for l, c in zip(labels, colors)]
        if len(axes_present) > 1:
            handles += [plt.Line2D([], [], color='0.3', ls=styles[a], label=a)
                        for a in axes_present]
        axes_arr.flat[0].legend(handles=handles, fontsize='x-small')

    def _per_bin_export(self, bins, per_scan, axes_present):
        export = {'bins': np.asarray(bins, dtype=float)}
        for k, by_bin in enumerate(per_scan):
            for axis in axes_present:
                ref = next((e[axis][0] for e in by_bin.values() if e is not None and axis in e),
                           None)
                if ref is None:
                    continue
                mean = np.full((len(bins), len(ref)), np.nan)
                std = np.full((len(bins), len(ref)), np.nan)
                for i, b in enumerate(bins):
                    entry = by_bin.get(b)
                    if entry is None or axis not in entry:
                        continue
                    c, m, s = entry[axis]
                    mean[i] = on_grid(np.asarray(c, float), np.asarray(m, float), ref, self.align)
                    std[i] = on_grid(np.asarray(c, float), np.asarray(s, float), ref, self.align)
                export[f'{axis}_coord_{k}'] = ref
                export[f'{axis}_mean_{k}'] = mean
                export[f'{axis}_std_{k}'] = std
        return export

    # ------------------------------------------------------------------
    def _axes_present(self, entries) -> List[str]:
        if self.axes is not None:
            return self.axes
        return sorted({axis for e in entries if e is not None for axis in e})

    def _plot_line(self, a, coord, mean, std, color, ls, label):
        a.plot(coord, mean, color=color, ls=ls, lw=1.2, label=label)
        if self.show_std:
            a.fill_between(coord, mean - std, mean + std, color=color, lw=0,
                           alpha=self.display_dict.get('std_alpha', 0.2))

    def _diag(self):
        return self.analyzer.output_diagnostic or self.analyzer.diagnostic
