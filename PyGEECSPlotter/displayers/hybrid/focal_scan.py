from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer
from PyGEECSPlotter.displayers.images.peak_width_images import PeakWidthImages


class FocalScanDisplayer(ScanDisplayer):
    """
    One-figure overview of a laser focal scan (scan parameter = the stage
    moving through focus).

    Top row: the max-counts shot from the peak bin of ``max_counts_col``,
    and from the bins ``width_multiple`` half widths below and above it
    (see ``PeakWidthImages``).
    Bottom row: max counts vs the scan parameter (left axis) with the 2nd
    moments x, y (left panel) and x45, y45 (right panel) on a right axis
    shared between the two panels. Image labels give the stage position.

    Parameters
    ----------
    analyzer : ImageAnalyzer
        Renders the images (``display_data``).
    max_counts_col : str
        Column of per-shot max counts, e.g. ``'HPDmodeMaxCounts'``.
    moment_cols : dict, optional
        ``{'x': col, 'y': col, 'x45': col, 'y45': col}``. Default: the
        analyzer's own ``'<diag> 2nd moment <axis>'`` columns written by
        ``analyze_scan`` (2*sigma).
    bg : optional
        Background forwarded to the per-shot pipeline for the images.
    width_at, width_multiple : float, optional
        See ``PeakWidthImages``. Defaults: FWHM, images at +-1 HWHM.
    label_fmt : str, optional
        Format of the stage position in the image labels.
    show_lineouts : bool, optional
        Overlay x / y lineouts through the origin on each image (default
        True), drawn inside the view window (``axlims``). Styled by the
        analyzer display_dict's ``lineout_*`` keys.
    display_dict : dict, optional
        ``figsize`` (default (12, 8)); ``vmin`` / ``vmax`` for the images
        (default: shared range of the three images); ``show_shots``
        (default True) draws every shot faintly behind the bin means;
        ``moment_ylabel``; ``axlims`` (window around the beam, as in
        ``ImageAnalyzer.display_data``) for the images.
    """

    _MOMENT_STYLE = {'x': 'C0', 'y': 'C1', 'x45': 'C0', 'y45': 'C1'}

    def __init__(
        self,
        analyzer,
        max_counts_col: str,
        moment_cols: Optional[Dict[str, str]] = None,
        bg=None,
        width_at: float = 0.5,
        width_multiple: float = 1.0,
        label_fmt: str = '{:.4g}',
        show_lineouts: bool = True,
        display_dict: Optional[Dict[str, Any]] = None,
        output_subdir: Optional[str] = None,
        timestamp_files: bool = True,
    ):
        diag = analyzer.output_diagnostic or analyzer.diagnostic
        super().__init__(name=f"{diag}_focal_scan", display_dict=display_dict,
                         output_subdir=output_subdir, timestamp_files=timestamp_files)
        self.analyzer = analyzer
        self.max_counts_col = max_counts_col
        self.moment_cols = moment_cols or {k: f'{diag} 2nd moment {k}'
                                           for k in ('x', 'y', 'x45', 'y45')}
        self.bg = bg
        self.width_at = width_at
        self.width_multiple = width_multiple
        self.show_lineouts = show_lineouts
        self.images = PeakWidthImages(analyzer, parameter=max_counts_col, width_at=width_at,
                                      width_multiple=width_multiple, bg=bg, label_fmt=label_fmt)

    def display(self, scan, fig=None, ax=None):
        active = scan.active_data
        x_col = scan.scan_parameter
        for c in [x_col, self.max_counts_col] + list(self.moment_cols.values()):
            if c not in active.columns:
                raise KeyError(f"Column '{c}' not in scan.active_data.")

        # PeakWidthImages labels are '<tag>\n<position>'; keep just the position
        panels = [(label.split('\n')[-1], d, rd)
                  for label, d, rd in self.images._collect_panels(scan)]
        peak = self.images.last_peak

        fig = plt.figure(constrained_layout=True,
                         figsize=self.display_dict.get('figsize', (12, 8)))
        # separate subfigures so the bottom row's inner axis labels don't
        # squeeze the middle image
        top, bottom = fig.subfigures(2, 1)
        img_axes = list(top.subplots(1, 3))
        axL, axR = bottom.subplots(1, 2)

        self._draw_images(fig, img_axes, panels)
        center, spread = scan.compute_bin_summary(mode='mean')
        twins = [self._draw_curves(a, active, center, spread, x_col, keys)
                 for a, keys in ((axL, ('x', 'y')), (axR, ('x45', 'y45')))]
        lo = min(t.get_ylim()[0] for t in twins)
        hi = max(t.get_ylim()[1] for t in twins)
        for t in twins:
            t.set_ylim(lo, hi)

        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        fig.suptitle(scan.scan_data_title(f'{diag} focal scan'))

        self.last_export = {
            'labels': np.asarray([lab for lab, _, _ in panels], dtype=object),
            'images': self._stack([d for _, d, _ in panels]),
            'image_bins': np.asarray([peak['bin_low'], peak['bin_peak'], peak['bin_high']]),
            'x_bin': peak['x'], 'max_counts_bin': peak['y'],
            'x_peak': peak['x_peak'], 'hwhm': peak['hwhm'],
            'x_low': peak['x_low'], 'x_high': peak['x_high'], 'level': peak['level'],
            'x_col': x_col, 'max_counts_col': self.max_counts_col,
        }
        for k, c in self.moment_cols.items():
            sub = center.set_index('temp Bin number').loc[peak['bins']]
            err = spread.set_index('temp Bin number').loc[peak['bins']]
            self.last_export[f'moment_{k}_bin'] = sub[c].to_numpy()
            self.last_export[f'moment_{k}_bin_err'] = err[c].to_numpy()
        return fig, [img_axes, axL, axR]

    def _draw_images(self, fig, img_axes, panels):
        data = [d for _, d, _ in panels if d is not None]
        dd = self.analyzer.display_dict
        keys = ('vmin', 'vmax', 'show_lineouts', 'axlims')
        saved = {k: dd[k] for k in keys if k in dd}
        dd['vmin'] = self.display_dict.get('vmin', dd.get('vmin', min(np.nanmin(d) for d in data)))
        dd['vmax'] = self.display_dict.get('vmax', dd.get('vmax', max(np.nanmax(d) for d in data)))
        dd['show_lineouts'] = self.show_lineouts
        dd['axlims'] = self.display_dict.get('axlims', dd.get('axlims'))
        try:
            for a, (label, d, rd) in zip(img_axes, panels):
                if d is None:
                    a.set_visible(False)
                    continue
                self.images._render_panel(fig, a, d, rd, label)
        finally:
            for k in keys:
                dd.pop(k, None)
            dd.update(saved)
        for a in img_axes[len(panels):]:
            a.set_visible(False)

    def _draw_curves(self, a, active, center, spread, x_col, keys):
        show_shots = self.display_dict.get('show_shots', True)
        if show_shots:
            a.scatter(active[x_col], active[self.max_counts_col], color='0.6', s=10, alpha=0.5)
        a.errorbar(center[x_col], center[self.max_counts_col], yerr=spread[self.max_counts_col],
                   fmt='o-', color='k', ms=4, capsize=2, label=self.max_counts_col)
        a.set_xlabel(x_col)
        a.set_ylabel(self.max_counts_col)

        t = a.twinx()
        for k in keys:
            c, color = self.moment_cols[k], self._MOMENT_STYLE[k]
            if show_shots:
                t.scatter(active[x_col], active[c], color=color, s=8, alpha=0.25)
            t.errorbar(center[x_col], center[c], yerr=spread[c], fmt='s-', color=color,
                       ms=4, capsize=2, label=f'2nd moment {k}')
        units = (self.analyzer.display_dict or {}).get('spatial_units')
        default = r'2nd moment ($2\sigma$)' + (rf' ($\mathrm{{{units}}}$)' if units else '')
        t.set_ylabel(self.display_dict.get('moment_ylabel', default))

        h1, l1 = a.get_legend_handles_labels()
        h2, l2 = t.get_legend_handles_labels()
        t.legend(h1 + h2, l1 + l2, fontsize='small', loc='upper left')
        return t

    @staticmethod
    def _stack(images):
        shapes = {np.asarray(im).shape for im in images if im is not None}
        if len(shapes) == 1 and all(im is not None for im in images):
            return np.stack([np.asarray(im) for im in images])
        out = np.empty(len(images), dtype=object)
        for k, im in enumerate(images):
            out[k] = None if im is None else np.asarray(im)
        return out
