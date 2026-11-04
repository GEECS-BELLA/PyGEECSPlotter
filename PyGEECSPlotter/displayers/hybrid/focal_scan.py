import re
from contextlib import contextmanager
from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer
from PyGEECSPlotter.displayers.images.peak_width_images import PeakWidthImages
from PyGEECSPlotter.displayers.scalars.scalar_vs_parameter import ScalarVsParameter
from PyGEECSPlotter.displayers.scalars.scan_stats_table import draw_table
from PyGEECSPlotter.utils.fitting import fit_export
import PyGEECSPlotter.utils.analysis_1d_utils as a1d
from PyGEECSPlotter.displayers import _focus


class FocalScanDisplayer(ScanDisplayer):
    """
    One-figure overview of a laser focal scan (scan parameter = the stage
    moving through focus).

    Top row: ``max_counts_col`` vs the scan parameter with a ``fit_model``
    fit, a dashed line and the title at the fitted peak position; then the
    max-counts shot from the peak bin of ``max_counts_col``, and from the
    bins ``width_multiple`` half widths below and above it (see
    ``PeakWidthImages``; the peak bin is the largest bin mean, not the fit).
    Middle row: max counts vs the scan parameter (left axis) with the 2nd
    moments x, y (left panel) and x45, y45 (right panel) on a right axis
    shared between the two panels. Image labels give the stage position.
    With ``detailed=True``, a third row for the peak-bin shot: its image on
    a log colour scale, the encircled energy vs radius (with a Gaussian of
    the same w0), and a table of FWHM, w0, 2nd moments, encircled-energy
    radii, peak fluence / intensity, a0 and Strehl ratio
    (``analysis_2d_utils.focal_spot_stats``).

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
    fit_model : str, optional
        ``utils.fitting`` preset fitted to max counts vs the scan
        parameter (default ``'super_gaussian'``); its ``center`` is the
        reported peak.
    fit_on : {'shots', 'bins'}, optional
        Fit every shot (default) or the per-bin means.
    detailed : bool, optional
        Add the peak-shot row (default False).
    energy_col : str, optional
        Column with the pulse energy in J (default ``'Amplifier 3 [J]'``),
        read from the peak shot's row.
    transmission_col : str, optional
        Column with the transmission from ``energy_col`` to the focus
        (default ``'1BL compressor transmission'``), read from the same row.
        Scans without it use 0.586, with a warning.
    energy_scale : float, optional
        Use this transmission instead of ``transmission_col``. Default None.
    norm_pipeline : list, optional
        Pipeline (as in ``ImageAnalyzer``) re-run on the peak shot to get
        the image whose total counts hold the pulse energy, and on which the
        encircled energy is measured. Default: the analyzer's pipeline
        without its ``'threshold'`` steps, so halo energy a threshold
        removes isn't credited to the peak. The peak counts and widths come
        from the analyzer's own image.
    pulse_fwhm_fs, wavelength_um : float, optional
        Pulse duration (default 40 fs) and wavelength (default 0.8 um) for
        the intensity and a0. The analyzer's dx / dy must be in um.
    display_dict : dict, optional
        ``figsize`` (default (16, 8), or (16, 12) when detailed); ``vmin``
        / ``vmax`` for the images (default: shared range of the three
        images); ``show_shots`` (default True) draws every shot faintly
        behind the bin means; ``moment_ylabel``; ``axlims`` (window around
        the beam, as in ``ImageAnalyzer.display_data``) for the images;
        ``show_fit_params`` (default False) adds the fit parameter box;
        ``log_vmin`` (default 1 count) and ``log_tick_dx`` (default
        automatic ticks) for the log image; ``ee_rmax``
        (default 10 w0) for the encircled-energy x range.
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
        fit_model: str = 'super_gaussian',
        fit_on: str = 'shots',
        detailed: bool = False,
        energy_col: str = 'Amplifier 3 [J]',
        transmission_col: str = '1BL compressor transmission',
        energy_scale: Optional[float] = None,
        norm_pipeline: Optional[list] = None,
        pulse_fwhm_fs: float = 40.0,
        wavelength_um: float = 0.8,
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
        self.fit_model = fit_model
        self.fit_plot = ScalarVsParameter(
            max_counts_col, bin_summary='mean', fit=fit_model, fit_on=fit_on,
            show_fit_params=self.display_dict.get('show_fit_params', False),
            show_shots=self.display_dict.get('show_shots', True),
            display_dict={'color': '0.6', 'alpha': 0.5, 'marker': '.', 'bin_color': 'k',
                          'fit_color': 'C3'})
        self.detailed = detailed
        self.energy_col = energy_col
        self.transmission_col = transmission_col
        self.energy_scale = energy_scale
        self.norm_pipeline = norm_pipeline
        self.pulse_fwhm_fs = pulse_fwhm_fs
        self.wavelength_um = wavelength_um

    def display(self, scan, fig=None, ax=None):
        active = scan.active_data
        x_col = scan.scan_parameter
        for c in [x_col, self.max_counts_col] + list(self.moment_cols.values()):
            if c not in active.columns:
                raise KeyError(f"Column '{c}' not in scan.active_data.")

        # PeakWidthImages labels are '<tag>\n<position>'; keep just the position
        labelled = self.images._collect_panels(scan)
        panels = [(label.split('\n')[-1], d, rd) for label, d, rd in labelled]
        peak = self.images.last_peak

        default_size = (16, 12) if self.detailed else (16, 8)
        fig = plt.figure(constrained_layout=True,
                         figsize=self.display_dict.get('figsize', default_size))
        # separate subfigures so the lower rows' inner axis labels don't
        # squeeze the middle image
        rows = fig.subfigures(3 if self.detailed else 2, 1)
        top, bottom = rows[0], rows[1]
        ax_fit, *img_axes = top.subplots(1, 4, width_ratios=[1.4, 1, 1, 1])
        axL, axR = bottom.subplots(1, 2)

        fit = self._draw_fit_panel(scan, top, ax_fit, x_col)
        self._draw_images(fig, img_axes, panels)
        center, spread = scan.compute_bin_summary(mode='mean')
        twins = [self._draw_curves(a, active, center, spread, x_col, keys)
                 for a, keys in ((axL, ('x', 'y')), (axR, ('x45', 'y45')))]
        lo = min(t.get_ylim()[0] for t in twins)
        hi = max(t.get_ylim()[1] for t in twins)
        for t in twins:
            t.set_ylim(lo, hi)
        if np.isfinite(fit['x_peak_fit']):
            for a in (axL, axR):
                a.axvline(fit['x_peak_fit'], color='C3', ls='--', lw=1)

        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        fig.suptitle(scan.scan_data_title(f'{diag} focal scan') + f"\n{fit['text']}")

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
        self.last_export.update(fit['export'])
        axes = [ax_fit, img_axes, axL, axR]

        if self.detailed:
            tags = [label.split('\n')[0] for label, _, _ in labelled]
            _, d_pk, rd_pk = labelled[tags.index('peak')]
            in_bin = np.where(active['temp Bin number'].to_numpy() == peak['bin_peak'])[0]
            pos = self.images._pick_in_bin(active, in_bin)
            detail_axes, export = self._draw_detailed(rows[2], scan, pos, d_pk, rd_pk, x_col)
            self.last_export.update(export)
            axes.append(detail_axes)
        return fig, axes

    def _draw_fit_panel(self, scan, fig, a, x_col):
        """Max counts vs the scan parameter with the fit, a line at the
        fitted centre and the peak in the title."""
        self.fit_plot.display(scan, fig=fig, ax=a)
        res = self.fit_plot.last_fit
        p = dict(zip(res['names'], res['params']))
        e = dict(zip(res['names'], res['errors']))
        x_pk, x_err, fwhm = p.get('center', np.nan), e.get('center', np.nan), np.nan
        if res['success'] and np.isfinite(x_pk):
            a.axvline(x_pk, color='C3', ls='--', lw=1)
            y = res['y_fit'] - p.get('offset', 0.0)
            fwhm = a1d.interpolated_width(y)*(res['x_fit'][1] - res['x_fit'][0])
            text = f'peak {x_col} = {x_pk:.4g} ± {x_err:.2g}'
            if np.isfinite(fwhm):
                text += f', FWHM = {fwhm:.3g}'
        else:
            text = f'{self.fit_model} fit failed'
        a.set_title(text.replace(f' {x_col}', '').replace(', ', '\n'),
                    fontweight='bold', fontsize='medium')
        a.set_xlabel(x_col)
        a.set_ylabel(self.max_counts_col)
        export = fit_export(res)
        export.update({'x_peak_fit': x_pk, 'x_peak_fit_err': x_err, 'fwhm_fit': fwhm})
        return {'x_peak_fit': x_pk, 'text': text, 'export': export}


    def _draw_images(self, fig, img_axes, panels):
        data = [d for _, d, _ in panels if d is not None]
        dd = self.analyzer.display_dict
        with self._analyzer_display(
                vmin=self.display_dict.get('vmin', dd.get('vmin', min(np.nanmin(d) for d in data))),
                vmax=self.display_dict.get('vmax', dd.get('vmax', max(np.nanmax(d) for d in data))),
                show_lineouts=self.show_lineouts,
                axlims=self.display_dict.get('axlims', dd.get('axlims'))):
            for a, (label, d, rd) in zip(img_axes, panels):
                if d is None:
                    a.set_visible(False)
                    continue
                self.images._render_panel(fig, a, d, rd, label)
        for a in img_axes[len(panels):]:
            a.set_visible(False)

    @contextmanager
    def _analyzer_display(self, **overrides):
        """Temporarily set keys of the analyzer's display_dict."""
        dd = self.analyzer.display_dict
        saved = {k: dd[k] for k in overrides if k in dd}
        dd.update(overrides)
        try:
            yield dd
        finally:
            for k in overrides:
                dd.pop(k, None)
            dd.update(saved)

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

    def _draw_detailed(self, fig, scan, pos, data, return_dict, x_col):
        """Log image, encircled energy and stats table for the peak shot
        (row ``pos`` of active_data)."""
        row = scan.active_data.iloc[pos]
        raw, transmission = _focus.shot_energy(row, self.energy_col, self.transmission_col,
                                               self.energy_scale)
        norm_an = _focus.norm_analyzer(self.analyzer, self.norm_pipeline)
        (_, norm_data, _, _), = list(scan._iter_shots(norm_an, bg=self.bg, show_progress=False,
                                                      rows=scan.active_data.iloc[[pos]]))
        norm_data = np.asarray(norm_data, dtype=float)
        stats = _focus.shot_focus_stats(self.analyzer, data, return_dict, norm_data,
                                        raw*transmission, self.pulse_fwhm_fs, self.wavelength_um)
        shot = int(row['Shotnumber']) if 'Shotnumber' in row.index else None

        ax_img, ax_ee, ax_tab = fig.subplots(1, 3, width_ratios=[1, 1.2, 1])
        self._draw_log_image(fig, ax_img, data, return_dict,
                             f'shot {shot}, {self.images.label_fmt.format(row[x_col])} (log)')
        r, frac = _focus.encircled_energy(self.analyzer, norm_data, np.shape(data), stats)
        ax_ee.plot(r, frac, color='C0', lw=1.5, label='data')
        _focus.finish_ee_axes(ax_ee, self.analyzer, r.max(), stats['w0'],
                              self.display_dict.get('ee_rmax'))
        ax_ee.set_title(f"energy within $w_0$ = {stats['ee w0']:.0%} "
                        f"({stats['ee w0 / gaussian']:.2f} of Gaussian)")
        self._draw_stats_table(ax_tab, shot, row[x_col], raw, transmission, stats)

        export = {'detail_shot': shot, 'detail_image': np.asarray(data),
                  'detail_norm_image': norm_data, 'detail_energy_raw': raw,
                  'detail_transmission': transmission, 'ee_r': r, 'ee_frac': frac}
        export.update({'detail_' + re.sub(r'\W+', '_', k).strip('_'): v for k, v in stats.items()})
        return [ax_img, ax_ee, ax_tab], export

    def _draw_log_image(self, fig, a, data, return_dict, title):
        dd = self.analyzer.display_dict
        cmap = dd.get('cmap', 'RdBu')
        cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        vmax = float(np.nanmax(data))
        vmin = self.display_dict.get('log_vmin', 1.0 if vmax > 1 else 1e-3*vmax)
        # zero / negative pixels as the lowest colour rather than blank
        with self._analyzer_display(norm=LogNorm(vmin=vmin, vmax=vmax), vmin=None, vmax=None,
                                    cmap=cmap.with_extremes(bad=cmap(0), under=cmap(0)),
                                    cbar_off=False, cbar_label='Counts', show_lineouts=False,
                                    axlims=self.display_dict.get('axlims', dd.get('axlims')),
                                    tick_dx=self.display_dict.get('log_tick_dx')):
            self.analyzer.display_data(data, return_dict=return_dict, fig=fig, ax=a, title=title)

    def _draw_stats_table(self, a, shot, position, raw_energy, transmission, stats):
        def pair(u, v, fmt='{:.3g}'):
            return f'{fmt.format(u)} / {fmt.format(v)}'

        table = [
            ('shot', f'{shot} at {self.images.label_fmt.format(position)}'),
            ('energy on target (J)', f"{stats['energy (J)']:.3g}"),
            (f'{self.energy_col} × transmission', f'{raw_energy:.3g} × {transmission:.3g}'),
            ('pulse FWHM (fs)', f'{self.pulse_fwhm_fs:g}'),
            ('FWHM x / y (µm)', pair(stats['fwhm x'], stats['fwhm y'])),
            ('w0 from FWHM x / y (µm)', pair(stats['w0 x'], stats['w0 y'])),
            ('FW 1/e² x / y (µm)', pair(stats['fw 1/e2 x'], stats['fw 1/e2 y'])),
            ('2nd moment x / y (µm)', pair(stats['2nd moment x'], stats['2nd moment y'])),
            ('2nd moment x45 / y45 (µm)', pair(stats['2nd moment x45'], stats['2nd moment y45'])),
            ('ee radius 50% / 86.5% (µm)', pair(stats['ee radius 50%'], stats['ee radius 86.5%'])),
            ('energy within w0', f"{stats['ee w0']:.1%} ({stats['ee w0 / gaussian']:.2f} of Gaussian)"),
            ('total counts (no threshold)', f"{stats['total counts']:.3g}"),
            ('energy per count (J)', f"{stats['energy per count (J)']:.3g}"),
            ('peak fluence (J/cm²)', f"{stats['peak fluence (J/cm^2)']:.3g}"),
            ('peak intensity (W/cm²)', f"{stats['peak intensity (W/cm^2)']:.3g}"),
            ('a0', f"{stats['a0']:.3g}"),
            ('Strehl (vs Gaussian, same w0)', f"{stats['strehl']:.3g}"),
        ]
        draw_table(a, [[v] for _, v in table], [k for k, _ in table], ['peak shot'],
                   display_dict={'fontsize': 9, 'wrap': 24}, fit_axes=True)

    @staticmethod
    def _stack(images):
        shapes = {np.asarray(im).shape for im in images if im is not None}
        if len(shapes) == 1 and all(im is not None for im in images):
            return np.stack([np.asarray(im) for im in images])
        out = np.empty(len(images), dtype=object)
        for k, im in enumerate(images):
            out[k] = None if im is None else np.asarray(im)
        return out
