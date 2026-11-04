import re
import warnings
from typing import Optional, Dict, Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer
from PyGEECSPlotter.displayers.scalars.scan_stats_table import draw_table, format_value
from PyGEECSPlotter.displayers import _focus

# (table row label, focal_spot_stats key)
FOCUS_ROWS = [
    ('energy on target (J)', 'energy (J)'),
    ('FWHM x (µm)', 'fwhm x'),
    ('FWHM y (µm)', 'fwhm y'),
    ('w0 from FWHM (µm)', 'w0'),
    ('FW 1/e² x (µm)', 'fw 1/e2 x'),
    ('FW 1/e² y (µm)', 'fw 1/e2 y'),
    ('2nd moment x (µm)', '2nd moment x'),
    ('2nd moment y (µm)', '2nd moment y'),
    ('2nd moment x45 (µm)', '2nd moment x45'),
    ('2nd moment y45 (µm)', '2nd moment y45'),
    ('ee radius 50% (µm)', 'ee radius 50%'),
    ('ee radius 86.5% (µm)', 'ee radius 86.5%'),
    ('energy within w0', 'ee w0'),
    ('peak counts', 'peak counts'),
    ('total counts (no threshold)', 'total counts'),
    ('peak fluence (J/cm²)', 'peak fluence (J/cm^2)'),
    ('peak intensity (W/cm²)', 'peak intensity (W/cm^2)'),
    ('a0', 'a0'),
    ('Strehl (vs Gaussian, same w0)', 'strehl'),
]


class FocusStatsDisplayer(ScanDisplayer):
    """
    Shot-to-shot statistics of a laser focus imaged over N shots (a no-scan
    with the stage at focus).

    Every active shot goes through ``analyzer`` (peak counts, widths, 2nd
    moments) and through ``norm_pipeline`` (the image whose counts hold the
    pulse energy, and the encircled energy), as in ``FocalScanDisplayer``'s
    detailed view; ``analysis_2d_utils.focal_spot_stats`` gives each shot's
    FWHM, w0, 2nd moments, encircled-energy radii, peak fluence /
    intensity, a0 and Strehl ratio with that shot's own energy.

    Left: the mean encircled energy vs radius with +-1 std error bars over
    the shots, and a Gaussian of the mean w0. Right: a stats table (as
    ``ScanStatsTable``) of those quantities, plus any sfile ``columns``.

    Parameters
    ----------
    analyzer : ImageAnalyzer
        dx / dy must be in um.
    bg : optional
        Background forwarded to the per-shot pipeline.
    columns : iterable of str, optional
        Extra sfile columns to add to the table.
    stats : iterable of str, optional
        ``describe()`` statistics to show (default count, mean, std, min,
        50%, max).
    energy_col, transmission_col, energy_scale, norm_pipeline,
    pulse_fwhm_fs, wavelength_um :
        As in ``FocalScanDisplayer``: energy on target = ``energy_col`` x
        ``transmission_col`` (0.586 if the sfile lacks it; ``energy_scale``
        overrides), per shot.
    max_workers : int, optional
        Threads for the per-shot analysis.
    fmt : str, optional
        Number format for the table cells (default ``'{:.4g}'``).
    display_dict : dict, optional
        ``figsize`` (default (15, 7)); ``ee_rmax`` (default 10 w0) for the
        encircled-energy x range; ``n_errorbars`` (default 25) error bars
        along the curve; ``show_shots`` (default False) draws every shot's
        curve faintly; ``fontsize`` for the table (default 9).

    ``last_shots`` holds the per-shot DataFrame (index: shot number) and
    ``last_table`` the summary; ``last_export`` holds the table values,
    the per-shot values and the encircled-energy grid, mean, std and
    per-shot curves.
    """

    def __init__(
        self,
        analyzer,
        bg=None,
        columns: Optional[Iterable[str]] = None,
        stats: Iterable[str] = ('count', 'mean', 'std', 'min', '50%', 'max'),
        energy_col: str = 'Amplifier 3 [J]',
        transmission_col: str = '1BL compressor transmission',
        energy_scale: Optional[float] = None,
        norm_pipeline: Optional[list] = None,
        pulse_fwhm_fs: float = 40.0,
        wavelength_um: float = 0.8,
        max_workers: Optional[int] = None,
        fmt: str = '{:.4g}',
        display_dict: Optional[Dict[str, Any]] = None,
        output_subdir: Optional[str] = None,
        timestamp_files: bool = True,
    ):
        diag = analyzer.output_diagnostic or analyzer.diagnostic
        super().__init__(name=f"{diag}_focus_stats", display_dict=display_dict,
                         output_subdir=output_subdir, timestamp_files=timestamp_files)
        self.analyzer = analyzer
        self.bg = bg
        self.columns = list(columns or [])
        self.stats = list(stats)
        self.energy_col = energy_col
        self.transmission_col = transmission_col
        self.energy_scale = energy_scale
        self.norm_pipeline = norm_pipeline
        self.pulse_fwhm_fs = pulse_fwhm_fs
        self.wavelength_um = wavelength_um
        self.max_workers = max_workers
        self.fmt = fmt
        self.last_shots = None
        self.last_table = None

    def display(self, scan, fig=None, ax=None):
        shots, curves = self._analyze_shots(scan)
        if not len(shots):
            raise RuntimeError("No shots with an image to analyse.")
        self.last_shots = shots
        r, curves = self._common_grid(curves)
        mean, std = curves.mean(axis=0), curves.std(axis=0, ddof=1 if len(curves) > 1 else 0)

        if fig is None or ax is None:
            fig, (ax_ee, ax_tab) = plt.subplots(1, 2, constrained_layout=True,
                                                width_ratios=[1, 1.3],
                                                figsize=self.display_dict.get('figsize', (15, 7)))
        else:
            ax_ee, ax_tab = ax
        self._draw_ee(ax_ee, r, curves, mean, std, shots)
        table = self._draw_table(ax_tab, scan, shots)

        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        summary = ', '.join(f'{name} = {self._mean_std(shots[k])}' for name, k in (
            ('a0', 'a0'), ('peak I', 'peak intensity (W/cm^2)'), ('energy in $w_0$', 'ee w0')))
        fig.suptitle(scan.scan_data_title(f'{diag} focus, {len(shots)} shots') + f'\n{summary}')

        self.last_export = {
            'values': table.to_numpy(dtype=float),
            'row_labels': np.asarray(table.index, dtype=object),
            'col_labels': np.asarray(table.columns, dtype=object),
            'shots': shots.index.to_numpy(),
            'ee_r': r, 'ee_mean': mean, 'ee_std': std, 'ee_curves': curves,
        }
        for k in shots.columns:
            self.last_export['shot_' + re.sub(r'\W+', '_', k).strip('_')] = shots[k].to_numpy()
        return fig, [ax_ee, ax_tab]

    def _analyze_shots(self, scan):
        """Per-shot stats DataFrame and [(r, fraction), ...] encircled-energy
        curves, over the active shots with an image."""
        norm_an = _focus.norm_analyzer(self.analyzer, self.norm_pipeline)

        def process(row):
            context, data, results, aux = scan._process_row(self.analyzer, self.bg, row)
            _, norm_data, _, _ = scan._process_row(norm_an, self.bg, row)
            return context, (data, norm_data), results, aux

        records, index, curves = [], [], []
        for context, (data, norm_data), results, _ in scan._iter_shots_parallel(
                self.analyzer, bg=self.bg, max_workers=self.max_workers, process=process):
            if data is None or norm_data is None:
                continue
            raw, transmission = _focus.shot_energy(context, self.energy_col,
                                                   self.transmission_col, self.energy_scale)
            norm_data = np.asarray(norm_data, dtype=float)
            stats = _focus.shot_focus_stats(self.analyzer, data, results, norm_data,
                                            raw*transmission, self.pulse_fwhm_fs,
                                            self.wavelength_um)
            curves.append(_focus.encircled_energy(self.analyzer, norm_data, np.shape(data), stats))
            records.append(stats)
            index.append(context.get('Shotnumber', len(index)))
        shots = pd.DataFrame(records, index=pd.Index(index, name='Shotnumber'))
        return shots, curves

    @staticmethod
    def _common_grid(curves, n=400):
        """Curves resampled onto 0 .. the smallest image's largest radius."""
        r = np.linspace(0, min(rc.max() for rc, _ in curves), n)
        return r, np.vstack([np.interp(r, rc, fc) for rc, fc in curves])

    @staticmethod
    def _mean_std(values, fmt='{:.3g}'):
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if not len(v):
            return '—'
        return f'{fmt.format(v.mean())} ± {fmt.format(v.std(ddof=1) if len(v) > 1 else 0)}'

    def _draw_ee(self, a, r, curves, mean, std, shots):
        if self.display_dict.get('show_shots', False):
            for c in curves:
                a.plot(r, c, color='C0', lw=0.5, alpha=0.2)
        every = max(1, len(r)//self.display_dict.get('n_errorbars', 25))
        a.errorbar(r, mean, yerr=std, color='C0', lw=1.5, ecolor='C0', elinewidth=1,
                   capsize=2, errorevery=every, label=f'mean ± std ({len(curves)} shots)')
        _focus.finish_ee_axes(a, self.analyzer, r.max(), float(np.nanmean(shots['w0'])),
                              self.display_dict.get('ee_rmax'))
        a.set_title(f"energy within $w_0$ = {self._mean_std(100*shots['ee w0'], '{:.1f}')} %")

    def _draw_table(self, a, scan, shots):
        rows = pd.DataFrame({label: shots[k].to_numpy(dtype=float) for label, k in FOCUS_ROWS},
                            index=shots.index)
        extra = [c for c in self.columns if c in scan.active_data.columns]
        if len(extra) < len(self.columns):
            warnings.warn(f"FocusStatsDisplayer: columns not in active_data: "
                          f"{[c for c in self.columns if c not in extra]}")
        if extra:
            sfile = scan.active_data.set_index('Shotnumber')[extra] \
                if 'Shotnumber' in scan.active_data.columns else scan.active_data[extra]
            rows = rows.join(sfile.apply(pd.to_numeric, errors='coerce'), how='left')
        desc = rows.describe()
        table = desc.loc[[s for s in self.stats if s in desc.index]].T
        self.last_table = table
        cells = [[format_value(v, self.fmt) for v in row] for row in table.to_numpy(dtype=float)]
        draw_table(a, cells, list(table.index), list(table.columns),
                   display_dict={'fontsize': self.display_dict.get('fontsize', 9), 'wrap': 24},
                   fit_axes=True)
        return table
