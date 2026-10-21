from typing import Optional, Dict, Any, Callable, Union

import numpy as np

from PyGEECSPlotter.displayers.multi_scan_displayer import MultiScanDisplayer
from PyGEECSPlotter.utils.fitting import run_fit, draw_fit, fit_export


class MultiScanScalarVsParameter(MultiScanDisplayer):
    """
    One scalar vs the scan parameter (or any column) for two or more scans
    on one axis, one colour per scan.

    Each scan shows its shots (faint) and its per-bin summary (error bars),
    from its own ``active_data``, so each keeps its own filters. Optionally
    fits each scan separately (same options as ``ScalarVsParameter``).

    Parameters
    ----------
    y_col : str
        Column for the y-axis; must exist in every scan.
    x_col : str, optional
        Column for the x-axis. Defaults to each scan's ``scan_parameter``
        (the axis label is taken from the first scan).
    label_by, label_fmt : optional
        Legend entry per scan: ``'scan'`` (scan number, default),
        ``('mean' | 'max' | 'min' | 'median', column)`` for that scan's
        value of another column, a list of labels, or callable(scan); see
        ``scan_label``.
    bin_summary : {'none', 'mean', 'median', 'std_err', 'max', 'min'}, optional
        Per-bin summary drawn for each scan (default ``'mean'``, ± std).
    show_shots : bool, optional
        Also scatter every shot (default True).
    fit, fit_on, fit_func, p0, bounds, poly_degree, show_fit_params :
        Per-scan fit, as in ``ScalarVsParameter``. Fit parameter boxes are
        stacked down the left side, one per scan, in its colour.
    output_dir : str, optional
        Where files go; default the first scan's analysis directory.
    display_dict : dict, optional
        ``figsize``, ``cmap`` (scan colours; default the ``C0..C9`` cycle),
        ``shot_alpha`` (default 0.25), ``marker``, ``xlabel``, ``ylabel``,
        ``legend_loc``.

    Notes
    -----
    ``last_export`` has ``labels`` and, for each scan index ``k``,
    ``x_k`` / ``y_k`` (shots), ``x_bin_k`` / ``y_bin_k`` / ``y_bin_err_k``
    and, with a fit, ``fit_k_*``. ``last_fits`` keeps the fit result dicts.
    """

    def __init__(
        self,
        y_col: str,
        x_col: Optional[str] = None,
        label_by='scan',
        label_fmt: str = '{:.3g}',
        bin_summary: str = 'mean',
        show_shots: bool = True,
        fit: Optional[Union[str, Callable]] = None,
        fit_on: str = 'shots',
        fit_func: Optional[Callable] = None,
        p0=None,
        bounds=None,
        poly_degree: int = 2,
        show_fit_params: bool = True,
        display_dict: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        output_subdir: Optional[str] = None,
        timestamp_files: bool = True,
    ):
        super().__init__(name=f'{y_col}_vs_param', label_by=label_by, label_fmt=label_fmt,
                         display_dict=display_dict, output_dir=output_dir,
                         output_subdir=output_subdir, timestamp_files=timestamp_files)
        if fit_on not in ('shots', 'bins'):
            raise ValueError(f"fit_on must be 'shots' or 'bins', got {fit_on!r}.")
        self.y_col = y_col
        self.x_col = x_col
        self.bin_summary = bin_summary
        self.show_shots = show_shots
        self.fit = fit
        self.fit_on = fit_on
        self.fit_func = fit_func
        self.p0 = p0
        self.bounds = bounds
        self.poly_degree = poly_degree
        self.show_fit_params = show_fit_params
        self.last_fits = []

    def display(self, scans, fig=None, ax=None):
        scans = self._check_scans(scans)
        fig, ax = self._new_fig(fig, ax, figsize=(7, 5))
        labels = self._labels(scans)
        colors = self._colors(len(scans))
        dd = self.display_dict

        self.last_export = {'labels': np.asarray(labels, dtype=object)}
        self.last_fits = []
        for k, (scan, label, color) in enumerate(zip(scans, labels, colors)):
            df = scan.active_data
            x_col = self.x_col or scan.scan_parameter
            for col in (x_col, self.y_col):
                if col not in df.columns:
                    raise KeyError(f"Column '{col}' not in active_data of Scan {int(scan.scan):03d}.")
            x, y = df[x_col].to_numpy(), df[self.y_col].to_numpy()
            self.last_export[f'x_{k}'] = x
            self.last_export[f'y_{k}'] = y

            # the legend entry goes on the most prominent mark drawn for this scan
            show_bins = self.bin_summary != 'none'
            if self.show_shots:
                ax.scatter(x, y, color=color, marker=dd.get('marker', 'o'), s=12, lw=0,
                           alpha=dd.get('shot_alpha', 0.25) if show_bins else 0.6,
                           label=None if show_bins else label)

            center = spread = None
            if show_bins:
                center, spread = scan.compute_bin_summary(mode=self.bin_summary)
                ax.errorbar(center[x_col], center[self.y_col], yerr=spread[self.y_col],
                            fmt=dd.get('marker', 'o') + '-', color=color, capsize=3,
                            ms=5, lw=1, label=label)
                self.last_export[f'x_bin_{k}'] = center[x_col].to_numpy()
                self.last_export[f'y_bin_{k}'] = center[self.y_col].to_numpy()
                self.last_export[f'y_bin_err_{k}'] = spread[self.y_col].to_numpy()

            if self.fit is not None:
                result = self._fit(scan, x, y, x_col, center, spread)
                self.last_fits.append(result)
                draw_fit(ax, result, color=color, label=f'{label} fit',
                         show_params=self.show_fit_params,
                         text_loc=(0.02, 0.98 - 0.2 * k))
                self.last_export.update(fit_export(result, prefix=f'fit_{k}_'))

        x_label = self.x_col or scans[0].scan_parameter
        ax.set_xlabel(dd.get('xlabel', x_label))
        ax.set_ylabel(dd.get('ylabel', self.y_col))
        ax.legend(fontsize='small', loc=dd.get('legend_loc', 'best'))
        ax.set_title(self._title(scans))
        return fig, ax

    def _fit(self, scan, x, y, x_col, center, spread):
        sigma = None
        if self.fit_on == 'bins':
            if center is None:
                center, spread = scan.compute_bin_summary(mode='mean')
            x = center[x_col].to_numpy()
            y = center[self.y_col].to_numpy()
            sigma = spread[self.y_col].to_numpy()
            if not np.any(np.nan_to_num(sigma) > 0):
                sigma = None
        return run_fit(x, y, self.fit, fit_func=self.fit_func, p0=self.p0,
                       bounds=self.bounds, poly_degree=self.poly_degree, sigma=sigma)
