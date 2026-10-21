from typing import Optional, Dict, Any, Callable, Union

import numpy as np

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer
from PyGEECSPlotter.fitting import run_fit, draw_fit, fit_export


class ScalarVsParameter(ScanDisplayer):
    """
    Scatter / errorbar plot of one scalar column vs the scan parameter
    (or any other column), optionally with a fitted curve.

    Parameters
    ----------
    y_col : str
        Column to plot on the y-axis. Must exist in ``scan.active_data``.
    x_col : str, optional
        Column to plot on the x-axis. Defaults to ``scan.scan_parameter``.
        Any scalar column works, e.g. for a linear-regression correlation
        of two scalars.
    bin_summary : {'none', 'mean', 'median', 'std_err', 'max', 'min'}, optional
        If not ``'none'``, also overlays a per-bin summary using
        ``scan.compute_bin_summary``.
    fit : str or callable, optional
        Fit to overlay (see ``PyGEECSPlotter.fitting``):
          * a preset: ``'gaussian'``, ``'gaussian_no_offset'``,
            ``'super_gaussian'``, ``'super_gaussian_no_offset'``,
            ``'lorentzian'``, ``'lorentzian_slope'``, ``'boltzmann'``,
            ``'extreme'``
          * ``'poly'`` — polynomial of ``poly_degree``
          * ``'linear'`` — linear regression with a 95% confidence band on
            the line, and r, R², p-value, N in the text box
          * ``'custom'`` (with ``fit_func``) or a callable
            ``f(x, p1, p2, ...)``
    fit_on : {'shots', 'bins'}, optional
        Fit every active shot (default), or the per-bin summary points
        (``bin_summary``, or ``'mean'`` if that is ``'none'``), weighted by
        their spread. ``'linear'`` is unweighted either way.
    fit_func : callable, optional
        Model for ``fit='custom'``; parameter names come from its signature.
    p0, bounds : optional
        Initial guess and ``(lower, upper)`` bounds for preset / custom fits.
        Presets guess ``p0`` from the data when omitted.
    poly_degree : int, optional
        Degree for ``fit='poly'`` (default 2).
    show_fit_params : bool, optional
        Write the fitted parameters ± 1 sigma in a text box (default True).
    show_shots : bool, optional
        Scatter every active shot (default True).
    display_dict : dict, optional
        Style overrides: ``color``, ``alpha``, ``marker``, ``bin_color``,
        (per-bin error bars, default ``'C0'`` blue), ``fit_color`` (default
        ``'k'``), ``fit_text_loc`` (axes fraction, default top left),
        ``legend`` (default False), ``legend_loc``, ``figsize``,
        ``ylabel``, ``xlabel``.

    Notes
    -----
    With a fit, ``last_export`` also holds ``fit_x``, ``fit_y``,
    ``fit_params``, ``fit_errors``, ``fit_param_names``, ``fit_r2`` and
    ``fit_model``; the full result dict is kept as ``last_fit``.
    """

    def __init__(
        self,
        y_col: str,
        x_col: Optional[str] = None,
        bin_summary: str = 'none',
        display_dict: Optional[Dict[str, Any]] = None,
        output_subdir: Optional[str] = None,
        timestamp_files: bool = True,
        fit: Optional[Union[str, Callable]] = None,
        fit_on: str = 'shots',
        fit_func: Optional[Callable] = None,
        p0=None,
        bounds=None,
        poly_degree: int = 2,
        show_fit_params: bool = True,
        show_shots: bool = True,
    ):
        super().__init__(name=f"{y_col}_vs_param", display_dict=display_dict,
                          output_subdir=output_subdir, timestamp_files=timestamp_files)
        if fit_on not in ('shots', 'bins'):
            raise ValueError(f"fit_on must be 'shots' or 'bins', got {fit_on!r}.")
        self.y_col = y_col
        self.x_col = x_col
        self.bin_summary = bin_summary
        self.fit = fit
        self.fit_on = fit_on
        self.fit_func = fit_func
        self.p0 = p0
        self.bounds = bounds
        self.poly_degree = poly_degree
        self.show_fit_params = show_fit_params
        self.show_shots = show_shots
        self.last_fit = None

    def display(self, scan, fig=None, ax=None):
        fig, ax = self._new_fig(fig, ax)
        df = scan.active_data
        x_col = self.x_col or scan.scan_parameter

        if self.y_col not in df.columns:
            raise KeyError(f"Column '{self.y_col}' not in scan.active_data.")
        if x_col not in df.columns:
            raise KeyError(f"Column '{x_col}' not in scan.active_data.")

        if self.show_shots:
            ax.scatter(
                df[x_col],
                df[self.y_col],
                color=self.display_dict.get('color', 'k'),
                marker=self.display_dict.get('marker', 'o'),
                alpha=self.display_dict.get('alpha', 0.5),
                label='shots',
            )

        self.last_export = {
            'x': df[x_col].to_numpy(),
            'y': df[self.y_col].to_numpy(),
            'x_col': x_col,
            'y_col': self.y_col,
        }

        center = spread = None
        if self.bin_summary != 'none':
            center, spread = scan.compute_bin_summary(mode=self.bin_summary)
            if x_col in center.columns and self.y_col in center.columns:
                ax.errorbar(
                    center[x_col],
                    center[self.y_col],
                    yerr=spread[self.y_col],
                    fmt='o',
                    color=self.display_dict.get('bin_color', 'C0'),
                    capsize=3,
                    label=f'per-bin {self.bin_summary}',
                )
                self.last_export.update({
                    'x_bin': center[x_col].to_numpy(),
                    'y_bin': center[self.y_col].to_numpy(),
                    'y_bin_err': spread[self.y_col].to_numpy(),
                })

        if self.fit is not None:
            self.last_fit = self._fit(scan, df, x_col, center, spread)
            draw_fit(ax, self.last_fit, color=self.display_dict.get('fit_color', 'k'),
                     show_params=self.show_fit_params,
                     text_loc=self.display_dict.get('fit_text_loc', (0.02, 0.98)))
            self.last_export.update(fit_export(self.last_fit))

        if self.display_dict.get('legend', False):
            ax.legend(fontsize='small', loc=self.display_dict.get('legend_loc', 'best'))
        ax.set_xlabel(self.display_dict.get('xlabel', x_col))
        ax.set_ylabel(self.display_dict.get('ylabel', self.y_col))
        ax.set_title(scan.scan_data_title())
        return fig, ax

    def _fit(self, scan, df, x_col, center, spread):
        """Fit result for the shots or the per-bin summary points."""
        sigma = None
        if self.fit_on == 'shots':
            x, y = df[x_col].to_numpy(), df[self.y_col].to_numpy()
        else:
            if center is None:
                center, spread = scan.compute_bin_summary(mode='mean')
            x = center[x_col].to_numpy()
            y = center[self.y_col].to_numpy()
            sigma = spread[self.y_col].to_numpy()
            # 'max' / 'min' summaries have zero spread: fit unweighted
            if not np.any(np.nan_to_num(sigma) > 0):
                sigma = None
        return run_fit(x, y, self.fit, fit_func=self.fit_func, p0=self.p0,
                       bounds=self.bounds, poly_degree=self.poly_degree, sigma=sigma)
