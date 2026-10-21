"""
1-D curve fits for scan-level scalar plots (``ScalarVsParameter``,
``MultiScanScalarVsParameter``): the GEECSPlotter preset models, polynomial
fits and linear regression.

Every fit returns the same flat dict, so displayers can draw and export any
of them the same way:

    {'model', 'success', 'message', 'names', 'params', 'errors', 'r2', 'n',
     'x_fit', 'y_fit', 'y_lo', 'y_hi'}

``errors`` are 1-sigma (square root of the covariance diagonal). ``y_lo`` /
``y_hi`` are a confidence band on the fitted mean (``linear_regression``
only; None otherwise). A fit that fails returns ``success=False`` with NaN
parameters instead of raising, so one bad scan doesn't stop a batch.
"""

import inspect
import warnings

import numpy as np
from scipy import optimize, stats

from PyGEECSPlotter.utils import super_gaussian


# ----------------------------------------------------------------------
# Preset models
# ----------------------------------------------------------------------
def gaussian(x, amplitude, center, sigma, offset):
    return super_gaussian(x, amplitude, center, sigma, offset, N=2)


def gaussian_no_offset(x, amplitude, center, sigma):
    return super_gaussian(x, amplitude, center, sigma, 0.0, N=2)


def super_gaussian_fit(x, amplitude, center, sigma, offset, N):
    return super_gaussian(x, amplitude, center, sigma, offset, N=N)


def super_gaussian_no_offset(x, amplitude, center, sigma, N):
    return super_gaussian(x, amplitude, center, sigma, 0.0, N=N)


def lorentzian(x, amplitude, center, gamma, offset):
    """Peak height ``amplitude`` above ``offset``; ``gamma`` is the HWHM."""
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2) + offset


def lorentzian_slope(x, amplitude, center, gamma, offset, slope):
    return lorentzian(x, amplitude, center, gamma, offset) + slope * x


def boltzmann(x, A1, A2, x0, dx):
    """Sigmoid from ``A1`` (x << x0) to ``A2`` (x >> x0), width ``dx``."""
    return A2 + (A1 - A2) / (1 + np.exp((x - x0) / dx))


def extreme(x, y0, A, xc, w):
    """Extreme-value (Gumbel) peak, as in Origin / LabVIEW: peak height
    ``A`` above ``y0`` at ``xc``, width ``w``."""
    z = (x - xc) / w
    return y0 + A * np.exp(-np.exp(-z) - z + 1)


def _peak_guess(x, y, dip=False):
    """(amplitude, center, width, offset) of the main peak, from the data;
    with ``dip=True``, of the main dip (negative amplitude, offset at the
    maximum)."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    offset = np.nanmax(y) if dip else np.nanmin(y)
    k = int(np.nanargmin(y) if dip else np.nanargmax(y))
    amplitude = y[k] - offset
    above = x[(y - offset) / amplitude >= 0.5] if amplitude else x[:0]
    width = (above.max() - above.min()) / 2.355 if len(above) > 1 else 0
    if not width > 0:
        width = (x.max() - x.min()) / 10 or 1.0
    return amplitude, x[k], width, offset


def _p0_gaussian(x, y, dip=False):
    a, c, w, o = _peak_guess(x, y, dip)
    return [a, c, w, o]


def _p0_gaussian_no_offset(x, y, dip=False):
    a, c, w, o = _peak_guess(x, y, dip)
    return [a + o, c, w]


def _p0_super_gaussian(x, y, dip=False):
    return _p0_gaussian(x, y, dip) + [2.0]


def _p0_super_gaussian_no_offset(x, y, dip=False):
    return _p0_gaussian_no_offset(x, y, dip) + [2.0]


def _p0_lorentzian(x, y, dip=False):
    a, c, w, o = _peak_guess(x, y, dip)
    return [a, c, 1.1775 * w, o]        # HWHM = 2.355 sigma / 2


def _p0_lorentzian_slope(x, y, dip=False):
    return _p0_lorentzian(x, y, dip) + [0.0]


def _p0_boltzmann(x, y, dip=False):
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = max(1, len(x) // 5)
    span = (x.max() - x.min()) or 1.0
    return [np.nanmean(y[:n]), np.nanmean(y[-n:]), 0.5 * (x.min() + x.max()), span / 10]


def _p0_extreme(x, y, dip=False):
    a, c, w, o = _peak_guess(x, y, dip)
    return [o, a, c, w]


# name: (function, parameter names, initial-guess function)
FIT_MODELS = {
    'gaussian': (gaussian, ['amplitude', 'center', 'sigma', 'offset'], _p0_gaussian),
    'gaussian_no_offset': (gaussian_no_offset, ['amplitude', 'center', 'sigma'],
                           _p0_gaussian_no_offset),
    'super_gaussian': (super_gaussian_fit, ['amplitude', 'center', 'sigma', 'offset', 'N'],
                       _p0_super_gaussian),
    'super_gaussian_no_offset': (super_gaussian_no_offset, ['amplitude', 'center', 'sigma', 'N'],
                                 _p0_super_gaussian_no_offset),
    'lorentzian': (lorentzian, ['amplitude', 'center', 'gamma', 'offset'], _p0_lorentzian),
    'lorentzian_slope': (lorentzian_slope, ['amplitude', 'center', 'gamma', 'offset', 'slope'],
                         _p0_lorentzian_slope),
    'boltzmann': (boltzmann, ['A1', 'A2', 'x0', 'dx'], _p0_boltzmann),
    'extreme': (extreme, ['y0', 'A', 'xc', 'w'], _p0_extreme),
}
# presets fitted from both a peak and a dip starting guess
_PEAK_MODELS = {name for name in FIT_MODELS if name != 'boltzmann'}


# ----------------------------------------------------------------------
# Fits
# ----------------------------------------------------------------------
def _clean(x, y, sigma=None):
    """Finite (x, y[, sigma]) as float arrays; sigma <= 0 or NaN drops the point."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        ok &= np.isfinite(sigma) & (sigma > 0)
        # every point without a usable error: fit unweighted rather than drop them all
        if not ok.any():
            ok = np.isfinite(x) & np.isfinite(y)
            sigma = None
    return x[ok], y[ok], (sigma[ok] if sigma is not None else None)


def _r2(y, y_model):
    ss_res = np.sum((y - y_model)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def _x_fit(x, n_points):
    return np.linspace(np.min(x), np.max(x), n_points) if len(x) else np.array([])


def _result(model, names, params, errors, r2, n, x_fit, y_fit,
            success=True, message='', y_lo=None, y_hi=None, **extra):
    out = {'model': model, 'success': success, 'message': message,
           'names': list(names), 'params': np.asarray(params, dtype=float),
           'errors': np.asarray(errors, dtype=float), 'r2': r2, 'n': n,
           'x_fit': x_fit, 'y_fit': y_fit, 'y_lo': y_lo, 'y_hi': y_hi}
    out.update(extra)
    return out


def _failed(model, names, x, n_points, message):
    nan = np.full(len(names), np.nan)
    x_fit = _x_fit(x, n_points)
    return _result(model, names, nan, nan, np.nan, len(x), x_fit,
                   np.full_like(x_fit, np.nan), success=False, message=message)


def fit_curve(x, y, model, p0=None, bounds=None, sigma=None, n_points=200, maxfev=20000):
    """
    Least-squares fit of ``model`` to (x, y) with ``scipy.optimize.curve_fit``.

    Parameters
    ----------
    x, y : array-like
        Data; non-finite points are dropped.
    model : str or callable
        A ``FIT_MODELS`` name, or ``f(x, p1, p2, ...)``. A callable's
        parameter names come from its signature.
    p0 : sequence, optional
        Initial guess. When omitted, presets guess from the data (peak
        shapes try a peak and a dip guess and keep the better fit); a
        callable starts from all ones (curve_fit's default).
    bounds : (lower, upper), optional
        Forwarded to ``curve_fit``.
    sigma : array-like, optional
        Per-point 1-sigma errors (e.g. per-bin spreads); weights the fit.
    n_points : int, optional
        Samples in the returned fit curve, spanning the data's x range.

    Returns
    -------
    dict, see the module docstring.
    """
    if callable(model):
        func = model
        names = list(inspect.signature(func).parameters)[1:]
        label = getattr(func, '__name__', 'custom')
        label = 'custom' if label == '<lambda>' else label
        guess = None
    elif model in FIT_MODELS:
        func, names, guess = FIT_MODELS[model]
        label = model
    else:
        raise ValueError(f"Unknown fit model {model!r}; use a callable or one of "
                         f"{sorted(FIT_MODELS)}.")

    x, y, sigma = _clean(x, y, sigma)
    if len(x) < len(names) + 1:
        return _failed(label, names, x, n_points,
                       f'{len(x)} points for {len(names)} parameters')
    if p0 is not None or guess is None:
        return _curve_fit(func, label, names, x, y, p0, bounds, sigma, n_points, maxfev)
    # peak-shaped presets: start from both a peak and a dip; keep the better fit
    tries = [_curve_fit(func, label, names, x, y, guess(x, y, dip), bounds, sigma,
                        n_points, maxfev)
             for dip in ((False, True) if model in _PEAK_MODELS else (False,))]
    ok = [r for r in tries if r['success'] and np.isfinite(r['r2'])]
    return max(ok, key=lambda r: r['r2']) if ok else tries[0]


def _curve_fit(func, label, names, x, y, p0, bounds, sigma, n_points, maxfev):
    """One ``curve_fit`` from ``p0``, as a result dict. An undetermined
    covariance gives NaN errors rather than a warning."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', optimize.OptimizeWarning)
            params, cov = optimize.curve_fit(
                func, x, y, p0=p0, sigma=sigma, absolute_sigma=sigma is not None,
                bounds=bounds if bounds is not None else (-np.inf, np.inf), maxfev=maxfev)
    except (RuntimeError, ValueError) as e:
        return _failed(label, names, x, n_points, str(e))

    errors = np.sqrt(np.abs(np.diag(cov))) if np.all(np.isfinite(cov)) \
        else np.full(len(params), np.nan)
    x_fit = _x_fit(x, n_points)
    return _result(label, names, params, errors, _r2(y, func(x, *params)), len(x),
                   x_fit, func(x_fit, *params))


def poly_fit(x, y, degree=2, sigma=None, n_points=200):
    """
    Polynomial of ``degree`` (``np.polyfit`` with its covariance).
    Parameter names are ``p{degree}`` ... ``p0`` (highest power first,
    ``np.polyval`` order).
    """
    names = [f'p{k}' for k in range(degree, -1, -1)]
    label = f'poly{degree}'
    x, y, sigma = _clean(x, y, sigma)
    # cov=True needs more points than parameters + 1
    if len(x) < degree + 3:
        return _failed(label, names, x, n_points,
                       f'{len(x)} points for a degree-{degree} polynomial')
    try:
        w = 1 / sigma if sigma is not None else None
        params, cov = np.polyfit(x, y, degree, w=w, cov='unscaled' if w is not None else True)
    except (np.linalg.LinAlgError, ValueError) as e:
        return _failed(label, names, x, n_points, str(e))
    x_fit = _x_fit(x, n_points)
    return _result(label, names, params, np.sqrt(np.abs(np.diag(cov))),
                   _r2(y, np.polyval(params, x)), len(x), x_fit, np.polyval(params, x_fit))


def linear_regression(x, y, conf=0.95, n_points=200):
    """
    Ordinary least-squares line (``scipy.stats.linregress``) with a
    ``conf`` confidence band on the fitted mean (t distribution, N - 2
    degrees of freedom).

    Extra keys beyond the common ones: ``'r'`` (Pearson), ``'p_value'``
    (two-sided, slope = 0), ``'conf'``.
    """
    names = ['slope', 'intercept']
    x, y, _ = _clean(x, y)
    if len(x) < 3 or np.ptp(x) == 0:
        out = _failed('linear', names, x, n_points, f'{len(x)} points / no x spread')
        out.update(r=np.nan, p_value=np.nan, conf=conf)
        return out
    res = stats.linregress(x, y)
    n = len(x)
    x_fit = _x_fit(x, n_points)
    y_fit = res.intercept + res.slope * x_fit
    resid = y - (res.intercept + res.slope * x)
    s = np.sqrt(np.sum(resid**2) / (n - 2))
    t = stats.t.ppf(0.5 + conf / 2, n - 2)
    half = t * s * np.sqrt(1 / n + (x_fit - x.mean())**2 / np.sum((x - x.mean())**2))
    return _result('linear', names, [res.slope, res.intercept],
                   [res.stderr, res.intercept_stderr], res.rvalue**2, n, x_fit, y_fit,
                   y_lo=y_fit - half, y_hi=y_fit + half,
                   r=res.rvalue, p_value=res.pvalue, conf=conf)


def run_fit(x, y, fit, fit_func=None, p0=None, bounds=None, poly_degree=2, sigma=None):
    """Dispatch on a displayer's ``fit`` argument: ``'linear'``, ``'poly'``,
    ``'custom'`` (uses ``fit_func``), a ``FIT_MODELS`` name, or a callable.
    ``sigma`` is ignored by ``'linear'`` (an unweighted regression)."""
    if fit == 'linear':
        return linear_regression(x, y)
    if fit == 'poly':
        return poly_fit(x, y, degree=poly_degree, sigma=sigma)
    if fit == 'custom':
        if fit_func is None:
            raise ValueError("fit='custom' needs fit_func=f(x, p1, p2, ...).")
        fit = fit_func
    return fit_curve(x, y, fit, p0=p0, bounds=bounds, sigma=sigma)


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def fit_text(result, fmt='{:.4g}'):
    """Multi-line ``name = value ± error`` summary of a fit result."""
    if not result['success']:
        return f"{result['model']} fit failed"
    lines = [f"{result['model']} fit"]
    for name, p, e in zip(result['names'], result['params'], result['errors']):
        lines.append(f"{name} = {fmt.format(p)} ± {fmt.format(e)}")
    if result['model'] == 'linear':
        lines.append(f"r = {result['r']:.3f}, R² = {result['r2']:.3f}")
        lines.append(f"p = {result['p_value']:.2g}, N = {result['n']}")
    else:
        lines.append(f"R² = {result['r2']:.3f}, N = {result['n']}")
    return '\n'.join(lines)


def draw_fit(ax, result, color='C3', label=None, show_params=True, band_alpha=0.2,
             text_loc=(0.02, 0.98)):
    """Fit curve (plus the confidence band, if any) and optionally a text
    box of the parameters in the axes' corner at ``text_loc``."""
    if not result['success']:
        if show_params:
            ax.text(*text_loc, fit_text(result), transform=ax.transAxes, va='top',
                    fontsize='x-small', color=color)
        return
    ax.plot(result['x_fit'], result['y_fit'], color=color, lw=1.5,
            label=label if label is not None else f"{result['model']} fit")
    if result['y_lo'] is not None:
        ax.fill_between(result['x_fit'], result['y_lo'], result['y_hi'], color=color,
                        alpha=band_alpha, lw=0,
                        label=f"{100 * result.get('conf', 0.95):.0f}% confidence")
    if show_params:
        ax.text(*text_loc, fit_text(result), transform=ax.transAxes, va='top',
                fontsize='x-small',
                bbox=dict(boxstyle='round', fc='white', ec=color, alpha=0.85))


def fit_export(result, prefix='fit_'):
    """``last_export`` entries for one fit result."""
    return {
        f'{prefix}x': result['x_fit'],
        f'{prefix}y': result['y_fit'],
        f'{prefix}params': result['params'],
        f'{prefix}errors': result['errors'],
        f'{prefix}param_names': np.asarray(result['names'], dtype=object),
        f'{prefix}r2': np.asarray(result['r2'], dtype=float),
        f'{prefix}model': np.asarray(result['model'], dtype=object),
    }
