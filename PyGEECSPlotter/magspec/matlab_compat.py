# MATLAB-compatible numerical primitives used by the magspec port.
# Only the behaviour the magspec code relies on is reproduced: NaN outside
# the sample range, automatic sorting of the sample points, 'cubic' on
# non-uniform samples (MATLAB falls back to a not-a-knot spline), and
# round-half-away-from-zero.

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, RegularGridInterpolator


def mround(a):
    """MATLAB ``round``: half away from zero (numpy rounds half to even)."""
    a = np.asarray(a, dtype=float)
    return np.sign(a) * np.floor(np.abs(a) + 0.5)


def interp1(x, y, xi, method='linear'):
    """MATLAB ``interp1(x, y, xi, method)`` with NaN extrapolation.

    ``y`` may be 1-D (same length as ``x``) or 2-D with ``len(x)`` rows,
    in which case each column is interpolated (MATLAB column semantics).
    Unsorted ``x`` is sorted first, as MATLAB does. ``method`` is
    ``'linear'``, ``'pchip'``, or ``'cubic'``/``'spline'`` -- MATLAB's
    ``'cubic'`` (cubic convolution) falls back to a not-a-knot spline for
    unevenly spaced ``x``, which is always the case for the magspec
    trajectory tables.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float)
    xi_arr = np.asarray(xi, dtype=float)
    vec_y = y.ndim == 1
    if vec_y:
        y = y.reshape(-1, 1)
    if y.shape[0] != x.size and y.shape[1] == x.size and vec_y is False:
        raise ValueError('y must have len(x) rows')
    order = np.argsort(x, kind='stable')
    x = x[order]
    y = y[order]
    if np.any(np.diff(x) == 0):
        raise ValueError('interp1: sample points must be unique')
    xq = xi_arr.ravel()
    if method == 'linear':
        out = np.empty((xq.size, y.shape[1]))
        for j in range(y.shape[1]):
            out[:, j] = np.interp(xq, x, y[:, j])
    elif method == 'pchip':
        out = PchipInterpolator(x, y, axis=0, extrapolate=False)(xq)
    elif method in ('cubic', 'spline'):
        out = CubicSpline(x, y, axis=0, bc_type='not-a-knot', extrapolate=False)(xq)
    else:
        raise ValueError(f'unsupported method {method!r}')
    outside = (xq < x[0]) | (xq > x[-1]) | np.isnan(xq)
    out[outside, :] = np.nan
    if vec_y:
        return out[:, 0].reshape(xi_arr.shape)
    return out.reshape(xi_arr.shape + (y.shape[1],))


def interp2(xg, yg, z, xq, yq):
    """MATLAB ``interp2(X, Y, Z, Xq, Yq)`` (linear) for meshgrid-style
    regular grids given as 1-D axes ``xg`` (columns) and ``yg`` (rows).
    Points outside the grid are NaN."""
    f = RegularGridInterpolator((np.asarray(yg, float), np.asarray(xg, float)),
                                np.asarray(z, float), method='linear',
                                bounds_error=False, fill_value=np.nan)
    pts = np.stack([np.asarray(yq, float).ravel(), np.asarray(xq, float).ravel()], axis=-1)
    return f(pts).reshape(np.shape(xq))


def find_first(mask):
    """0-based index of the first True, or None (MATLAB ``find(...,1,'first')``)."""
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def find_last(mask):
    """0-based index of the last True, or None (MATLAB ``find(...,1,'last')``)."""
    idx = np.flatnonzero(mask)
    return int(idx[-1]) if idx.size else None


def nearest_index(arr, value):
    """MATLAB ``[~, i] = min(abs(arr - value))`` (first minimum), 0-based."""
    return int(np.nanargmin(np.abs(np.asarray(arr, float) - value)))
