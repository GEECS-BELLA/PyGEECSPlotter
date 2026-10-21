"""1-D analysis functions: lineouts, traces and spectra.

Widths at a fraction of the peak, centroids, super-Gaussian fits, moving
averages, trimming zero-padded traces and resampling them onto a grid, and
the optical-spectrum helpers (background on an axis, camera response, spectral
widths / edge wavelengths). The analyzers keep their old static methods as
aliases of these functions.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


# ---------------------------------------------------------------- lineouts

def super_gaussian(x, amplitude, center, sigma, offset, N=2):
    """
    Basic super‐Gaussian model:
       f(x) = amplitude * exp(-0.5 * ((x-center)/sigma)^N ) + offset
    """
    if sigma <= 0:
        return np.full_like(x, np.inf)
    return amplitude * np.exp(-0.5 * np.abs((x - center)/sigma)**N) + offset


def calculate_moving_average_and_std(data, window_size):
    """
    Calculate the moving average and standard deviation for a given data array, returning them as NumPy arrays.

    Parameters:
    -----------
    data : array-like
        The input data for which the moving average and standard deviation will be calculated.
        
    window_size : int
        The size of the moving window. This determines how many data points are considered for each calculation of the moving average and standard deviation.

    Returns:
    --------
    moving_average : np.ndarray
        The moving average of the input data over the specified window size.
        
    moving_std : np.ndarray
        The moving standard deviation of the input data over the specified window size.

    """
    data_series = pd.Series(data)
    moving_average = data_series.rolling(window=window_size).mean().to_numpy()
    moving_std = data_series.rolling(window=window_size).std().to_numpy()
    return moving_average, moving_std


def get_lineout_width(lineout, x0=None, max_data=None, from_center=True, width_at=0.5):
    """
    Find the width of a lineout at a specified fraction of the maximum value
    (e.g. half‐max).

    Parameters
    ----------
    lineout : np.ndarray
        1D array of values (e.g., a horizontal/vertical slice of data).
    x0 : float or int
        A reference index for the lineout (e.g. the center or peak location).
    max_data : float, optional
        The maximum value of lineout; if None, it will be computed internally.
    from_center : bool, optional
        If True, find left/right crossing around x0. If False, find the full
        extent over the entire lineout.
    width_at : float, optional
        Fraction of the maximum value at which to measure the width.
        For example, 0.5 => FWHM.

    Returns
    -------
    width : float
        The width in number of points (or indices).
    high_idx : float
        The right index where the lineout crosses 'width_at'.
    low_idx : float
        The left index where the lineout crosses 'width_at'.
    """
    if max_data is None:
        max_data = np.nanmax(lineout)
    threshold = width_at * max_data
    if x0 is None:
        x0 = np.argmax(lineout)
    x0 = int(x0)

    # If from_center, measure half-max from x0 outward
    if from_center:
        if x0 < 0 or x0 >= len(lineout):
            return np.nan, np.nan, np.nan
        left_indices = np.where(lineout[:x0] > threshold)[0]
        right_indices = np.where(lineout[x0:] > threshold)[0]
        if (len(left_indices) == 0) or (len(right_indices) == 0):
            return np.nan, np.nan, np.nan
        ldx = left_indices[0]
        hdx = right_indices[-1] + x0
        return (hdx - ldx), hdx, ldx
    else:
        # Over entire array
        indices = np.where(lineout > threshold)[0]
        if len(indices) > 0:
            return indices[-1] - indices[0], indices[-1], indices[0]
        else:
            return np.nan, np.nan, np.nan


def width_on_axis(axis, amp, width_at=0.5):
    """
    Width of ``amp`` at ``width_at`` of its peak, in axis units.

    ``get_lineout_width`` returns ``(width, high_idx, low_idx)`` — note the
    high index comes first.
    """
    _, hdx, ldx = get_lineout_width(amp, from_center=False, width_at=width_at)
    if not np.isfinite(hdx) or not np.isfinite(ldx):
        return np.nan
    return float(axis[int(hdx)] - axis[int(ldx)])


def centroid_1d(axis, amp):
    weights = np.nan_to_num(amp, nan=0.0)
    total = np.sum(weights)
    return float(np.sum(axis * weights) / total) if total > 0 else np.nan


def normalise(amp):
    peak = np.nanmax(amp) if amp.size else np.nan
    return amp / peak if np.isfinite(peak) and peak > 0 else amp


def fit_super_gaussian(xdata, ydata, fit_exponent=False, p0=None, bounds=None,
                       loss='linear', max_nfev=5000):
    """
    Fit a super‐Gaussian function to (xdata,ydata).
    """
    xdata = np.array(xdata, dtype=float)
    ydata = np.array(ydata, dtype=float)

    valid_mask = np.isfinite(xdata) & np.isfinite(ydata)
    xdata = xdata[valid_mask]
    ydata = ydata[valid_mask]
    if xdata.size == 0:
        return {'success': False, 'message': 'No valid data'}

    def estimate_initial_params(xx, yy):
        y_min, y_max = np.nanmin(yy), np.nanmax(yy)
        amp_guess    = y_max - y_min
        idx_max      = np.nanargmax(yy)
        cen_guess    = xx[idx_max]
        off_guess    = y_min

        half_level = y_min + 0.5*amp_guess
        crosses = np.where(np.diff((yy > half_level).astype(int)) != 0)[0]
        if len(crosses) >= 2:
            x_left, x_right = xx[crosses[0]], xx[crosses[-1]]
            sigma_guess = (x_right - x_left)/2.355
            sigma_guess = max(1e-6, np.abs(sigma_guess))
        else:
            sigma_guess = max(1e-6, (xx.max() - xx.min())/10)
        return [amp_guess, cen_guess, sigma_guess, off_guess]

    if p0 is None:
        p0_est = estimate_initial_params(xdata, ydata)
        if fit_exponent:
            p0_est.append(2.0)
        p0 = p0_est

    if bounds is None:
        if fit_exponent:
            bounds = ([0, -np.inf, 1e-12, -np.inf, 1],
                      [np.inf, np.inf, np.inf, np.inf, 20])
        else:
            bounds = ([0, -np.inf, 1e-12, -np.inf],
                      [np.inf, np.inf, np.inf, np.inf])

    def residual(params, xx, yy):
        if fit_exponent:
            amp, cen, sig, off, N = params
        else:
            amp, cen, sig, off = params
            N = 2
        return yy - super_gaussian(xx, amp, cen, sig, off, N)

    res = least_squares(residual, p0, args=(xdata, ydata),
                        bounds=bounds, loss=loss, max_nfev=max_nfev)

    result = {
        'success':  res.success,
        'message':  res.message,
        'cost':     res.cost,
        'params':   res.x
    }
    if fit_exponent:
        amp, cen, sig, off, exponent = res.x
        result.update({'amplitude': amp, 'center': cen, 'sigma': sig,
                       'offset': off, 'N': exponent})
    else:
        amp, cen, sig, off = res.x
        result.update({'amplitude': amp, 'center': cen, 'sigma': sig,
                       'offset': off})
    return result


def interpolate_nan_values(data):
    """
    Interpolate over 1D array's NaN regions using nearest valid points.
    """
    nans = np.isnan(data)
    notnans = ~nans
    if not np.any(notnans):
        return data  # all NaNs
    indices = np.arange(len(data))
    data[nans] = np.interp(indices[nans], indices[notnans], data[notnans])
    return data


def angular_difference_deg(theta, theta0):
    """
    Smallest signed angular difference (theta - theta0) in degrees, wrapped to [-180, 180).
    theta, theta0 can be scalars or arrays.
    """
    return (theta - theta0 + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- resampling

def valid_length(axis):
    """
    Number of leading rows before the zero padding starts.

    The retrieval writes a monotonic axis and pads the remainder with
    zeros, so the padding begins where the axis first stops moving in its
    initial direction. Returns the full length if it never does.

    A descending axis needs the extra zero-strip below: its step down into
    the padding is still a decrease, so monotonicity alone does not spot
    the first padded row (only the flat 0 -> 0 after it).
    """
    axis = np.asarray(axis, dtype=float)
    if axis.size < 2:
        return axis.size

    steps = np.diff(axis)
    direction = steps[0]
    if direction == 0:
        return 1

    stalled = np.where(steps * np.sign(direction) <= 0)[0]
    n = int(stalled[0]) + 1 if stalled.size else axis.size

    # Trailing exact zeros are padding. A real axis may pass through 0
    # (time does), but only in its interior, never at the end of the run.
    while n > 1 and axis[n - 1] == 0:
        n -= 1

    return n


def resolve_grid(grid, axis):
    """
    Turn a grid spec into an array.

    Accepts an explicit array, a ``(min, max, n)`` triple, or ``None`` to
    keep the shot's own axis (which leaves shots un-alignable, so the
    scan-level displayers ask for an explicit grid).
    """
    if grid is None:
        return np.asarray(axis, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if grid.size == 3 and grid.ndim == 1:
        lo, hi, n = grid
        return np.linspace(lo, hi, int(n))
    return grid


def resample_trace(axis, grid, amp, phase):
    """
    Interpolate a trace onto ``grid``.

    Points outside the shot's own support become NaN rather than being
    clamped to the edge value, so a short retrieval does not masquerade as
    a wide flat pulse.
    """
    grid = np.asarray(grid, dtype=float)
    if axis.size < 2:
        nan = np.full(grid.shape, np.nan)
        return grid, nan, nan.copy()

    outside = (grid < axis[0]) | (grid > axis[-1])
    amp_out = np.interp(grid, axis, amp)
    phase_out = np.interp(grid, axis, phase)
    amp_out[outside] = np.nan
    phase_out[outside] = np.nan
    return grid, amp_out, phase_out


def get_TW_per_J(t_fs, intensity, t_min=None, t_max=None):
    """
    Peak laser power per joule of pulse energy, in TW/J.

    Parameters
    ----------
    t_fs : array
        Time axis in femtoseconds.
    intensity : array
        FROG-retrieved temporal intensity in arbitrary units (the
        normalisation cancels).
    t_min, t_max : float or None
        Integration bounds in fs. If None, use the full range.

    Returns
    -------
    float
        ``max(I) / integral(I dt)``. With time in fs the 1e15 s/fs and the
        1e-12 W/TW combine into the factor of 1000.
    """
    t_fs = np.asarray(t_fs, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    mask = np.ones_like(t_fs, dtype=bool)
    if t_min is not None:
        mask &= t_fs >= t_min
    if t_max is not None:
        mask &= t_fs <= t_max
    if mask.sum() < 2:
        return np.nan

    t_sel = t_fs[mask]
    i_sel = intensity[mask]
    integral = np.trapezoid(i_sel, t_sel)
    if not np.isfinite(integral) or integral <= 0:
        return np.nan

    return float(1000 * np.nanmax(i_sel) / integral)


# ---------------------------------------------------------------- spectra

def camera_response_ratio(wl, counts, cam_wl, cam_response):
    """
    Fraction of the spectrum a camera registers:
    ``sum(S * r) / sum(S)``, with ``r`` the camera's spectral response
    interpolated onto ``wl`` (0 outside the curve's range) and ``S`` the
    spectrum clipped at 0 (residual negative counts after background
    subtraction would otherwise make the ratio noisy on dim shots).

    Plain sums, so ``r`` is taken as a response per unit energy and
    ``wl`` as a uniform grid (true for ``wl_lin`` and the combined
    spectrum). NaN when the spectrum has no positive signal.
    """
    s = np.clip(np.nan_to_num(np.asarray(counts, dtype=float), nan=0.0), 0, None)
    total = s.sum()
    if total <= 0:
        return np.nan
    r = np.interp(wl, cam_wl, cam_response, left=0.0, right=0.0)
    return float(np.sum(s * r) / total)


def background_on_axis(bg, wl):
    """
    Background counts on the shot's wavelength axis ``wl``. ``bg`` is a
    'Wavelength (nm)' / 'Counts' DataFrame, a (n, 2) array of the same
    columns (as ``scan.mean_std_diagnostic`` returns), or a bare counts
    array already on ``wl``. With a wavelength column, a background on
    a different axis is interpolated rather than mis-subtracted.
    """
    if isinstance(bg, pd.DataFrame):
        bg_wl = np.asarray(bg['Wavelength (nm)'].values, dtype=float)
        bg_counts = np.asarray(bg['Counts'].values, dtype=float)
    else:
        arr = np.asarray(bg, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            bg_wl, bg_counts = arr[:, 0], arr[:, 1]
        else:
            if arr.shape != wl.shape:
                raise ValueError(f"Background has {arr.shape} counts but the spectrum has "
                                 f"{wl.shape}; pass a DataFrame with a wavelength column.")
            return arr
    if bg_wl.shape == wl.shape and np.allclose(bg_wl, wl):
        return bg_counts
    return np.interp(wl, bg_wl, bg_counts)


def window_mean(x, y, x_low, x_high):
    """Mean of ``y`` between the samples of ``x`` nearest ``x_low`` and ``x_high``
    (end exclusive); None if the window is empty."""
    ldx = int(np.argmin(np.abs(x - x_low)))
    hdx = int(np.argmin(np.abs(x - x_high)))
    if hdx <= ldx:
        return None
    return np.nanmean(y[ldx:hdx])


def safe_index(arr, idx):
    try:
        return arr[int(idx)]
    except (IndexError, ValueError, TypeError):
        return np.nan


# (label, fraction of the peak) for the width / edge wavelengths
WIDTH_LEVELS = [('half max', 0.5), ('1/e', 1.0 / np.e), ('5pct', 0.05), ('1pct', 0.01)]


def compute_spectrum_shifts(data, wl, threshold=0, fill=0):
    """
    Compute FWHM / 1-e / 5% / 1% widths and the bracketing (blue, red)
    wavelengths. Every value is ``fill`` if ``max(data) < threshold``.
    """
    if not np.any(np.isfinite(data)) or np.nanmax(data) < threshold:
        keys = []
        for label, _ in WIDTH_LEVELS:
            keys += ['fwhm (nm)' if label == 'half max' else f'width at {label} (nm)',
                     f'lambda_b {label} (nm)', f'lambda_r {label} (nm)']
        return dict.fromkeys(keys, fill)

    safe = safe_index
    x0 = int(np.argmax(np.nan_to_num(data, nan=-np.inf)))

    result = {}
    for label, frac in WIDTH_LEVELS:
        _, ldx, hdx = get_lineout_width(data, x0, from_center=False, width_at=frac)
        lam_b = safe(wl, ldx)
        lam_r = safe(wl, hdx)
        width_key = 'fwhm (nm)' if label == 'half max' else f'width at {label} (nm)'
        result[width_key] = lam_r - lam_b
        result[f'lambda_b {label} (nm)'] = lam_b
        result[f'lambda_r {label} (nm)'] = lam_r
    return result


# (label, fraction of the total) for the cumulative-sum edges; the red
# side uses 1 - fraction
CUMULATIVE_LEVELS = [('1pct', 0.01), ('5pct', 0.05), ('10pct', 0.10),
                     ('20pct', 0.20), ('1/e', 1.0 / np.e)]


def compute_cumulative_spectrum_shifts(wl, data, threshold=0, fill=0):
    """
    Wavelengths at which the cumulative sum first crosses 1%, 5%, 10%,
    20%, 1/e of the total (blue side, ``lambda_b``) and 1 minus those
    (red side, ``lambda_r``). Every value is ``fill`` if
    ``max(data) < threshold``.
    """
    keys = [f'lambda_{side} cumulative {label} (nm)'
            for side in 'br' for label, _ in CUMULATIVE_LEVELS]
    if not np.any(np.isfinite(data)) or np.nanmax(data) < threshold:
        return dict.fromkeys(keys, fill)

    cumsum = np.nancumsum(data)
    total = np.nansum(data)
    if total <= 0:
        return dict.fromkeys(keys, fill)

    def first_wl_above(frac):
        idx = np.where(cumsum > frac * total)[0]
        return float(wl[idx[0]]) if idx.size else np.nan

    result = {}
    for label, frac in CUMULATIVE_LEVELS:
        result[f'lambda_b cumulative {label} (nm)'] = first_wl_above(frac)
    for label, frac in CUMULATIVE_LEVELS:
        result[f'lambda_r cumulative {label} (nm)'] = first_wl_above(1.0 - frac)
    return result
