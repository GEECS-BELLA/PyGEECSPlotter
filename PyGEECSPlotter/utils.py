import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import least_squares

from PyGEECSPlotter.ecs_live_dumps import get_ecs_live_dumps_parameter_value_from_sfilename

def _convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj
    
def find_matching_row_index(df, keys, d, tolerances=None, default_tol=1e-3):
    """
    Return index of row where all specified columns match dictionary values
    within per-column tolerances. If a column has no tolerance specified,
    `default_tol` is used.
    """
    if tolerances is None:
        tolerances = {}

    mask = pd.Series(True, index=df.index)

    for k in keys:
        if k not in df.columns:
            raise KeyError(f"Column '{k}' not in DataFrame.")
        if k not in d:
            raise KeyError(f"Key '{k}' not found in the dictionary.")

        ref_val = d[k]
        tol = tolerances.get(k, default_tol)  # use provided or default

        mask &= np.abs(df[k] - ref_val) <= tol

    matches = df[mask]

    if matches.empty:
        return None
    return matches.index

def write_controls_from_python(filename, controls_dict, print_data=False):
    controls_dict_serializable = _convert_to_serializable(controls_dict)
    with open(filename, 'w') as file:
        json.dump(controls_dict_serializable, file, indent=4)
    if print_data:
        print(f'Controls saved to : {filename}')

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


def merge_dicts_overwrite(*dicts):
    """
    Merge multiple dictionaries by overwriting overlapping keys.
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result
    
def polynomial_fit(x, y, degree=1, xmin=None, xmax=None, num_points=100, force_through_point=False, x0=0, y0=0):
    """
    Fits a polynomial function to data, with an option to force it through a specific point (x0, y0).

    Parameters:
    x (array-like): x data points
    y (array-like): y data points
    degree (int): degree of the polynomial fit
    xmin (float): minimum x value for the fitted line
    xmax (float): maximum x value for the fitted line
    num_points (int): number of points for the fitted line
    force_through_point (bool): If True, forces the fit through (x0, y0)
    x0 (float): x-coordinate of the point to force the fit through
    y0 (float): y-coordinate of the point to force the fit through

    Returns:
    x_fitted (numpy array): x values of the fitted polynomial
    y_fitted (numpy array): y values of the fitted polynomial
    p (array): coefficients of the fit
    """
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)

    try:
        if force_through_point:
            # Shift data to make (x0, y0) the origin for fitting
            x_shifted = x - x0
            y_shifted = y - y0

            # Build the Vandermonde matrix for a polynomial of given degree
            A = np.vander(x_shifted, degree + 1)
            A[:, -1] = 0  # Force the intercept term to be zero (through (x0, y0))

            # Solve for coefficients
            p_shifted, _, _, _ = np.linalg.lstsq(A, y_shifted, rcond=None)
            p = np.append(p_shifted[:-1], y0)  # Add y0 as the intercept term
        else:
            # Standard polynomial fit
            p = np.polyfit(x, y, degree)

        # Generate fitted data points
        x_fitted = np.linspace(xmin, xmax, num_points)
        y_fitted = np.polyval(p, x_fitted)

        return x_fitted, y_fitted, p
    except Exception as e:
        # Return zeros if fitting fails
        p = np.zeros(degree + 1)
        x_fitted = np.linspace(xmin, xmax, num_points)
        y_fitted = np.zeros_like(x_fitted)

        return x_fitted, y_fitted, p

def plot_target_ellipse(ax, x0=0, y0=0, x_size=1.0, y_size=1.0, 
                       line_color='black', linewidth=0.5):

    # Draw ellipse
    ellipse = Ellipse((x0, y0), width=2*x_size, height=2*y_size, 
                      fill=False, color=line_color, linewidth=linewidth)
    ax.add_patch(ellipse)

    # Draw crosshairs
    ax.plot([x0 - x_size, x0 + x_size], [y0, y0], color=line_color, linewidth=linewidth)  # Horizontal
    ax.plot([x0, x0], [y0 - y_size, y0 + y_size], color=line_color, linewidth=linewidth)  # Vertical

    return ax, ellipse

def plot_alignment_overview(diagnostic_dicts,
                            sfilename, 
                            display_dict={} 
                           ):
    """
    Plot a grid of diagnostic images with optional target crosshairs.

    Parameters
    ----------
    diagnostic_dicts : list of dict
        Each dict must contain at least keys: 'data', 'cmap', 'diagnostic', 
        'target_on', 'crosshair', and optionally 'vmin', 'vmax'.
    sfilename : str
        Filename used to extract ECS parameter values.
    display_dict : dict
        Controls display options such as 'vmin', 'axes_on', 'oneline', 'aspect', and 'norm'.
    get_ecs_live_dumps_parameter_value_from_sfilename : callable
        Function to extract ECS live dump values.
    plot_target_ellipse : callable
        Function to draw an ellipse on an axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
    ellipses : list of ellipse artists (or None if not used)
    """

    vmin = display_dict.get('vmin', 0)
    axes_on = display_dict.get('axes_on', False)
    oneline = display_dict.get('oneline', False)

    # Determine grid size
    N = len(diagnostic_dicts)
    if oneline:
        n = N
        m = 1
    else:
        n = int(np.ceil(np.sqrt(N)))
        m = int(np.ceil(N / n))

    # Create figure and axes
    fig, axes = plt.subplots(m, n, figsize=(n * 2.5, m * 2.5))
    axes = axes.flatten()

    ellipses = []

    # Plot each image
    for i, d in enumerate(diagnostic_dicts):
        im = axes[i].imshow(
            d['data'],
            aspect=display_dict.get('aspect', 'equal'),
            norm=display_dict.get('norm', None),
            cmap=d['cmap'],
            origin='lower',
            vmin=d.get('vmin', vmin),
            vmax=d.get('vmax', np.nanmax(d['data'])),
        )

        ellipse = None
        if d.get('target_on', False):
            if d.get('crosshair') == 1:
                _, x0 = get_ecs_live_dumps_parameter_value_from_sfilename(sfilename, d['diagnostic'], 'Target.X')
                _, y0 = get_ecs_live_dumps_parameter_value_from_sfilename(sfilename, d['diagnostic'], 'Target.Y')
                _, x_size = get_ecs_live_dumps_parameter_value_from_sfilename(sfilename, d['diagnostic'], 'TargetSize.X')
                _, y_size = get_ecs_live_dumps_parameter_value_from_sfilename(sfilename, d['diagnostic'], 'TargetSize.Y')
            elif d.get('crosshair') == 2:
                _, x0 = get_ecs_live_dumps_parameter_value_from_sfilename(sfilename, d['diagnostic'], 'Target2.X')
                _, y0 = get_ecs_live_dumps_parameter_value_from_sfilename(sfilename, d['diagnostic'], 'Target2.Y')
                _, x_size = get_ecs_live_dumps_parameter_value_from_sfilename(sfilename, d['diagnostic'], 'Target2Size.X')
                _, y_size = get_ecs_live_dumps_parameter_value_from_sfilename(sfilename, d['diagnostic'], 'Target2Size.Y')
            axes[i], ellipse = plot_target_ellipse(axes[i], x0=x0, y0=y0, x_size=x_size, y_size=y_size)

        ellipses.append(ellipse)

        axes[i].set_title(d['diagnostic'])
        if not axes_on:
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])

    # Turn off unused axes
    for j in range(len(diagnostic_dicts), m * n):
        axes[j].axis('off')

    fig.suptitle(sfilename)
    plt.show()

    return fig, axes

def parse_controls_from_python(filename):
    with open(filename, 'r') as file:
        controls_dict = json.load(file)
    return controls_dict


def gaussian_2d(xydata, amplitude, center_x, center_y, sigma_x, sigma_y, offset):
    x, y = xydata
    if sigma_x <= 0 or sigma_y <= 0:
        return np.full_like(x, np.inf)
    exponent = -0.5 * ( ((x - center_x)/sigma_x)**2 + ((y - center_y)/sigma_y)**2 )
    return amplitude * np.exp(exponent) + offset

def fit_gaussian_2d(xdata, ydata, zdata, p0=None, bounds=None,
                    loss='linear', max_nfev=5000):
    xdata = np.array(xdata, dtype=float)
    ydata = np.array(ydata, dtype=float)
    zdata = np.array(zdata, dtype=float)
    if xdata.shape != ydata.shape or xdata.shape != zdata.shape:
        return {'success': False,
                'message': 'xdata, ydata, and zdata must have the same shape.'}

    valid_mask = np.isfinite(xdata) & np.isfinite(ydata) & np.isfinite(zdata)
    if not np.any(valid_mask):
        return {'success': False, 'message': 'No valid data'}

    x_valid = xdata[valid_mask]
    y_valid = ydata[valid_mask]
    z_valid = zdata[valid_mask]

    xy_valid = (x_valid, y_valid)

    def estimate_initial_params(x_arr, y_arr, z_arr):
        z_min, z_max = np.nanmin(z_arr), np.nanmax(z_arr)
        amp_guess = z_max - z_min
        offset_guess = z_min
        # center guess at the peak
        idx_max = np.argmax(z_arr)
        cx_guess = x_arr[idx_max]
        cy_guess = y_arr[idx_max]
        # rough sigma guesses
        half_level = z_min + 0.5 * amp_guess
        above_half = z_arr > half_level
        if not np.any(above_half):
            sigma_x_guess = (np.max(x_arr) - np.min(x_arr)) / 6.0
            sigma_y_guess = (np.max(y_arr) - np.min(y_arr)) / 6.0
        else:
            x_half = x_arr[above_half]
            y_half = y_arr[above_half]
            sigma_x_guess = np.sqrt(np.mean((x_half - cx_guess)**2)) or 1e-3
            sigma_y_guess = np.sqrt(np.mean((y_half - cy_guess)**2)) or 1e-3
        return [amp_guess, cx_guess, cy_guess, sigma_x_guess, sigma_y_guess, offset_guess]

    if p0 is None:
        p0 = estimate_initial_params(x_valid, y_valid, z_valid)

    if bounds is None:
        lower_bounds = [0.0, -np.inf, -np.inf, 1e-12, 1e-12, -np.inf]
        upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        bounds = (lower_bounds, upper_bounds)

    def residual(params, xy, z):
        model = gaussian_2d(xy, *params)
        return z - model

    res = least_squares(residual, x0=p0, args=(xy_valid, z_valid),
                        bounds=bounds, loss=loss, max_nfev=max_nfev)

    amp, cx, cy, sx, sy, offset = res.x
    return {
        'success':    res.success,
        'message':    res.message,
        'cost':       res.cost,
        'params':     res.x,
        'amplitude':  amp,
        'center_x':   cx,
        'center_y':   cy,
        'sigma_x':    sx,
        'sigma_y':    sy,
        'offset':     offset
    }



def save_lineouts_to_txt(lineouts, filename="lineouts.txt",
                         cols=None, float_format="%.3f", na_rep="nan"):
    """
    Save lineouts dict (arrays of different lengths) to a tab-delimited text file,
    padding with NaNs.

    Parameters
    ----------
    lineouts : dict
        Dict of 1D arrays, e.g. {'x': ..., 'x_lo': ..., 'y': ..., ...}
        Some keys may be missing (e.g. only 'x' and 'x_lo').
    filename : str
        Output filename.
    cols : list of str or None
        Column order to use. If None, use lineouts.keys() in their existing order.
        If provided, any names not found in lineouts are silently skipped.
    float_format : str
        Format for floats, e.g. '%.3f'.
    na_rep : str
        Representation of NaN in the file.
    """
    if cols is None:
        cols = list(lineouts.keys())
    else:
        # Only keep columns that actually exist in lineouts
        cols = [c for c in cols if c in lineouts]

    # Convert to arrays and find max length
    arrays = {k: np.asarray(lineouts[k]) for k in cols}
    max_len = max(len(arr) for arr in arrays.values())

    # Pad with NaN
    padded = {
        k: np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
        for k, arr in arrays.items()
    }

    df = pd.DataFrame(padded, columns=cols)
    df.to_csv(filename, sep="\t", index=False,
              float_format=float_format, na_rep=na_rep)
    return df


def load_lineouts_from_txt(filename):
    """
    Load lineouts from a tab-delimited file created by save_lineouts_to_txt.

    Automatically detects coord/value pairs of the form:
        <coord>, <coord> + '_lo'
    and uses the coord column's NaNs to remove padding from both.

    Returns
    -------
    lineouts : dict of np.ndarray
        Dict with original-length arrays.
    df : pandas.DataFrame
        The raw DataFrame as read from file (still padded with NaNs).
    """
    df = pd.read_csv(filename, sep="\t")

    cols = list(df.columns)
    lineouts = {}

    # --- Detect coord/value pairs automatically ---
    coord_pairs = []
    for col in cols:
        if not col.endswith("_lo"):
            lo_col = col + "_lo"
            if lo_col in df.columns:
                coord_pairs.append((col, lo_col))

    # Process each coord/value pair
    used_cols = set()
    for coord, val in coord_pairs:
        coord_vals = df[coord].to_numpy()
        mask = ~np.isnan(coord_vals)  # keep only real entries, drop padded NaNs

        lineouts[coord] = coord_vals[mask]
        lineouts[val] = df[val].to_numpy()[mask]

        used_cols.add(coord)
        used_cols.add(val)

    # Any remaining columns (not part of coord pairs) are returned as-is
    for col in cols:
        if col not in used_cols:
            lineouts[col] = df[col].to_numpy()

    return lineouts
