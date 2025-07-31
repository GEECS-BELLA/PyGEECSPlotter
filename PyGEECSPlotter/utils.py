import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
# from PyGEECSPlotter.ecs_live_dumps import get_ecs_live_dumps_parameter_value_from_sfilename

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
