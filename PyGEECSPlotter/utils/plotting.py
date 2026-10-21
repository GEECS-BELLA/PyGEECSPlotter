import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib import font_manager
from matplotlib.patches import Ellipse

def cmap_to_rgb_array(cmap_name='Reds', num_samples=256):
    # Get the colormap from matplotlib
    cmap = plt.get_cmap(cmap_name)
    
    # Generate an array of values from 0 to 1 to sample the colormap
    values = np.linspace(0, 1, num_samples)
    
    # Apply the colormap to get RGBA values and take only the RGB part
    rgb_array = cmap(values)[:, :3]
    
    return rgb_array

def rgb_array_to_cmap(rgb_array, cmap_name='custom_cmap'):
    """
    Convert an RGB array into a matplotlib colormap.
    """
    # Ensure the RGB values are within the range [0, 1]
    rgb_array = np.clip(rgb_array, 0, 1)

    # Create a ListedColormap from the RGB array
    cmap = ListedColormap(rgb_array, name=cmap_name)
    return cmap


def is_font_available(font_name):
    """
    Check if the given font is available in the system.
    
    Parameters:
    font_name (str): Name of the font to check.
    
    Returns:
    bool: True if the font is found, False otherwise.
    """
    available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
    return font_name in available_fonts

def configure_plotting(fontsize=12, dpi=100, sf=1.0):
    """
    Configures matplotlib settings for plotting.

    Parameters:
    fontsize (int or float): Base font size for the plots. Default is 12.
    dpi (int): Dots per inch for figure resolution. Default is 200.
    """
    if is_font_available('CMU Sans Serif'):
        font = {'family' : 'CMU Sans Serif',
                'weight' : 'normal',
                'size'   : sf*fontsize}
        
        plt.rcParams['font.sans-serif'] = "CMU Sans Serif"
        plt.rcParams['font.family'] = "sans-serif"
        plt.rcParams['mathtext.fontset'] = 'custom' 
        matplotlib.rcParams['axes.unicode_minus'] = False
        
    else:
        font = {'size'   : sf*fontsize}
        print('Not using CMU Sans Serif')
    matplotlib.rc('font', **font)
    
    # Tick settings
    plt.rcParams['xtick.direction'] = 'in' 
    plt.rcParams['ytick.direction'] = 'in' 
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True

    # # Tick label size
    # plt.rcParams['xtick.labelsize'] = fontsize - 2
    # plt.rcParams['ytick.labelsize'] = fontsize - 2
    # plt.rcParams['xtick.major.size'] = 5  # Major tick size
    # plt.rcParams['ytick.major.size'] = 5
    # plt.rcParams['xtick.minor.size'] = 3  # Minor tick size
    # plt.rcParams['ytick.minor.size'] = 3

    # Legend settings
    plt.rcParams['legend.fontsize'] = sf*(fontsize - 2)
    plt.rcParams['legend.loc'] = 'best'  # 'upper right', 'lower left', etc.

    # Line width
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 6.0

    # Figure DPI and size
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['figure.figsize'] = (6.4, 4.8)  # (width, height) in inches

    # Savefig settings
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['savefig.format'] = 'png'  # Default format for saving plots
    plt.rcParams['savefig.bbox'] = 'tight'  # Reduce excess white space

    # Title and label sizes
    plt.rcParams['axes.labelsize'] = sf*fontsize
    plt.rcParams['axes.titlesize'] = sf*fontsize

    # Error bar settings
    plt.rcParams['errorbar.capsize'] = 3  # Length of error bar caps

    # Retrieve the color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    return colors

colors = configure_plotting(fontsize=12, dpi=100, sf=1.0)

# Use this to format log colorbar ticks in matplotlib with mathtext (e.g., $10^{-2}$)
# This is a workaround to CMU Sans Serif not having the "-" (U+2212) character that matplotlib
# uses by default for the minus sign in mathtext.
# We use the default '-' character instead, which is available in CMU Sans Serif.
def log_mathtext_formatter(x, pos=None):
    if x <= 0:
        return ""
    exponent = int(np.round(np.log10(x)))
    if exponent == 0:
        return r"$1$"
    return rf"$10^{{{exponent}}}$"

# Same as above function but insstead of display as 10^{x} display a decimal instead
def log_decimal_formatter(x, pos=None):
    """Format log tick values as plain decimals: 0.01, 0.1, 1, 10, 100."""
    if x <= 0:
        return ""
    if x >= 1:
        s = f"{x:g}"                      # 1, 10, 100
    else:
        decimals = max(0, -int(np.floor(np.log10(x))))
        s = f"{x:.{decimals}f}"          # 0.1, 0.01, ...
    return rf"${s}$"                      # mathtext-wrapped, no U+2212 issue

# Example usage for these formatters:
"""
from matplotlib.ticker import FuncFormatter
fig.colorbar(img, cax=cax, label="Charge Density [pC/GeV]", format=FuncFormatter(plotting.log_decimal_formatter))
"""


# ---------------------------------------------------------------- overlays

def overlay_lineouts(ax, x=None, y=None, x_lo=None, y_lo=None, amp=0.25, normalize=True,
                     style_dict=None):
    """
    Draw lineouts over an image, inside the current view window: x_lo
    along the bottom edge (peak at `amp` of the window height), y_lo
    along the left edge (peak at `amp` of the window width). Call after
    the axis limits are set. Each lineout is scaled to its own max when
    normalize=True; otherwise it is used as given (already 0-1).

    style_dict: matplotlib line keywords (default color 'C3', lw 1,
    alpha 0.8). Returns (x_lines, y_lines), lists of Line2D (empty if
    that lineout wasn't given).
    """
    style = {'color': 'C3', 'lw': 1, 'alpha': 0.8}
    style.update(style_dict or {})
    xw, yw = ax.get_xlim(), ax.get_ylim()

    def scaled(v):
        v = np.asarray(v, dtype=float)
        if not normalize:
            return v
        m = np.nanmax(np.abs(v)) if np.any(np.isfinite(v)) else np.nan
        return v/m if np.isfinite(m) and m > 0 else np.full_like(v, np.nan)

    x_lines, y_lines = [], []
    if x is not None and x_lo is not None:
        x_lines = ax.plot(x, yw[0] + amp*(yw[1] - yw[0])*scaled(x_lo), **style)
    if y is not None and y_lo is not None:
        y_lines = ax.plot(xw[0] + amp*(xw[1] - xw[0])*scaled(y_lo), y, **style)
    ax.set_xlim(xw)     # plotting must not move the window
    ax.set_ylim(yw)
    return x_lines, y_lines


def apply_axlims(ax, axlims, tick_dx=None):
    """
    Set the view window. axlims: a number L (x and y in [-L, L]),
    (Lx, Ly) (x in [-Lx, Lx], y in [-Ly, Ly]) or [x0, x1, y0, y1].
    tick_dx: tick spacing, ticks at multiples of it inside the window.
    """
    if axlims is None:
        return
    a = np.atleast_1d(np.asarray(axlims, dtype=float)).ravel()
    if a.size == 1:
        xl = yl = (-a[0], a[0])
    elif a.size == 2:
        xl, yl = (-a[0], a[0]), (-a[1], a[1])
    elif a.size == 4:
        xl, yl = (a[0], a[1]), (a[2], a[3])
    else:
        raise ValueError(f"axlims must be L, (Lx, Ly) or [x0, x1, y0, y1], got {axlims!r}")
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    if tick_dx is not None:
        for lims, setter in ((xl, ax.set_xticks), (yl, ax.set_yticks)):
            lo, hi = min(lims), max(lims)
            setter(np.arange(np.ceil(lo/tick_dx)*tick_dx, hi + 0.5*tick_dx, tick_dx))


def display_imshow_lineouts(fig, ax, extent, x=None, y=None, x_lo=None, y_lo=None, amp=0.15, color='k', ls='-'):
    """Legacy wrapper for overlay_lineouts (extent is unused: the current
    view window is used instead)."""
    overlay_lineouts(ax, x, y, x_lo, y_lo, amp=amp,
                                   style_dict={'color': color, 'ls': ls})
    return fig, ax


def add_lineout_to_imshow(ax, x, y, x_lo, y_lo, axlims=None, normalize=True, style_dict=None):
    """
    Legacy wrapper for overlay_lineouts: x_lo along the bottom, y_lo
    along the left of the current view window, peaking at 1/8 of it
    (the old 0.25*axlims in a 2*axlims window). axlims is unused.
    style_dict keys: 'color' (default 'k'), 'linestyle' ('-'), 'alpha' (0.75).
    Returns (x_lines, y_lines).
    """
    style_dict = style_dict or {}
    style = {'color': style_dict.get('color', 'k'),
             'ls': style_dict.get('linestyle', '-'),
             'alpha': style_dict.get('alpha', 0.75)}
    return overlay_lineouts(ax, x, y, x_lo, y_lo, amp=0.125,
                                          normalize=normalize, style_dict=style)


def overlay_fitted_curve(ax, return_dict, curve_function, axlims, style_dict=None):
    fitted_x = curve_function(return_dict['x'], *return_dict['fit_x']['params'])
    fitted_y = curve_function(return_dict['y'], *return_dict['fit_y']['params'])

    x = return_dict['x']
    y = return_dict['y']
    x_lo = return_dict['x_lo']
    y_lo = return_dict['y_lo']

    max_x = np.nanmax([x_lo, fitted_x])
    max_y = np.nanmax([y_lo, fitted_y])
    line_x, line_y = add_lineout_to_imshow(ax, x, y, x_lo/max_x, y_lo/max_y, axlims, normalize=False, style_dict=style_dict)
    line_fitx, line_fity = add_lineout_to_imshow(ax, x, y, fitted_x/max_x, fitted_y/max_y, axlims, normalize=False, style_dict={'color' : 'r'})
    return line_x, line_y


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
    from PyGEECSPlotter.utils.ecs_live_dumps import get_ecs_live_dumps_parameter_value_from_sfilename


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
