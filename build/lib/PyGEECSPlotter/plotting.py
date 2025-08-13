import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib import font_manager

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