"""# 03 - Lineouts across a scan

With a `lineouts` step, `ImageAnalyzer` returns each shot's x and y
lineouts through the beam centre in `aux`:
`{'x': x, 'x_lo': x_lineout, 'y': y, 'y_lo': y_lineout}`. The lineout
displayers stack or average these across shots:

- `LineoutWaterfall`: one axis's lineout for every shot, as an image
- `LineoutMeanPerBin`: the mean lineout(s) of each bin, one panel per bin
- `LineoutMeanWaterfall`: each bin's mean lineout, stacked as an image

**Alignment.** With `centroid_method='centroid'` the lineout axis has its
origin at each shot's centroid, so shots are aligned on the beam before
stacking. `round_centroid=True` (the default) keeps the origin on a whole
pixel, so shots line up exactly (`align='snap'`); with
`round_centroid=False` use `align='interp'`.

Example data: the BELLA PL1 axicon line focus (`CAM-PL1-LPMode`),
2024-01-04 scan 3.
"""

# %%
import matplotlib.pyplot as plt

from PyGEECSPlotter.scan_data_analyzers import ScanDataAnalyzer
from PyGEECSPlotter.analyzers import ImageAnalyzer
from PyGEECSPlotter.displayers import (ScalarVsParameter, MeanImagePerBin, LineoutWaterfall,
                                      LineoutMeanPerBin, LineoutMeanWaterfall)
from PyGEECSPlotter.utils.plotting import configure_plotting
import PyGEECSPlotter.utils.geecs_cmaps as gcm

configure_plotting()

DATA_ROOT = r'N:\data'
YEAR, MONTH, DAY, SCAN = 2024, 1, 4, 3
DIAGNOSTIC = 'CAM-PL1-LPMode'
BG_SCAN = 2
WRITE_SFILE = False
OUTPUT_SUBDIR = 'examples'

# %% [markdown]
# ## The analyzer
#
# `auto_roi` crops a fixed-size box around the beam on every shot, so each
# shot's lineout has the same length; `lineouts` with
# `centroid_method='centroid'` puts the axis origin on the beam.

# %%
analyzer = ImageAnalyzer(
    diagnostic=DIAGNOSTIC,
    file_ext='.png',
    analyzer_dict={
        'dx': 0.62,
        'dy': 0.62,
        'round_centroid': True,
        'pipeline': [
            ('n_saturated', {}),
            ('subtract_bg', {}),
            ('auto_roi',    {'size': 200, 'thresh': 0.9, 'presmooth': 0}),
            ('counts',      {}),
            ('lineouts',    {'centroid_method': 'centroid', 'centroid_thresh': 0.85}),
            ('fwhm',        {}),
        ],
    },
    display_dict={'cmap': gcm.white_blue_yellow(), 'vmin': 0, 'vmax': 2400,
                  'cbar_off': True, 'spatial_units': r'\mu m'},
)

scan = ScanDataAnalyzer(experiment_dir=DATA_ROOT, year=YEAR, month=MONTH, day=DAY, scan=SCAN)
scan.load_scan_data(analyzer=analyzer)
bg = analyzer.load_data(scan.get_bg_file_path(DIAGNOSTIC, which_scan=BG_SCAN))
print(scan, '| scan parameter:', scan.scan_parameter)

# %% [markdown]
# ## Analyse the scan

# %%
scan.analyze_scan(analyzer, bg=bg, write_columns_to_sfile=WRITE_SFILE, max_workers=8)
for name in ['fwhm x', 'fwhm y', 'max_counts']:
    scan.display_scan(ScalarVsParameter(f'{DIAGNOSTIC} {name}', bin_summary='mean',
                                        output_subdir=OUTPUT_SUBDIR))

# %% [markdown]
# ## Lineout views
#
# Every view re-runs the analyzer on the shots it needs, so pass the same
# `bg`. `label_column` labels each bin with that column's mean.

# %%
lineouts = dict(bg=bg, align='snap', output_subdir=OUTPUT_SUBDIR)
scan.display_scan(LineoutWaterfall(analyzer, axis='x', display_dict=analyzer.display_dict, **lineouts))
# this scan has ~140 stage positions: show every 10th bin in the panel grid
some_bins = scan.bins()[::10]
scan.display_scan(LineoutMeanPerBin(analyzer, axes=['x', 'y'], bins=some_bins, ncols=5,
                                    label_column=scan.scan_parameter, **lineouts))
scan.display_scan(LineoutMeanWaterfall(analyzer, axis='x', display_dict=analyzer.display_dict,
                                       label_column=scan.scan_parameter, **lineouts))

# %% [markdown]
# For comparison, the mean image of each bin:

# %%
scan.display_scan(MeanImagePerBin(analyzer, bg=bg, bins=some_bins, ncols=5,
                                  label_column=scan.scan_parameter, output_subdir=OUTPUT_SUBDIR))

# %% [markdown]
# **If a waterfall raises `ValueError` naming a shot**, that shot's lineout
# axis differs from the others. Usually the ROI size changed (use `auto_roi`
# or `roi` with a fixed size), or `round_centroid=False` with
# `align='snap'`. The error means the analyzer settings don't suit this
# view; the displayer is behaving as intended.

# %%
plt.close('all')
