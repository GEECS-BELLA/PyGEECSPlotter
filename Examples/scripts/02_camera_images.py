"""# 02 - Camera images: the ImageAnalyzer pipeline

Analyse every shot's camera image with an `ImageAnalyzer`, write the results
as new scalar columns, and make image views of the scan, here a laser
focal scan (stage position stepped through focus).

An `ImageAnalyzer` runs a list of named **steps** on each image. Each step
reads and updates the image and adds scalars to the shot's results:

| step | does |
|---|---|
| `n_saturated` | count saturated pixels |
| `subtract_bg` / `bg_const` | subtract the background image / a constant |
| `roi` / `auto_roi` | crop to a fixed ROI / a box around the beam |
| `threshold`, `remove_outliers`, `median_filter`, `radial_mask` | clean up |
| `counts` | max / mean / sum counts |
| `lineouts`, `fwhm`, `gauss_fit` | x / y lineouts through the centre, widths, super-Gaussian fits |
| `second_moments`, `encircled_energy` | beam size |
| `misalignment`, `fluence` | offset from a target, fluence calibration |

The maths behind each step is in `PyGEECSPlotter.utils.analysis_2d_utils`
and can be used on plain arrays.

Example data: BELLA HPD focal scan, 2024-01-31 scan 23, camera `CAM-HPD-CCD`.
"""

# %%
import os
import matplotlib.pyplot as plt

from PyGEECSPlotter.scan_data_analyzers import ParallelScanDataAnalyzer
from PyGEECSPlotter.analyzers import ImageAnalyzer
from PyGEECSPlotter.displayers import (MeanImagePerBin, RepresentativeImagePerBin,
                                      SampledImages, ScalarVsParameter, FocalScanDisplayer)
from PyGEECSPlotter.utils.plotting import configure_plotting
import PyGEECSPlotter.utils.geecs_cmaps as gcm

configure_plotting()

DATA_ROOT = r'N:\data'
YEAR, MONTH, DAY, SCAN = 2024, 1, 31, 23
DIAGNOSTIC = 'CAM-HPD-CCD'
BG_SCAN = 3                  # scan whose averaged image is the background
ROI = [80, 485, 143, 461]    # [row_start, row_end, col_start, col_end]

# Results stay in memory unless this is True. Writing changes the shared sfile.
WRITE_SFILE = False
OUTPUT_SUBDIR = 'examples'   # figures saved under analysis/ScanNNN/<this>

# %% [markdown]
# ## The analyzer
#
# `dx`, `dy` (µm per pixel) and `spatial_units` are shared by the steps that
# report sizes; `display_dict` styles `display_data` and the image grids.

# %%
analyzer = ImageAnalyzer(
    diagnostic=DIAGNOSTIC,
    file_ext='.png',
    analyzer_dict={
        'dx': 4.43,
        'dy': 4.43,
        'pipeline': [
            ('n_saturated',      {}),
            ('subtract_bg',      {}),          # the bg passed to analyze_scan / displayers
            ('roi',              {'roi': ROI}),
            ('threshold',        {'low': 16, 'units': 'counts'}),
            ('median_filter',    {'N': 5}),
            ('counts',           {}),
            ('lineouts',         {'centroid_method': 'centroid'}),
            ('fwhm',             {}),
            ('second_moments',   {'r_max': 100}),
            ('encircled_energy', {'fractions': [0.5, 0.865]}),
        ],
    },
    display_dict={
        'cmap': gcm.white_blue_yellow(),
        'vmin': 0,
        'cbar_off': True,
        'spatial_units': r'\mu m',
    },
)

# %% [markdown]
# ## Load the scan and the background
#
# `load_scan_data(analyzer=...)` adds the per-shot file paths and drops shots
# whose file is missing. `get_bg_file_path` finds
# `analysis/ScanNNN<diagnostic>_averaged.png`.

# %%
scan = ParallelScanDataAnalyzer(experiment_dir=DATA_ROOT, year=YEAR, month=MONTH, day=DAY, scan=SCAN)
scan.load_scan_data(analyzer=analyzer)
bg_path = scan.get_bg_file_path(DIAGNOSTIC, which_scan=BG_SCAN)
bg = analyzer.load_data(bg_path)
print(scan, '\nbackground:', os.path.basename(bg_path))

# %% [markdown]
# ## One shot
#
# Try the pipeline on a single image before running the whole scan.

# %%
first = scan.active_data.iloc[0]
image = analyzer.load_data(first[f'{DIAGNOSTIC} file_list'])
data, results, aux = analyzer.analyze_data(image, bg=bg)
print({k: round(float(v), 2) for k, v in results.items() if isinstance(v, (int, float))})
fig, ax = analyzer.display_data(data, title=f'shot {first["Shotnumber"]}')

# %% [markdown]
# ## The whole scan
#
# `ParallelScanDataAnalyzer` runs the shots in threads (`max_workers`); a plain
# `ScanDataAnalyzer` runs them one by one. Each result becomes a column named
# `'<diagnostic> <scalar>'`, e.g. `'CAM-HPD-CCD 2nd moment x'`.

# %%
df = scan.analyze_scan(analyzer, bg=bg, write_columns_to_sfile=WRITE_SFILE, max_workers=8)
print('new columns:', scan.last_merged_columns[:6], '...')


def col(name):
    return f'{DIAGNOSTIC} {name}'


# %% [markdown]
# ## Scalars vs the scan parameter

# %%
for name in ['max_counts', '2nd moment x', '2nd moment y', 'ee radius 50%']:
    scan.display_scan(ScalarVsParameter(col(name), bin_summary='mean', output_subdir=OUTPUT_SUBDIR))

# %% [markdown]
# ## Image views
#
# - `MeanImagePerBin`: the pixel-wise mean image of each bin
# - `RepresentativeImagePerBin`: one real shot per bin (`mode='first' | 'last'
#   | 'max' | 'min'`; max / min rank by `parameter`)
# - `SampledImages`: `n_samples` shots spread evenly across the scan
#
# Image displayers re-run the analyzer on the shots they show, so they take
# the same `bg`.

# %%
grid = dict(bg=bg, ncols=7, label_column=scan.scan_parameter, label_fmt='{:.3g}',
            output_subdir=OUTPUT_SUBDIR)
scan.display_scan(MeanImagePerBin(analyzer, **grid))
scan.display_scan(RepresentativeImagePerBin(analyzer, mode='max', parameter=col('max_counts'), **grid))
scan.display_scan(SampledImages(analyzer, n_samples=14, bg=bg, ncols=7, output_subdir=OUTPUT_SUBDIR))

# %% [markdown]
# ## Focal scan summary
#
# `FocalScanDisplayer` shows the brightest bin and the bins one half-width
# either side, above max counts and the second moments vs stage position.
# Here it ranks by `HPDmodeMaxCounts`, a column the LabVIEW analysis already
# wrote; `col('max_counts')` works too.

# %%
scan.display_scan(FocalScanDisplayer(analyzer, 'HPDmodeMaxCounts', bg=bg, output_subdir=OUTPUT_SUBDIR))

# %% [markdown]
# Add `save=True` to any `display_scan` call to write the figure to
# `analysis/ScanNNN/<OUTPUT_SUBDIR>/`.

# %%
plt.close('all')
