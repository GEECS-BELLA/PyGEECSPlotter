"""# 04 - FROG traces

`FrogAnalyzer` reads FROG-retrieved pulses (one tab-separated file per shot,
with spectral and temporal amplitude / phase) and reports the pulse duration,
bandwidth, peak power per joule and centroids. Its per-shot output is a dict
of two DataFrames, `{'temporal': ..., 'spectral': ...}`, so the **trace**
displayers apply:

- `TraceWaterfall`: every shot's trace as an image
- `TraceMeanPerBin`: the mean trace per bin (panels, or `overlay=True` for
  one colour-mapped axis)
- `TraceMeanWaterfall`: each bin's mean trace stacked as an image

Shots must be on a common axis to be stacked or averaged, so the analyzer
resamples each one onto `t_grid` / `wl_grid`. Widths and centroids are
computed before resampling, on the retrieval's own grid.

Example data: Thomson FROG (`HTT-FROG-TemporalTrace`), 2025-11-11 scan 22.
"""

# %%
import os
import matplotlib.pyplot as plt

from PyGEECSPlotter.scan_data_analyzers import ScanDataAnalyzer
from PyGEECSPlotter.analyzers.frog_analysis import FrogAnalyzer
from PyGEECSPlotter.displayers import (TraceWaterfall, TraceMeanPerBin, TraceMeanWaterfall,
                                      ScalarVsParameter)
from PyGEECSPlotter.utils.plotting import configure_plotting

configure_plotting()

DATA_ROOT = r'Z:\data\Thomson'
YEAR, MONTH, DAY, SCAN = 2025, 11, 11, 22
DIAGNOSTIC = 'HTT-FROG-TemporalTrace'
WRITE_SFILE = False
OUTPUT_SUBDIR = 'examples'

# %% [markdown]
# ## The analyzer
#
# Grids are `(min, max, n)` or an explicit array. FROG files live under
# `analysis/ScanNNN/<diagnostic>/`; the file lookup falls back there
# automatically.

# %%
analyzer = FrogAnalyzer(
    diagnostic=DIAGNOSTIC,
    file_ext='.txt',
    analyzer_dict={
        'normalise': True,              # each amplitude to peak 1
        'resample': True,               # onto the grids below
        't_grid': (-400, 400, 401),     # fs
        'wl_grid': (770, 870, 401),     # nm
    },
    display_dict={'panel': 'both'},     # 'both' | 'temporal' | 'spectral'
)

scan = ScanDataAnalyzer(experiment_dir=DATA_ROOT, year=YEAR, month=MONTH, day=DAY, scan=SCAN)
scan.load_scan_data(analyzer=analyzer)
print(scan, '| shots with a FROG file:', len(scan.active_data))

# %% [markdown]
# ## One shot

# %%
row = scan.active_data.iloc[len(scan.active_data) // 2]
filename = row[f'{DIAGNOSTIC} file_list']
data, results, aux = analyzer.analyze_data(analyzer.load_data(filename))
print({k: round(float(v), 2) for k, v in results.items()})
fig, ax = analyzer.display_data(data, return_dict=results, title=os.path.basename(filename))

# %% [markdown]
# ## The whole scan

# %%
scan.analyze_scan(analyzer, write_columns_to_sfile=WRITE_SFILE)
for name in ['Temporal FWHM (fs)', 'Spectral FWHM (nm)', 'Peak Power (TW/J)']:
    scan.display_scan(ScalarVsParameter(f'{DIAGNOSTIC} {name}', bin_summary='mean',
                                        output_subdir=OUTPUT_SUBDIR))

# %% [markdown]
# ## Trace views
#
# `component` picks the trace (`'temporal'` or `'spectral'`). Bins are
# labelled with the scan parameter's per-bin mean by default.

# %%
scan.display_scan(TraceWaterfall(analyzer, component='spectral', overlay_fwhm=True,
                                 display_dict={'cmap': 'viridis'}, output_subdir=OUTPUT_SUBDIR))
scan.display_scan(TraceMeanPerBin(analyzer, component='temporal', ncols=6, show_std=True,
                                  output_subdir=OUTPUT_SUBDIR))
scan.display_scan(TraceMeanPerBin(analyzer, component='temporal', overlay=True,
                                  output_subdir=OUTPUT_SUBDIR), suffix='_overlay')
scan.display_scan(TraceMeanWaterfall(analyzer, component='temporal', output_subdir=OUTPUT_SUBDIR))

# %% [markdown]
# `display_scan(..., export=True)` also saves the plotted arrays; each
# displayer keeps them in `.last_export` (e.g. a waterfall's `'stack'`,
# `'axis'`, `'shots'`).

# %%
plt.close('all')
