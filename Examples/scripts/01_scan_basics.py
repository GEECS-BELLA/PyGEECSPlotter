"""# 01 - Scan basics

Load a scan's sfile, select and re-bin shots, and plot scalar columns against
the scan parameter, with no diagnostic analysis yet.

- **sfile**: `analysis/sN.txt`, one row per shot, loaded as `scan.data`
- **filters** are a mask: `scan.active_data` holds the selected rows; the data
  itself is never changed
- **bins**: per-bin views use the `temp Bin number` column, set from `Bin #`
  at load and changed by `scan.rebin`

Example data: the BELLA HPD laser focal scan of 2024-01-31, scan 23. Change
`DATA_ROOT` / the date / `SCAN` for your own data.
"""

# %%
import matplotlib.pyplot as plt

from PyGEECSPlotter.scan_data_analyzers import ScanDataAnalyzer
from PyGEECSPlotter.displayers import ScalarVsParameter, ScanStatsTable, CorrelationHeatmap
from PyGEECSPlotter.utils.plotting import configure_plotting

configure_plotting()

DATA_ROOT = r'N:\data'                 # the folder holding Y2024\01-Jan\24_0131\...
YEAR, MONTH, DAY, SCAN = 2024, 1, 31, 23
COUNTS = 'HPDmodeMaxCounts'            # an sfile column used below

# %% [markdown]
# ## Load a scan
#
# Give either the experiment root + date + scan number, or the sfile path
# directly (`ScanDataAnalyzer(sfilename=r'...\analysis\s23.txt')`).

# %%
scan = ScanDataAnalyzer(experiment_dir=DATA_ROOT, year=YEAR, month=MONTH, day=DAY, scan=SCAN)
scan.load_scan_data()
print(scan)
print('scan parameter:', scan.scan_parameter)
print('shots:', len(scan.data), '| columns:', len(scan.data.columns))

# %% [markdown]
# ## Bins
#
# A real scan steps a parameter and increments `Bin #` at each step. When the
# steps are uneven, or for a "no scan" that drifted, re-bin on any column:
# `unique`, `rounding` (`rounding_factor=`), `zscore`, `kmeans` (`n_bins=`),
# `edges` (`bin_edges=`), `quantile` (`n_bins=`) or `width` (`bin_width=`).

# %%
print('bins from the sfile:', scan.data['temp Bin number'].nunique())
scan.rebin('kmeans', n_bins=6)
print('after rebin        :', scan.data['temp Bin number'].nunique())
# one row per bin: the centre (mean / median / max / min) and the spread
centre, spread = scan.compute_bin_summary('mean')
print('per-bin mean of', COUNTS, ':', centre[COUNTS].round(0).tolist())
scan.reset_bins()

# %% [markdown]
# ## Filter shots
#
# Filters are named rules kept on the scan; `scan.active_data` is the rows
# every rule keeps. They are re-checked against the current data, so they
# stay correct when columns are added or bins change.
#
# - `filter(column, low, high)`: keep `low <= value <= high` (either bound
#   may be `None`); `exclusive=True` keeps the outside instead
# - `filter_values(column, values)`: keep rows whose value is in a list
# - `remove_filter(name)` / `clear_filters()`; `scan.filters` lists them
#   (shots with missing diagnostic files are a rule too, kept by `clear_filters`)
# - `with scan.filtered(): ...` or `with scan.temporary_filter(...)` undoes
#   whatever the block changed when it ends

# %%
scan.filter(COUNTS, 200, 4096)
scan.filter(scan.scan_parameter, 215, 235, name='near focus')
print(scan.filters)

with scan.filtered():
    scan.filter_values('Shotnumber', scan.active_data['Shotnumber'].values[::2], name='every other')
    print('inside the block:', scan.n_active, 'shots')
print('after the block :', scan.n_active, 'shots')

scan.remove_filter('near focus')
print('one filter left :', scan.n_active, 'of', len(scan.data))

# %% [markdown]
# ## Scalar vs scan parameter, with a fit
#
# Fit presets: `gaussian`, `gaussian_no_offset`, `super_gaussian`,
# `super_gaussian_no_offset`, `lorentzian`, `lorentzian_slope`, `boltzmann`,
# `extreme`; also `'poly'` (`poly_degree=`), `'linear'` (regression with a 95%
# band) and `'custom'` (`fit_func=`, `p0=`). `fit_on='bins'` fits the per-bin
# means instead of every shot.

# %%
svp = ScalarVsParameter(COUNTS, bin_summary='mean', fit='gaussian', fit_on='bins')
scan.display_scan(svp)
print('fit params:', {k: round(float(v), 3) for k, v in zip(svp.last_fit['names'], svp.last_fit['params'])})
print('R^2       :', round(float(svp.last_fit['r2']), 3))

# %% [markdown]
# Plot against another column instead of the scan parameter with `x_col=`;
# `fit='linear'` gives a linear regression with R², r and p.

# %%
scan.display_scan(ScalarVsParameter(COUNTS, x_col='Shotnumber', fit='linear'))

# %% [markdown]
# ## Statistics and correlations

# %%
columns = [COUNTS, 'Shotnumber', scan.scan_parameter]
scan.display_scan(ScanStatsTable(columns))
scan.display_scan(CorrelationHeatmap(columns))

# %% [markdown]
# ## Saving
#
# `display_scan(d, save=True)` writes the figure to the scan's analysis
# folder (`analysis/ScanNNN/`, or a sub-folder given by the displayer's
# `output_subdir=`); `export=True` also writes the plotted data as text.

# %%
plt.close('all')
