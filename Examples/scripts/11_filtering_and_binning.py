"""# 11 - Filtering and binning in depth

Example 01 introduces filters and bins; this one covers them fully:

- **filters** are named rules kept on the scan. `scan.active_data` is the
  rows every rule keeps, and every analysis and display uses only those.
  Rules are re-checked against the current data, so they stay correct when
  columns are added by an analysis or bins change.
- **bins** are the `temp Bin number` column. It starts as the sfile's
  `Bin #` (one bin per scan step) and `rebin` redefines it from any column.
- **re-analysing a subset** only updates those shots' results.

Example data: the BELLA HPD laser focal scan of 2024-01-31, scan 23
(126 shots, 21 stage positions, camera `CAM-HPD-CCD`).
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

from PyGEECSPlotter.scan_data_analyzers import ScanDataAnalyzer
from PyGEECSPlotter.analyzers import ImageAnalyzer
from PyGEECSPlotter.displayers import ScalarVsParameter, MeanImagePerBin
from PyGEECSPlotter.utils.plotting import configure_plotting

configure_plotting()

DATA_ROOT = r'N:\data'
YEAR, MONTH, DAY, SCAN = 2024, 1, 31, 23
DIAGNOSTIC = 'CAM-HPD-CCD'
COUNTS = 'HPDmodeMaxCounts'        # peak counts written by the online analysis

analyzer = ImageAnalyzer(diagnostic=DIAGNOSTIC, file_ext='.png', analyzer_dict={
    'dx': 4.43, 'dy': 4.43,
    'pipeline': [('subtract_bg', {}), ('roi', {'roi': [80, 485, 143, 461]}),
                 ('threshold', {'low': 16, 'units': 'counts'}), ('counts', {}),
                 ('second_moments', {'r_max': 100})],
})

scan = ScanDataAnalyzer(experiment_dir=DATA_ROOT, year=YEAR, month=MONTH, day=DAY, scan=SCAN)
scan.load_scan_data(analyzer=analyzer)
bg = analyzer.load_data(scan.get_bg_file_path(DIAGNOSTIC, which_scan=3))
STAGE = scan.scan_parameter
print(scan.n_active, 'of', len(scan.data), 'shots selected')
print(scan.filters)

# %% [markdown]
# Loading with an analyzer already adds one rule: shots whose camera file is
# missing are dropped (`files: CAM-HPD-CCD`). It shows in `scan.filters`, and
# `clear_filters()` keeps it unless asked (`keep_file_filters=False`).

# %% [markdown]
# ## Range filters
#
# `filter(column, low, high)` keeps `low <= value <= high`. Either bound may
# be `None` for a one-sided cut, and shots where the column is NaN are
# dropped. Rules combine with AND. Each rule has a name (the column, unless
# you give `name=`); adding a rule with an existing name replaces it.

# %%
scan.filter(COUNTS, low=150)                        # one-sided
scan.filter(STAGE, 210, 245, name='near focus')     # named
print(scan.filters)

scan.filter(COUNTS, low=250)                        # same name: replaces the first cut
print('after tightening the counts cut:', scan.n_active)

# %% [markdown]
# `exclusive=True` keeps the shots *outside* the range instead, e.g. to drop
# a stretch of bad shots.

# %%
scan.filter('Shotnumber', 40, 50, exclusive=True, name='drop shots 40-50')
print(scan.filters)

# %% [markdown]
# ## Other kinds of rule
#
# - `filter_values(column, values)`: keep shots whose value is in a list
#   (`exclusive=True` to drop them instead)
# - `filter_rows(func, name)`: any condition, as a function of the data that
#   returns True for the shots to keep. It's re-evaluated whenever the data
#   changes, so it can use columns an analysis adds later.

# %%
scan.filter_values('Shotnumber', [7, 13, 21], exclusive=True, name='known bad shots')


def counts_above_bin_median(data):
    """Keep shots brighter than the median of their own bin."""
    median = data.groupby('temp Bin number')[COUNTS].transform('median')
    return data[COUNTS] >= median


scan.filter_rows(counts_above_bin_median, name='bright half of each bin')
print(scan.filters)

# %% [markdown]
# ## Removing and restoring
#
# `remove_filter(name)` drops one rule, `clear_filters()` all of them.
# `with scan.filtered():` undoes whatever the block adds or removes when it
# ends, even if it raises an error. `with scan.temporary_filter(col, lo, hi):`
# is the one-rule shortcut.

# %%
scan.remove_filter('bright half of each bin')
before = scan.n_active

with scan.filtered():
    scan.filter_values('Shotnumber', scan.active_data['Shotnumber'].values[::3], name='every 3rd')
    scan.remove_filter('near focus')
    print('inside the block:', scan.n_active, 'shots')
print('after the block :', scan.n_active, 'shots (was', before, ')')

with scan.temporary_filter(STAGE, 226, 228):           # the stage steps by 3 mm
    print('shots at the 227 mm step:', scan.n_active)

scan.clear_filters()
print(scan.filters)

# %% [markdown]
# ## Bins
#
# `scan.bins()` lists the bin numbers among the selected shots;
# `rows_in_bin(b)` and `iter_bins()` give each bin's shots. Pass those rows to
# anything that takes `rows=` to work bin by bin without changing the
# filters, e.g. `mean_std_diagnostic` for each bin's mean image.

# %%
print('bins:', scan.bins().tolist())
for b, rows in scan.iter_bins(scan.bins()[::5]):
    mean_img, std_img = scan.mean_std_diagnostic(analyzer, bg=bg, rows=rows)
    print(f'bin {b:2d}: {len(rows)} shots at {STAGE} = {rows[STAGE].mean():.0f}, '
          f'mean-image peak {np.nanmax(mean_img):.0f}')

centre, spread = scan.compute_bin_summary('mean')     # also 'median', 'std_err', 'max', 'min'
print('per-bin mean', COUNTS, ':', centre[COUNTS].round(0).astype(int).tolist())

# %% [markdown]
# ## Rebinning
#
# `rebin(method, scan_parameter=column, ...)` recomputes `temp Bin number`
# from any column, e.g. the measured stage position instead of the
# commanded step, or a diagnostic value to group shots by brightness.
#
# | method | argument | bins |
# |---|---|---|
# | `unique` | | one per distinct value |
# | `rounding` | `rounding_factor` | values rounded to a multiple, then unique |
# | `zscore` | `z_threshold` | split where sorted values jump |
# | `kmeans` | `n_bins` | 1-D k-means clusters |
# | `edges` | `bin_edges` | explicit edges (outside -> bin 0) |
# | `quantile` | `n_bins` | equal numbers of shots |
# | `width` | `bin_width` or `n_bins` | equal widths |
#
# NaN values get bin 0. `reset_bins()` goes back to the sfile's `Bin #`.

# %%
for method, kwargs in [('rounding', {'rounding_factor': 5}), ('kmeans', {'n_bins': 6}),
                       ('edges', {'bin_edges': [200, 215, 235, 260]})]:
    scan.rebin(method, **kwargs)

scan.rebin('quantile', scan_parameter=COUNTS, n_bins=4)     # brightness quartiles
scan.display_scan(ScalarVsParameter(STAGE, x_col=COUNTS, bin_summary='mean'))
scan.display_scan(MeanImagePerBin(analyzer, bg=bg, ncols=4, label_column=COUNTS, label_fmt='{:.0f}'))

# %% [markdown]
# A filter on `temp Bin number` follows the bins: it is re-checked after
# `rebin`, so it always means "these bin numbers now".

# %%
scan.filter('temp Bin number', 4, 4, name='brightest quartile')
print('brightest quartile:', scan.n_active, 'shots')
scan.reset_bins()
print('same rule after reset_bins (now sfile bin 4):', scan.n_active, 'shots')
scan.clear_filters()

# %% [markdown]
# ## Re-analysing a subset
#
# `analyze_scan` analyses only the selected shots. If a column already
# exists, only those shots' values are replaced; the others keep theirs (in
# memory, and in the sfile if `write_columns_to_sfile=True`). Pass
# `keep_unanalysed=False` to replace the whole column instead, leaving the
# other shots NaN.

# %%
col = f'{DIAGNOSTIC} 2nd moment x'
scan.analyze_scan(analyzer, bg=bg)                          # every shot
print(col, 'filled for', int(scan.data[col].notna().sum()), 'shots')

scan.filter(STAGE, 220, 230)
scan.analyze_scan(analyzer, bg=bg)                          # re-run near focus only
print('after re-running near focus:', int(scan.data[col].notna().sum()), 'shots filled')

scan.analyze_scan(analyzer, bg=bg, keep_unanalysed=False)
print('with keep_unanalysed=False  :', int(scan.data[col].notna().sum()), 'shots filled')
scan.clear_filters()

# %% [markdown]
# ## Older names
#
# Scripts written before these methods keep working:
# `filter_scan_data(col, lo, hi)` (bounds *excluded*, repeated calls
# stack), `filter_scan_data_by_array`, `reset_filters()`, and
# `save_mask()` / `restore_mask()`. New code should use the methods above.

# %%
plt.close('all')
