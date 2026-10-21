"""# 07 - Comparing scans

`MultiScanDisplayer`s overlay several scans on one figure. Each scan keeps
its own filters, bins and analysis; load and analyse each as usual, then
pass the list to `display_scans`:

- `MultiScanScalarVsParameter`: one scalar vs the scan parameter per scan,
  each with its own fit if requested
- `MultiScanLineoutMean`: each scan's mean lineouts, per bin (matched by
  bin number) or over the whole scan

`label_by` names each scan in the legend: `'scan'` (its number),
`('mean' | 'max' | 'min' | 'median', column)` (a statistic of another
column), a list of labels, or a callable. Figures go to `output_dir`
(default: the first scan's analysis folder) named `Scans018-023_...`.

Example data: two HPD focal scans on 2024-01-31, scan 18 ("end of day
focal scan") and scan 23 (after optimisation).
"""

# %%
import matplotlib.pyplot as plt

from PyGEECSPlotter.scan_data_analyzers import ScanDataAnalyzer
from PyGEECSPlotter.analyzers import ImageAnalyzer
from PyGEECSPlotter.displayers import (MultiScanScalarVsParameter, MultiScanLineoutMean,
                                      display_scans)
from PyGEECSPlotter.utils.plotting import configure_plotting

configure_plotting()

DATA_ROOT = r'N:\data'
YEAR, MONTH, DAY = 2024, 1, 31
SCANS = [18, 23]
DIAGNOSTIC = 'CAM-HPD-CCD'

# %% [markdown]
# ## Load and analyse each scan
#
# A fixed ROI (not `auto_roi`) keeps every shot's lineout on the same axis,
# which the lineout comparison needs.

# %%
analyzer = ImageAnalyzer(diagnostic=DIAGNOSTIC, file_ext='.png', analyzer_dict={
    'dx': 4.43, 'dy': 4.43,
    'pipeline': [('subtract_bg', {}), ('roi', {'roi': [80, 485, 143, 461]}),
                 ('threshold', {'low': 16, 'units': 'counts'}), ('counts', {}),
                 ('lineouts', {'centroid_method': 'centroid'}), ('second_moments', {'r_max': 100})],
})

scans = []
for n in SCANS:
    scan = ScanDataAnalyzer(experiment_dir=DATA_ROOT, year=YEAR, month=MONTH, day=DAY, scan=n)
    scan.load_scan_data(analyzer=analyzer)
    scans.append(scan)
bg = analyzer.load_data(scans[0].get_bg_file_path(DIAGNOSTIC, which_scan=3))
for scan in scans:
    scan.analyze_scan(analyzer, bg=bg, write_columns_to_sfile=False)
    print(scan.scan, len(scan.active_data), 'shots')

# %% [markdown]
# ## A scalar across scans

# %%
display_scans(MultiScanScalarVsParameter('HPDmodeMaxCounts', fit='gaussian', fit_on='bins'), scans)
display_scans(MultiScanScalarVsParameter(f'{DIAGNOSTIC} 2nd moment x',
                                         label_by=('max', 'HPDmodeMaxCounts')), scans)

# %% [markdown]
# ## Mean lineouts across scans
#
# `per_bin=False` averages each scan over all its shots; `per_bin=True` makes
# one panel per bin number.

# %%
display_scans(MultiScanLineoutMean(analyzer, axes=['x'], per_bin=False, bg=bg), scans)

# %% [markdown]
# Add `save=True` to write the figures.

# %%
plt.close('all')
