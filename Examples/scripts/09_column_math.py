"""# 09 - Renaming columns and column math

Two small config files adjust an sfile as it loads, without touching the
file on disk:

- **search / replace**: rename columns (and the scan parameter), e.g. a
  device name to a physical quantity
- **column math**: new columns computed from existing ones with a formula

Both are lists of dicts saved as JSON with `write_controls_from_python`, and
read by `load_scan_data(search_replace_filename=..., column_math_filename=...)`.
Keep them next to your analysis and reuse them for every scan of a campaign.
(LabVIEW-made files also work, with `parse_from='labview'`.)

Example data: the HPD focal scan, 2024-01-31 scan 23.
"""

# %%
import os
import tempfile

from PyGEECSPlotter.scan_data_analyzers import ScanDataAnalyzer
from PyGEECSPlotter.displayers import ScalarVsParameter
from PyGEECSPlotter.utils.io_utils import write_controls_from_python
from PyGEECSPlotter.utils.plotting import configure_plotting

configure_plotting()

DATA_ROOT = r'N:\data'
YEAR, MONTH, DAY, SCAN = 2024, 1, 31, 23
CONFIG_DIR = tempfile.mkdtemp()       # use your own analysis folder instead

# %% [markdown]
# ## Search / replace
#
# Each pair replaces `Search` with `Replace` in every column name and in the
# scan parameter.

# %%
search_replace = [
    {'Search': 'HPD stage position', 'Replace': 'Focus stage (mm)'},
]
search_replace_file = os.path.join(CONFIG_DIR, 'search_replace.txt')
write_controls_from_python(search_replace_file, search_replace)

# %% [markdown]
# ## Column math
#
# `formula` uses the short names in `variables`, each mapped to a column (after
# search / replace). `^`, `sqrt`, `exp`, `sin`, `cos` and `abs` are available.
# A formula whose input columns are missing is skipped.

# %%
column_math = [
    {'column_name': 'Focus offset (um)',
     'formula': '(a - 225) * 1000',
     'variables': [{'vars': 'a', 'column_name': 'Focus stage (mm)'}]},
    {'column_name': 'Counts per shot (norm.)',
     'formula': 'a / 4095',
     'variables': [{'vars': 'a', 'column_name': 'HPDmodeMaxCounts'}]},
]
column_math_file = os.path.join(CONFIG_DIR, 'column_math.txt')
write_controls_from_python(column_math_file, column_math)

# %% [markdown]
# ## Load with both

# %%
scan = ScanDataAnalyzer(experiment_dir=DATA_ROOT, year=YEAR, month=MONTH, day=DAY, scan=SCAN)
scan.load_scan_data(search_replace_filename=search_replace_file,
                    column_math_filename=column_math_file)
print('scan parameter:', scan.scan_parameter)
print('new columns present:', [c for c in ('Focus offset (um)', 'Counts per shot (norm.)')
                               if c in scan.data.columns])

scan.display_scan(ScalarVsParameter('Counts per shot (norm.)', x_col='Focus offset (um)',
                                    bin_summary='mean'))

# %% [markdown]
# The new columns exist only in memory. To add them to the sfile itself,
# `scan.merge_column_math_to_sfile(column_math_file, parse_from='python')`
# writes them. This changes shared experimental data, so only run it on
# purpose.
