"""# 12 - Online analysis during a run

Two watchers run analysis while data is being taken.

**`ScanWatcher`** waits for each scan to finish (its sfile `analysis/sN.txt`
appears) and then runs a list of jobs on it. Each job:

1. runs `analyze_scan` (in threads, unless the analyzer is not thread-safe)
2. merges the results into the sfile, and optionally into
   `scans/ScanNNN/ScanDataScanNNN.txt`, while holding a lock file
3. runs its displayers and saves each figure to
   `<day>/online_analysis/ScanNNN/<diagnostic>/`

**`LiveShotWatcher`** runs one analyzer *while the scan is running*. A
shot is analysed as soon as its diagnostic file(s) and its row in the
scan's `.tdms` are both complete. The results are appended to
`online_analysis/ScanNNN/<name>_live.txt`, so you can watch them build up.
When the scan ends:

- shots that never got their `.tdms` context in time are analysed with the
  sfile
- everything is merged into the sfile
- the displayers run

Then the watcher waits for the next scan.

Both watchers poll the folders instead of using file-system events, which
are unreliable on network drives. With `experiment_dir` they follow today's
date, so a run that goes past midnight moves on to the new day folder.

Run this as a script; a notebook cell would block. `WRITE = False` analyses
and plots, but never writes to the sfile.
"""

# %%
import matplotlib
matplotlib.use('Agg')        # no windows: figures are only saved

from PyGEECSPlotter.scan_data_analyzers import ScanWatcher, LiveShotWatcher, LiveJob
from PyGEECSPlotter.analyzers import ImageAnalyzer
from PyGEECSPlotter.analyzers.column_math_analysis import ColumnMathAnalyzer
from PyGEECSPlotter.displayers import ScalarVsParameter, SampledImages, MeanImagePerBin

DATA_ROOT = r'N:\data'
MODE = 'scan'                # 'scan': ScanWatcher; 'shots': LiveShotWatcher
WRITE = False                # True: write results into the sfiles

# %% [markdown]
# ## Analyzers
#
# Any analyzer works. A `ColumnMathAnalyzer` subclass computes new columns from
# each shot's row. An `ImageAnalyzer` analyses a camera.

# %%
class LaserEnergy(ColumnMathAnalyzer):
    """Energy on target from the amplifier energy meter."""
    def analyze_data(self, data, bg=None, context=None):
        return None, {'Laser energy [J]': 0.586 * context['Amplifier 3 [J]']}, {}


hpd = ImageAnalyzer(diagnostic='CAM-HPD-CCD', file_ext='.png', analyzer_dict={
    'pipeline': [('roi', {'roi': [80, 485, 143, 461]}),
                 ('threshold', {'low': 16, 'units': 'counts'}),
                 ('counts', {}),
                 ('lineouts', {'centroid_method': 'centroid'})]})

# %% [markdown]
# ## Jobs
#
# `LiveJob` pairs an analyzer with its displayers and its write options:
#
# - **Defaults:** options left as `None` come from the analyzer's
#   `analyzer_dict` flags (`add_columns_to_masterlog`,
#   `update_scan_data_file`, `use_bg`, `write_analyzed`, `analysis_label`),
#   so existing analyzers behave as they did with the old watcher.
# - **Bare analyzers:** an analyzer passed without a `LiveJob` gets one with
#   the defaults.
# - **Order:** jobs run in list order, so put cheap ones first. A later job
#   can use columns an earlier one added.

# %%
jobs = [
    LaserEnergy('laser energy'),
    LiveJob(hpd, use_bg=True, displayers=[
        ScalarVsParameter('CAM-HPD-CCD sum_counts'),
        SampledImages(hpd, n_samples=6, order_by='CAM-HPD-CCD max_counts', top=True),
        MeanImagePerBin(hpd),
    ]),
]

# %% [markdown]
# ## ScanWatcher: after each scan
#
# `watcher.reanalyze([3, 4])` runs the same jobs on scans already taken.
# `ScanWatcher(top_dir=...)` watches a fixed day folder instead of today's.

# %%
if MODE == 'scan':
    watcher = ScanWatcher(experiment_dir=DATA_ROOT, jobs=jobs, write=WRITE)
    watcher.start()              # Ctrl+C to stop

# %% [markdown]
# ## LiveShotWatcher: during each scan
#
# **Required columns.** A diagnostic file can arrive before its row in the
# `.tdms`, and a row can arrive before every device's value is in it. If the
# analyzer needs a value from the shot's row, e.g. a filter-wheel position,
# list it in `required_columns`. The shot then waits for it, for up to
# `context_timeout_s` seconds; after that it is analysed at the end of the
# scan.
#
# **Backlog.** Shots can arrive faster than they are analysed. They queue up
# and are analysed in order; `watcher.progress()` shows how far behind it is.
#
# **Restarts.** A restarted watcher reads `<name>_live.txt` and skips the
# shots that are already in it.

# %%
if MODE == 'shots':
    live = LiveShotWatcher(experiment_dir=DATA_ROOT, job=jobs[1], write=WRITE,
                           required_columns=[], context_timeout_s=30)
    live.start()                 # Ctrl+C to stop
