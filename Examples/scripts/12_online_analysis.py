"""# 12 - Online analysis during a run

Two watchers run analysis while data is being taken.

**`ScanWatcher`** waits for each scan to finish (its sfile `analysis/sN.txt`
appears) and then runs a list of jobs on it. Each job:

1. runs `analyze_scan` (in threads, unless the analyzer is not thread-safe)
2. merges the results into the sfile, and optionally into
   `scans/ScanNNN/ScanDataScanNNN.txt`, while holding a lock file
3. runs its displayers and saves each figure to
   `<day>/analysis/ScanNNN/online/<diagnostic>/`

**`LiveShotWatcher`** runs one analyzer *while the scan is running*. A
shot is analysed as soon as its diagnostic file(s) and its row in the
scan's `.tdms` are both complete. The results are appended to
`analysis/ScanNNN/online/<name>_live.txt`, so you can watch them build up.
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
MODE = 'scan'                # 'scan': ScanWatcher; 'tags': with scan tags + day trackers;
                             # 'shots': LiveShotWatcher; 'windows': with the live GUI + trigger
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
# ## Scan tags and day trackers
#
# Hashtags in a scan's info text (the start info typed in GEECS, saved in
# `ScanInfoScanNNN.ini`, e.g. `focus scan #focus`) choose extra jobs for that
# scan. A `ScanClassifier` collects:
#
# - the default jobs, run on every scan;
# - each `Tag`'s extra jobs. A tag can also change the default jobs for that
#   scan (`modify=`, with `tweak_job`) or replace them (`replace_default=True`);
# - day trackers, updated in `analysis/today/` after every scan.
#
# Keep one classifier per experiment in its own config file. Tags missing
# from the config are reported in the log, which catches typos.
#
# - **`DisplayJob`** draws figures from the columns the default jobs have just
#   added, without analysing again.
# - **`AveragedBackground`** writes the scan's mean image as
#   `analysis/ScanNNN<diag>_averaged.png`, which later `use_bg` scans pick up.
# - **`DayImages`** shows one shot per scan (first, max or median of a column)
#   with a shared colour scale, to catch drift.
# - **`DayScalars`** plots per-scan mean ± std of chosen columns.
# - **`ScanSheet(columns=[...])`** keeps the day's scan sheet,
#   `analysis/today/scan_sheet.txt` (tab-delimited, like an sfile). It has one row per
#   scan: date, scan, shots, scan parameter and range, scan info, tags, and
#   `'<column> MEAN'` for each column (`stats=` adds STD, MEDIAN, ...).
#   `make_scan_sheet(DATA_ROOT, '2026-11-03', '2026-11-04', columns=[...])` builds the
#   same sheet offline for a range of days.
#
# ## Provenance
#
# Every result written to the sfile gets a record beside its summary,
# `analysis/ScanNNN/Scan003_<diag>_online_provenance.json`. It holds:
#
# - when and where the analysis ran;
# - the analyzer's class and `analyzer_dict`;
# - its settings: plain-type attributes, so constructor arguments such as a
#   column name or a background file are included;
# - the version and git commit of PyGEECSPlotter, and of the analyzer's
#   package when it is a git checkout;
# - the scan's tags;
# - the config file's path and SHA-256 (`ScanClassifier(config_file=...)`).
#
# With these you can later tell which settings and which code produced a
# number. Offline `analyze_scan(write_columns_to_sfile=True)` writes the same
# record.

# %%
from PyGEECSPlotter.scan_data_analyzers import (ScanClassifier, Tag, DisplayJob,
                                                AveragedBackground, tweak_job)
from PyGEECSPlotter.displayers import FocalScanDisplayer, DayImages, DayScalars, ScanSheet

classifier = ScanClassifier(
    config_file=globals().get('__file__'),   # path + hash go into provenance (None in a notebook)
    jobs=jobs,
    tags=[
        Tag('focus', jobs=[DisplayJob([FocalScanDisplayer(hpd, 'CAM-HPD-CCD max_counts')],
                                      name='CAM-HPD-CCD_focus')]),
        Tag('nobg', modify=lambda js: [tweak_job(j, use_bg=False) if isinstance(j, LiveJob) else j
                                       for j in js]),
        Tag('bg', jobs=[AveragedBackground([('CAM-HPD-CCD', '.png')])], replace_default=True),
    ],
    day=[
        DayImages(hpd, select='max', column='CAM-HPD-CCD max_counts'),
        DayScalars(['CAM-HPD-CCD x0', 'CAM-HPD-CCD y0', 'CAM-HPD-CCD max_counts'], name='HPD'),
        ScanSheet(columns=['Amplifier 3 [J]', 'CAM-HPD-CCD max_counts']),
    ],
)

if MODE == 'tags':
    ScanWatcher(experiment_dir=DATA_ROOT, classifier=classifier, write=WRITE).start()

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
# shots that are already in it. Started mid-scan, it picks up the scan that
# is running.
#
# **Timing.** A shot's file counts as written once its size has not changed
# for `file_settle_s`. A scan has ended when its `sNinfo.txt` exists (GEECS
# writes it after the sfile), or after `settle_s` with no info file. A scan
# that never gets an sfile (aborted) is given up `abandon_grace_s` after the
# next scan starts; its `_live.txt` is kept but nothing is merged.
#
# **Errors.** A shot whose analysis fails is tried again once at the end of
# the scan. Other errors, such as a network drive dropping out, are printed
# and the watcher carries on.
#
# **Several diagnostics.** A `MultiDiagnosticAnalyzer` job waits for every
# input's file. Its result columns are prefixed with its
# `output_diagnostic`, and with `use_bg` each input gets its own background.
# By default, a shot missing one of the inputs is skipped, as `analyze_scan`
# does. With `analyze_partial=True` it is analysed without that input (its
# data is None, so the analyzer must handle it) once a later shot has every
# input, or after `missing_input_timeout_s`.

# %%
if MODE == 'shots':
    live = LiveShotWatcher(experiment_dir=DATA_ROOT, job=jobs[1], write=WRITE,
                           required_columns=[], context_timeout_s=30)
    live.start()                 # Ctrl+C to stop

# %% [markdown]
# ## The live GUI
#
# `show_shots=True` opens the live GUI, a Qt window in its own process, so
# drawing never slows the analysis. Each shot is sent as soon as it is
# analysed, before its files are written. The window shows:
#
# - **left:** the latest shot's images (one panel per camera of a
#   multi-diagnostic job) and its 1-D view (a lineout or spectrum)
# - **right:** trends of the results and of the `.tdms` device scalars,
#   against shot or the scan parameter, and a waterfall of the 1-D views
# - **controls** for the x range, color scales, colormap, log scale,
#   waterfall length, the trends, and pause. They are remembered between runs.
#
# What a shot shows comes from the analyzer's `live_payload()`. The default
# covers 2-D images and lineout-style `aux`; an analyzer can override it, as
# the magspec analyzer does. `gui_options` sets labels, default trends and so
# on. Without a `job`, the window shows the `.tdms` scalars alone.
# `live_view='pyplot'` gives the older in-process pyplot window instead, and
# `live_displayers` still draws scan displayers in pyplot windows.
#
# **Master-control trigger.** GEECS master control multicasts a message per
# shot naming its files, and one when the scan ends. With `trigger` (the
# experiment's multicast group and port) and `trigger_root` (the root path
# it sends), all from the experiment's GEECS database, the watcher checks
# each shot's files by name as soon as the message arrives, and ends the
# scan on `ScanEnd`. Shots whose message is lost are still found by polling.
#
# **End of scan.** `end_process=True` runs the job's displayers (and
# `write_displayed` per-shot figures) in a separate process, so the next
# scan isn't delayed.
#
# **Latency.** At the end of each scan the log shows how long after their
# files were written the shots were shown (median and max).
# `gui_options={'log_path': 'gui.csv'}` also logs each shot's sent, received
# and drawn times in the window.
#
# **Start-up checks.** `start()` first calls the analyzer's `check_ready()`
# (e.g. missing default backgrounds stop it there), and warns if Windows
# caches the network drive's listings (new files then arrive in ~10 s bursts).

# %%
TRIGGER = '234.5.6.17:58456'                                # ControlRoom, from its GEECS database
TRIGGER_ROOT = r'\\192.168.13.6\pdna2\data\ControlRoom'     # the root path master control sends ...
EXPERIMENT_DIR = r'N:\data\ControlRoom'                     # ... and its local mapping

if MODE == 'windows':
    live = LiveShotWatcher(experiment_dir=EXPERIMENT_DIR, job=jobs[1], write=WRITE, show_shots=True,
                           trigger=TRIGGER, trigger_root=TRIGGER_ROOT, end_process=True)
    live.start()
