"""# 08 - Batch: analyse every new scan of a day

Run the same analysis on every scan of an experiment day that hasn't been
done yet, e.g. at the end of a run day, or repeatedly during one.

The pattern:

1. list the day's scans that have an sfile
   (`generate_sfilename_list_from_scans_dir`)
2. treat a scan as done if its output folder
   `analysis/ScanNNN/<OUTPUT_SUBDIR>/` exists. Saving any figure with
   `output_subdir=OUTPUT_SUBDIR` creates it.
3. run `analyse_scan(n)` on the rest, catching failures so one bad scan
   doesn't stop the loop. A failed scan usually has no output folder, so the
   next run retries it.

Replace `analyse_scan` with your own pipeline. The one below is the PW
alignment overview from example 06; any of examples 02–05 fits the same
shape. Set `DRY_RUN = True` to only list what would run.
"""

# %%
import os
import re
import traceback
import matplotlib.pyplot as plt

from PyGEECSPlotter.scan_data_analyzers import ScanDataAnalyzer
from PyGEECSPlotter.analyzers.alignment_analysis import AlignmentReader
from PyGEECSPlotter.displayers import MultiDiagnosticAlignment
from PyGEECSPlotter.utils.navigation_utils import get_top_dir, generate_sfilename_list_from_scans_dir
import PyGEECSPlotter.utils.geecs_cmaps as gcm

DATA_ROOT = r'N:\data'
YEAR, MONTH, DAY = 2026, 10, 8
START_SCAN, END_SCAN = 1, 1e999       # limit the scan range if needed
OUTPUT_SUBDIR = 'output-py-alignment' # also marks a scan as done
DRY_RUN = True

# %% [markdown]
# ## The per-scan pipeline
#
# Everything for one scan, saving with `output_subdir=OUTPUT_SUBDIR`.

# %%
def camera(diagnostic, cmap=gcm.white_blue_yellow(), crosshair=1):
    return {'diagnostic': diagnostic, 'file_ext': '.png', 'target_on': True,
            'crosshair': crosshair, 'cmap': cmap}


CAMERAS_1BL = [camera('CAM-1BL-DM'), camera('CAM-1BL-CompInput'), camera('CAM-1BL-CompWindow'),
               camera('CAM-1BL-M1'), camera('CAM-1BL-M2Near'), camera('CAM-1BL-M5'),
               camera('CAM-1BL-Ghost'), camera('CAM-HPD-CCD'),
               camera('CAM-HPD-HolyWedge', 'RdBu', crosshair=2)]


def analyse_scan(scan_number):
    reader = AlignmentReader(CAMERAS_1BL, alignment_name='1BL')
    scan = ScanDataAnalyzer(experiment_dir=DATA_ROOT, year=YEAR, month=MONTH, day=DAY,
                            scan=scan_number)
    scan.load_scan_data(analyzer=reader, remove_missing_diagnostic_files=False)
    scan.display_scan(MultiDiagnosticAlignment(reader, shot_selector='first',
                                               output_subdir=OUTPUT_SUBDIR), save=True)


# %% [markdown]
# ## Find the scans still to do

# %%
top_dir = get_top_dir(DATA_ROOT, YEAR, MONTH, DAY)
analysis_dir = os.path.join(top_dir, 'analysis')
sfiles = generate_sfilename_list_from_scans_dir(top_dir, start_scan=START_SCAN, end_scan=END_SCAN)
scan_numbers = sorted(int(re.search(r's(\d+)\.txt$', f).group(1)) for f in sfiles)
todo = [n for n in scan_numbers
        if not os.path.isdir(os.path.join(analysis_dir, f'Scan{n:03d}', OUTPUT_SUBDIR))]
print(f'{len(scan_numbers)} scans with sfiles; {len(todo)} not done yet: {todo}')

# %% [markdown]
# ## Run them

# %%
done, failed = [], []
for n in ([] if DRY_RUN else todo):
    print(f'\n=== Scan {n:03d} ===')
    try:
        analyse_scan(n)
        done.append(n)
    except Exception as e:
        failed.append(n)
        print(f'  FAILED: {type(e).__name__}: {e}')
        traceback.print_exc(limit=3)
    finally:
        plt.close('all')

print(f'\n{len(done)} done {done}; {len(failed)} failed {failed}')
