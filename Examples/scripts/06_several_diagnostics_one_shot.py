"""# 06 - Several diagnostics for one shot

Two ways to put several cameras of the same shot side by side:

- **`ShotCompare`** runs one analyzer per diagnostic (through a
  `ShotCompareReader`) and draws each one's `display_data` for the chosen
  shot(s). It can add scan-level panels: a `ScalarVsParameter` with the
  shot ringed, and a table of the shot's scalars next to the scan's mean
  and std.
- **`MultiDiagnosticAlignment`** draws one raw frame from each alignment
  camera, with the camera's ECS target crosshair (read by an
  `AlignmentReader`). This is the laser alignment overview.

Both make one figure per selected shot. `shots` / `shot_selector`:
`'first'`, `'last'`, a Shotnumber, a list, `'all'`, or `('sample', n)`.

Example data: the HPD focal scan (2024-01-31 scan 23) for `ShotCompare`, and
the PW 1BL / 2BL alignment cameras of 2026-10-08.
"""

# %%
import os
import matplotlib.pyplot as plt

from PyGEECSPlotter.scan_data_analyzers import ScanDataAnalyzer
from PyGEECSPlotter.analyzers import ImageAnalyzer, ShotCompareReader
from PyGEECSPlotter.analyzers.alignment_analysis import AlignmentReader
from PyGEECSPlotter.displayers import ShotCompare, MultiDiagnosticAlignment, ScalarVsParameter
from PyGEECSPlotter.utils.plotting import configure_plotting
import PyGEECSPlotter.utils.geecs_cmaps as gcm

configure_plotting()

DATA_ROOT = r'N:\data'
OUTPUT_SUBDIR = 'examples'

# %% [markdown]
# ## ShotCompare
#
# One analyzer per diagnostic. The focus camera gets the full pipeline (so
# its scalars exist for the extra panels); the others just show the raw
# frame (`counts` only).

# %%
focus = ImageAnalyzer(diagnostic='CAM-HPD-CCD', file_ext='.png',
                      analyzer_dict={'dx': 4.43, 'dy': 4.43, 'pipeline': [
                          ('subtract_bg', {}), ('roi', {'roi': [80, 485, 143, 461]}),
                          ('threshold', {'low': 16, 'units': 'counts'}), ('counts', {}),
                          ('second_moments', {'r_max': 100})]},
                      display_dict={'cmap': gcm.white_blue_yellow(), 'vmin': 0})
wedge = ImageAnalyzer(diagnostic='CAM-HPD-HolyWedge', file_ext='.png',
                      analyzer_dict={'pipeline': [('counts', {})]}, display_dict={'cmap': 'viridis'})
near = ImageAnalyzer(diagnostic='CAM-HPD-M3Near', file_ext='.png',
                     analyzer_dict={'pipeline': [('counts', {})]}, display_dict={'cmap': 'magma'})

scan = ScanDataAnalyzer(experiment_dir=DATA_ROOT, year=2024, month=1, day=31, scan=23)
scan.load_scan_data(analyzer=focus)
bg = focus.load_data(scan.get_bg_file_path('CAM-HPD-CCD', which_scan=3))
scan.analyze_scan(focus, bg=bg, write_columns_to_sfile=False)     # scalars for the extra panels

reader = ShotCompareReader([focus, wedge, near])
reader.register_with_scan(scan)       # adds the other cameras' file lists

# %% [markdown]
# `bg` is a dict keyed by diagnostic. `extra_panels` takes `ScalarVsParameter`
# displayers and `('stats', [columns])` tables.

# %%
compare = ShotCompare(
    reader, shots=('sample', 2), ncols=3,
    bg={'CAM-HPD-CCD': bg},
    extra_panels=[ScalarVsParameter('CAM-HPD-CCD max_counts', bin_summary='mean'),
                  ('stats', ['CAM-HPD-CCD max_counts', 'CAM-HPD-CCD 2nd moment x',
                             'CAM-HPD-CCD 2nd moment y'])],
    output_subdir=OUTPUT_SUBDIR,
)
scan.display_scan(compare)            # save=True writes one PNG per shot
print(len(compare.figures), 'figures')

# %% [markdown]
# ## Laser alignment overview
#
# Each camera: `diagnostic`, `file_ext`, `cmap`, and `target_on` / `crosshair`
# (1 → the ECS `Target.*` crosshair, 2 → `Target2.*`). Cameras that didn't
# save a frame in a shot leave an empty panel, so load with
# `remove_missing_diagnostic_files=False`.

# %%
def camera(diagnostic, cmap=gcm.white_blue_yellow(), crosshair=1):
    return {'diagnostic': diagnostic, 'file_ext': '.png', 'target_on': True,
            'crosshair': crosshair, 'cmap': cmap}


ALIGNMENTS = {
    '1BL': [camera('CAM-1BL-DM'), camera('CAM-1BL-CompInput'), camera('CAM-1BL-CompWindow'),
            camera('CAM-1BL-M1'), camera('CAM-1BL-M2Near'), camera('CAM-1BL-M5'),
            camera('CAM-1BL-Ghost'), camera('CAM-HPD-CCD'),
            camera('CAM-HPD-HolyWedge', 'RdBu', crosshair=2)],
    '2BL': [camera('CAM-2BL-DM', gcm.enhance_low()), camera('CAM-2BL-M2A', 'viridis'),
            camera('CAM-2BL-M4A', 'jet'), camera('CAM-2BL-F3M5', gcm.enhance_low()),
            camera('CAM-2BL-F3M7', gcm.enhance_low()), camera('CAM-PL1-BesselM1', gcm.enhance_low()),
            camera('CAM-PL1-2-BesselM3', gcm.enhance_low()),
            camera('CAM-PL1-3-BesselM4')],
}

scan = ScanDataAnalyzer(experiment_dir=DATA_ROOT, year=2026, month=10, day=8, scan=48)
scan_dir = os.path.join(scan.top_dir, 'scans', f'Scan{scan.scan:03d}')

for name, cameras in ALIGNMENTS.items():
    if not any(os.path.isdir(os.path.join(scan_dir, c['diagnostic'])) for c in cameras):
        print(f'{name}: none of its cameras saved data in this scan, skipped')
        continue
    reader = AlignmentReader(cameras, alignment_name=name)
    scan.load_scan_data(analyzer=reader, remove_missing_diagnostic_files=False)
    scan.display_scan(MultiDiagnosticAlignment(reader, shot_selector='first',
                                               output_subdir=OUTPUT_SUBDIR))

# %% [markdown]
# To make these figures for every new scan of a day, see example 08.

# %%
plt.close('all')
