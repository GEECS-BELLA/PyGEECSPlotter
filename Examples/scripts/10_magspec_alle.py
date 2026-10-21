"""# 10 - Magnetic spectrometer (allE)

**Needs the private `pw_py_magspec` package** (BellaCenter/PW-py-magspec),
which holds the numerics; PyGEECSPlotter provides the analyzers and
displayers around it.

`MagSpecAllEAnalyzer` loads all ten `CAM-TEA-MagSpec*` cameras for a shot and
produces the full-range allE image (momentum × angle), the electron spectrum
and divergence, and scalars such as `charge_pC` and `peakMomentum_GeV/c`. It
is a port of the MATLAB `allE` analysis, validated against it.

It takes about 0.7 s per shot, so the usual workflow is:

1. run the analyzer **once** with `write_analyzed=True`, which saves each
   shot's image and tables under `analysis/ScanNNN/<output_diagnostic>/`
2. make every scan-level view from the saved files with a
   `MagSpecAllEReader` (about 20 ms per shot)

The sfile needs `HALLPROBE-TEA-MAGSPEC Field` and the e-beam profile angle
columns, which `EBeamProfileAnalyzer` produces (see the end).

Example data: 2026-05-21 scan 23, where the analysis has already been saved
as `py-magspec` (every 10th shot, sfile columns labelled `py_test_v1`).
"""

# %%
import os
import matplotlib.pyplot as plt

from PyGEECSPlotter.scan_data_analyzers import ScanDataAnalyzer
from PyGEECSPlotter.analyzers.magspec.magspec_alle_analysis import MagSpecAllEAnalyzer
from PyGEECSPlotter.analyzers.magspec.magspec_alle_reader import MagSpecAllEReader
from PyGEECSPlotter.displayers import (MagSpecAllEWaterfall, MagSpecAllESpectrumPerBin,
                                      MagSpecAllESpectrumMeanWaterfall, MagSpecAllESampledShots,
                                      MagSpecAllERepresentativePerBin, ScalarVsParameter)
from PyGEECSPlotter.utils.plotting import configure_plotting
import PyGEECSPlotter.utils.geecs_cmaps as gcm

configure_plotting()

DATA_ROOT = r'N:\data'
YEAR, MONTH, DAY, SCAN = 2026, 5, 21, 23
CALIB_DIR = os.path.join(DATA_ROOT, 'Calibrations', 'ESMCalib')
CALIB_DAY = '26_0521'        # picks the calibration files in force on this day
SAVED_DIAG = 'py-magspec'    # where the saved analysis is (or will be written)
LABEL = 'py_test_v1'         # analysis_label used when it was saved

# Writing analysed files and sfile columns changes shared data: off by default.
WRITE_ANALYZED = False
WRITE_SFILE = False
OUTPUT_SUBDIR = 'examples'


def new_scan():
    return ScanDataAnalyzer(experiment_dir=DATA_ROOT, year=YEAR, month=MONTH, day=DAY, scan=SCAN)


# %% [markdown]
# ## Analyse shots
#
# `bg_dir` holds the day's averaged camera backgrounds. Here a handful of
# shots run in memory; for a real run use the whole scan (or
# `ParallelScanDataAnalyzer`) with `WRITE_ANALYZED = True`.

# %%
alle = MagSpecAllEAnalyzer(CALIB_DIR, CALIB_DAY, bg_dir=os.path.join(new_scan().top_dir, 'analysis'),
                           output_diagnostic=SAVED_DIAG)
scan = new_scan()
scan.load_scan_data(analyzer=alle)
scan.filter_scan_data_by_array('Shotnumber', scan.active_data['Shotnumber'].values[::100])
df = scan.analyze_scan(alle, write_analyzed=WRITE_ANALYZED, write_columns_to_sfile=WRITE_SFILE,
                       analysis_label=LABEL if WRITE_SFILE else '')
print(len(df), 'shots analysed; charge (pC):', df['charge_pC'].round(1).tolist())

# %% [markdown]
# ## Views from the saved analysis
#
# `MagSpecAllEReader(load=...)` reads only what a view needs: `'spec'` for
# spectra, `'image'` for allE images, `'div'`, `'infoE'` or `'all'`. Shots
# without saved files are skipped. `diagnostic='allE'` reads the MATLAB
# outputs instead (same file layout).

# %%
spec = MagSpecAllEReader(load='spec', diagnostic=SAVED_DIAG)
image = MagSpecAllEReader(load='image', diagnostic=SAVED_DIAG)
scan = new_scan()
scan.load_scan_data(analyzer=spec)
image.register_with_scan(scan)
print(len(scan.active_data), 'shots with saved outputs')

view = dict(display_dict={'cmap': gcm.white_blue_yellow(), 'xlims': (0.8, 2.5)},
            output_subdir=OUTPUT_SUBDIR)
scan.display_scan(MagSpecAllEWaterfall(spec, **view))
scan.display_scan(MagSpecAllESpectrumMeanWaterfall(spec, **view))
scan.display_scan(MagSpecAllESpectrumPerBin(spec, ncols=4, **view))

# %% [markdown]
# Image views rank shots by an sfile column written by the analysis
# (`'<diagnostic> <scalar> <label>'`).

# %%
charge = f'{SAVED_DIAG} charge_pC {LABEL}'
scan.display_scan(ScalarVsParameter(charge, bin_summary='mean', output_subdir=OUTPUT_SUBDIR))
scan.display_scan(MagSpecAllESampledShots(image, n_samples=8, ncols=4, **view))
scan.display_scan(MagSpecAllERepresentativePerBin(image, mode='max', parameter=charge, ncols=4, **view))

# %% [markdown]
# ## A full run, for reference
#
# ```python
# from PyGEECSPlotter.scan_data_analyzers import ParallelScanDataAnalyzer
# from PyGEECSPlotter.analyzers.magspec.ebeam_profile_analysis import EBeamProfileAnalyzer
#
# bg_dir = os.path.join(top_dir, 'analysis')
# # 1. e-beam profile -> angle columns the allE analysis reads
# ebeam = EBeamProfileAnalyzer(CALIB_DIR, CALIB_DAY, bg_dir=bg_dir, output_diagnostic='py-ebeam')
# scan = ParallelScanDataAnalyzer(sfilename=sfile); scan.load_scan_data(analyzer=ebeam)
# scan.analyze_scan(ebeam, write_analyzed=True, write_columns_to_sfile=True, max_workers=8)
# # 2. allE, saved once
# alle = MagSpecAllEAnalyzer(CALIB_DIR, CALIB_DAY, bg_dir=bg_dir, output_diagnostic='py-magspec',
#                            ebeam_diagnostic='py-ebeam',
#                            analyzer_dict={'ebeam_columns': ('py-ebeam', '')})
# scan = ParallelScanDataAnalyzer(sfilename=sfile); scan.load_scan_data(analyzer=alle)
# scan.analyze_scan(alle, write_analyzed=True, write_columns_to_sfile=True, max_workers=8)
# # 3. views from MagSpecAllEReader, as above
# ```

# %%
plt.close('all')
