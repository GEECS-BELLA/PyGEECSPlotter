"""# 05 - Optical spectra: VIS + NIR

`OpticalSpectrumAnalyzer` processes one spectrometer's spectrum per shot
(background, smoothing, spectral-response correction) and reports the peak
wavelength, counts, and the spectral width and edge wavelengths.
`CombinedVisNIRSpectrum` runs a VIS and a NIR analyzer on the same shot and
stitches them into one spectrum, scaling NIR onto VIS in the overlap.

Both return lineout-style `aux` (`'wl'` / `'wl_lo'`), so the
`OpticalSpectrum*` displayers (lineout-family subclasses with spectral axis
labels) show waterfalls and per-bin means.

The last section shows a scalar-only analyzer: `CameraEnergyAnalyzer`
corrects a camera's summed counts for how much of each shot's spectrum the
camera registers.

Example data: BELLA HPD spectrometers (`SPEC-AA-Hamamastsu`,
`SPEC-TEA-Qred`), 2025-04-21 scan 51, with a laser-off background scan from
2025-04-22.
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt

from PyGEECSPlotter.scan_data_analyzers import ScanDataAnalyzer
from PyGEECSPlotter.analyzers import ImageAnalyzer
from PyGEECSPlotter.analyzers.spectrum_analysis import OpticalSpectrumAnalyzer
from PyGEECSPlotter.analyzers.combined_vis_nir_spectra import CombinedVisNIRSpectrum
from PyGEECSPlotter.analyzers.camera_energy_analysis import CameraEnergyAnalyzer
from PyGEECSPlotter.displayers import (OpticalSpectrumWaterfall, OpticalSpectrumMeanPerBin,
                                      OpticalSpectrumMeanWaterfall, ScalarVsParameter)
from PyGEECSPlotter.utils.plotting import configure_plotting
import PyGEECSPlotter.utils.geecs_cmaps as gcm

configure_plotting()

DATA_ROOT = r'N:\data'
SCAN_DATE, SCAN = (2025, 4, 21), 51          # the scan to analyse
BG_DATE, BG_SCAN = (2025, 4, 22), 12         # a laser-off scan, for backgrounds
VIS_DIAG, NIR_DIAG = 'SPEC-AA-Hamamastsu', 'SPEC-TEA-Qred'

# spectral response curves: 2-column (wavelength, response) text files
CALIB = os.path.join(DATA_ROOT, 'Calibrations')
VIS_RESPONSE = os.path.join(CALIB, 'Hamamatsu HPD', 'Spline Response & Transmission Curves',
                            'HPDHamamatsu_Normalized_SpectralResponse_Spline_0.3pctThresh.txt')
NIR_RESPONSE = os.path.join(CALIB, 'Qred', 'Qred2500Response.txt')

# optional camera-energy section: the camera's spectral response file, or None to skip
CAM_DIAG = 'CAM-HPD-M3Near'
CAM_RESPONSE = None

WRITE_SFILE = False
OUTPUT_SUBDIR = 'examples'
VIEW = dict(display_dict={'cmap': gcm.white_blue_yellow(), 'xlims': (500, 1800)},
            output_subdir=OUTPUT_SUBDIR)


def load_scan(date, scan_number, analyzer):
    year, month, day = date
    scan = ScanDataAnalyzer(experiment_dir=DATA_ROOT, year=year, month=month, day=day, scan=scan_number)
    scan.load_scan_data(analyzer=analyzer)
    return scan


# %% [markdown]
# ## Backgrounds and response curves
#
# `average_background` takes the mean raw spectrum over a scan's shots, the
# laser-off scan here. `load_response` interpolates a response curve onto a
# wavelength axis (`wl_lin`). NaN entries mark wavelengths to ignore; the
# NIR curve's blue end is set to NaN, and the stitching uses that to find the
# usable overlap.

# %%
vis = OpticalSpectrumAnalyzer(VIS_DIAG, '.txt')
nir = OpticalSpectrumAnalyzer(NIR_DIAG, '.txt')
bg_vis = vis.average_background(load_scan(BG_DATE, BG_SCAN, vis))
bg_nir = nir.average_background(load_scan(BG_DATE, BG_SCAN, nir))

vis_wl = np.linspace(bg_vis['Wavelength (nm)'].min(), bg_vis['Wavelength (nm)'].max(), len(bg_vis))
nir_wl = np.linspace(bg_nir['Wavelength (nm)'].min(), bg_nir['Wavelength (nm)'].max(), len(bg_nir))
vis_response = vis.load_response(VIS_RESPONSE, vis_wl)
nir_response = nir.load_response(NIR_RESPONSE, nir_wl, nan_below_index=29)
print('backgrounds:', bg_vis.shape, bg_nir.shape)

# %% [markdown]
# ## One spectrometer
#
# `analyzer_dict` keys: `bg_file` (subtract the `bg` passed in),
# `bg_constant` (subtract the mean in a signal-free window), smoothing
# (`apply_mov_mean_filter` / `apply_median_filter`, `N_filt`), response
# correction (`include_diagnostic_response`, `diagnostic_response_arrays`,
# `wl_lin`), and `calculate_red_blue_shifts` for widths and edge wavelengths
# (shots whose peak is below `threshold_for_shifts` get
# `below_threshold_value`).

# %%
vis.analyzer_dict = {
    'bg_file': True,
    'bg_constant': True, 'bg_constant_low_wl': 200, 'bg_constant_high_wl': 300,
    'apply_mov_mean_filter': True, 'N_filt': 9,
    'include_diagnostic_response': True, 'diagnostic_response_arrays': [vis_response],
    'wl_lin': vis_wl,
    'calculate_red_blue_shifts': True, 'threshold_for_shifts': 400,
}
nir.analyzer_dict = {
    'bg_file': True,
    'bg_constant': True, 'bg_constant_low_wl': 2300, 'bg_constant_high_wl': 2400,
    'apply_mov_mean_filter': True, 'N_filt': 5,
    'include_diagnostic_response': True, 'diagnostic_response_arrays': [nir_response],
    'wl_lin': nir_wl,
}

scan_vis = load_scan(SCAN_DATE, SCAN, vis)
scan_vis.analyze_scan(vis, bg=bg_vis, write_columns_to_sfile=WRITE_SFILE)
scan_vis.display_scan(OpticalSpectrumWaterfall(vis, bg=bg_vis, overlay_y_values=True, **VIEW))
scan_vis.display_scan(ScalarVsParameter(f'{VIS_DIAG} fwhm (nm)', bin_summary='mean',
                                        output_subdir=OUTPUT_SUBDIR))

# %% [markdown]
# ## VIS + NIR combined
#
# `bg` for a multi-diagnostic analyzer is a dict keyed by diagnostic.
# `scale_method='ratio'` scales NIR by sum(VIS) / sum(NIR) over the overlap
# (`'legacy'` reproduces the older notebook). The 2nd-order 800 nm watchdog
# zeroes the NIR spectrum above the check window when that window is dim.

# %%
combined = CombinedVisNIRSpectrum(vis, nir, output_diagnostic='SPEC-Combined', analyzer_dict={
    'remove_2nd_order_800nm': True,
    'check_2nd_order_800_low_wl': 1400, 'check_2nd_order_800_high_wl': 1500,
    'check_2nd_order_800_thresh': 750,
    'scale_method': 'ratio',
    'calculate_red_blue_shifts': True, 'threshold_for_shifts': 400,
    **({'camera_responses': {CAM_DIAG: vis.load_camera_response(CAM_RESPONSE)}} if CAM_RESPONSE else {}),
})
bg_combined = {VIS_DIAG: bg_vis, NIR_DIAG: bg_nir}

scan = load_scan(SCAN_DATE, SCAN, combined)
df = scan.analyze_scan(combined, bg=bg_combined, write_columns_to_sfile=WRITE_SFILE)
print('NIR scale factor, median: %.3g' % df['nir scale'].median())

# drop near-empty shots before the views
scan.filter('SPEC-Combined max counts', low=3000)

scan.display_scan(OpticalSpectrumWaterfall(combined, bg=bg_combined, overlay_y_values=True, **VIEW))
scan.display_scan(OpticalSpectrumMeanPerBin(combined, axes=['wl', 'vis_wl', 'nir_wl'],
                                            bg=bg_combined, **VIEW))
scan.display_scan(OpticalSpectrumMeanWaterfall(combined, bg=bg_combined, overlay_y_values=True, **VIEW))

# %% [markdown]
# ## Camera energy corrected for the spectrum (optional)
#
# A camera behind a filter registers a spectrum-dependent fraction R of the
# beam energy. `camera_responses` (above) adds `'<camera> response ratio'`
# per shot; `CameraEnergyAnalyzer` divides the camera's summed counts C by R
# and normalises to a reference C/R (here the first bin; use a vacuum scan
# for real numbers). It only reads existing sfile columns, so it has no file
# to load.

# %%
if CAM_RESPONSE:
    cam = ImageAnalyzer(CAM_DIAG, '.png', analyzer_dict={'pipeline': [
        ('n_saturated', {'bit_depth': 12}), ('subtract_bg', {}), ('counts', {})]})
    scan.add_file_list_to_scan_data(CAM_DIAG, '.png')
    cam_bg_path = scan.get_bg_file_path(CAM_DIAG)
    scan.analyze_scan(cam, bg=cam.load_data(cam_bg_path) if cam_bg_path else None,
                      write_columns_to_sfile=WRITE_SFILE)

    energy = CameraEnergyAnalyzer(f'{CAM_DIAG} sum_counts', f'SPEC-Combined {CAM_DIAG} response ratio',
                                  output_diagnostic=f'{CAM_DIAG} energy')
    with scan.temporary_filter('temp Bin number', 1, 1):
        energy.reference = energy.reference_from_scan(scan)
    scan.analyze_scan(energy, write_columns_to_sfile=WRITE_SFILE)
    scan.display_scan(ScalarVsParameter(f'{CAM_DIAG} energy relative energy', bin_summary='mean',
                                        output_subdir=OUTPUT_SUBDIR))

# %%
plt.close('all')
