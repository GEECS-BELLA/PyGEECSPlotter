# PyGEECSPlotter

Scan analysis and plotting for GEECS / BELLA experimental data.

A GEECS scan is a table of per-shot scalars (the **sfile**, `analysis/sN.txt`)
plus per-shot diagnostic files (camera images, spectra, traces). PyGEECSPlotter
loads the sfile, filters and bins the shots, runs a diagnostic **analyzer** over
each shot, writes the resulting scalars back as new columns, and makes
scan-level figures with **displayers**.

## Install

```
pip install -e .              # from this directory
pip install -e .[live]        # + watchdog / filelock for live sfile watching
```

or create a conda environment with `conda env create -f environment.yml`.

Optional extras, installed separately:

- **Magnetic spectrometer / e-beam profile** analyzers (`analyzers/magspec/`)
  need the private `pw_py_magspec` package (BellaCenter/PW-py-magspec).
- **`WavefrontAnalyzer`** needs the Imagine Optic WaveKit SDK (`wavekit_py`).
  Point `PYGEECS_WAVEKIT_PATH` at the `wavekit_dlls` folder if it isn't at
  the default shared-drive location.

## Quick start

```python
from PyGEECSPlotter.scan_data_analyzers import ScanDataAnalyzer
from PyGEECSPlotter.analyzers import ImageAnalyzer
from PyGEECSPlotter.displayers import ScalarVsParameter, MeanImagePerBin, SampledImages

a = ImageAnalyzer(diagnostic='CAM-HPD-CCD', file_ext='.png', analyzer_dict={
    'dx': 4.43, 'dy': 4.43,
    'pipeline': [('subtract_bg', {}), ('roi', {'roi': [80, 485, 143, 461]}),
                 ('threshold', {'low': 16, 'units': 'counts'}), ('counts', {}),
                 ('second_moments', {'r_max': 100})],
})

scan = ScanDataAnalyzer(sfilename=r'<data root>\Y2024\01-Jan\24_0131\analysis\s23.txt')
scan.load_scan_data(analyzer=a)
scan.filter('HPD stage position', 200, 260)                # a named rule; data is untouched
scan.rebin('kmeans', n_bins=8)                              # or unique / rounding / quantile ...

bg = a.load_data(r'<data root>\...\analysis\Scan003CAM-HPD-CCD_averaged.png')
scan.analyze_scan(a, bg=bg, write_columns_to_sfile=False)  # True writes to the real sfile

scan.display_scan(ScalarVsParameter('CAM-HPD-CCD max_counts', fit='gaussian', fit_on='bins'), save=True)
scan.display_scan(MeanImagePerBin(a, bg=bg), save=True)
scan.display_scan(SampledImages(a, n_samples=12))
```

`write_columns_to_sfile=True` changes shared experimental data. Leave it off
while exploring.

More worked examples (camera images, lineouts, FROG, spectra, multi-diagnostic
and multi-scan views, batch runs, magspec) are in [Examples/](Examples/README.md),
as scripts and notebooks.

## How it fits together

```
ScanDataAnalyzer  ──loads──▶ sfile (DataFrame), filters (mask), bins
      │
      ├─ analyze_scan(analyzer) ──▶ per shot: load_data → analyze_data → (data, results, aux)
      │                              results → new sfile columns
      └─ display_scan(displayer) ──▶ scan-level figure, re-running the analyzer as needed
```

Every analyzer implements the same per-shot contract:

```python
load_data(filename) -> data
analyze_data(data, bg=None, context=None) -> (data, results, aux)   # always a 3-tuple
display_data(data, return_dict=None, title=None, fig=None, ax=None) -> (fig, ax)
write_analyzed_data(data, analysis_dir, scan, shot_num, context=None)
```

`results` is a flat dict of scalars, merged into the sfile. The shape of `aux`
(or of `data`) decides which displayers apply: a 2-D array → image displayers,
`{'x': coords, 'x_lo': values, ...}` → lineout displayers, a dict of
DataFrames → trace displayers.

## Package layout

```
PyGEECSPlotter/
  scan_data_analyzers/   ScanDataAnalyzer, ParallelScanDataAnalyzer, DataDirAnalyzer,
                         live_analysis (watch for new sfiles), binning
  analyzers/             per-shot analyzers and readers (below)
    magspec/             MagSpecAllEAnalyzer / Reader, EBeamProfileAnalyzer (need pw_py_magspec)
  displayers/            scan-level figures (below)
    scalars/ lineouts/ images/ hybrid/ magspec/
  utils/
    analysis_2d_utils    image maths: ROI, threshold, centroids, moments, encircled energy, lineouts
    analysis_1d_utils    lineout / trace / spectrum maths: widths, centroids, resampling, spectral edges
    fitting              curve fits (presets, polynomial, linear regression, custom)
    navigation_utils     GEECS directory layout: top dirs, sfiles, scan / analysis dirs
    io_utils             controls JSON, lineout text files, scaled 16-bit PNGs
    ecs_live_dumps       ECS live dump files
    sfile_utils          masterlog helpers
    ni_imread            12-bit / IMAQ PNG reader
    plotting, geecs_cmaps, pix_cmaps, widgets
```

### Analyzers

| Module | Class | What it does |
|---|---|---|
| `diagnostic_analyzer` | `DiagnosticAnalyzer` | Base class: the per-shot contract |
| `multi_diagnostic_analyzer` | `MultiDiagnosticAnalyzer` | Base for analyzers needing several diagnostics per shot |
| `data_reader` | `DataReader` | Load only (`.png`, `.txt`), no analysis |
| `image_analysis` | `ImageAnalyzer`, `ScaledImageAnalyzer` | Configurable image pipeline (bg, ROI, threshold, filters, centroids, moments, lineouts, fits, fluence) |
| `column_math_analysis` | `ColumnMathAnalyzer` | New sfile columns from existing ones |
| `shot_compare_analysis` | `ShotCompareReader` | Several analyzers on the same shot |
| `spectrum_analysis` | `OpticalSpectrumAnalyzer` | 1-D optical spectra |
| `combined_vis_nir_spectra` | `CombinedVisNIRSpectrum` | VIS + NIR spectrometers stitched |
| `frog_analysis` | `FrogAnalyzer` | FROG-retrieved temporal / spectral traces |
| `wavefront_analysis` | `WavefrontAnalyzer` | HASO / Phasics wavefronts (WaveKit) |
| `camera_energy_analysis` | `CameraEnergyAnalyzer` | Camera energy, corrected for spectral response |
| `alignment_analysis` | `AlignmentReader` | Raw frames of alignment cameras + ECS targets |
| `magspec/*` | `MagSpecAllEAnalyzer`, `MagSpecAllEReader`, `EBeamProfileAnalyzer` | Magnetic spectrometer allE, e-beam profile |

The first six are the generic framework; the rest are diagnostic-specific.
For slow analyzers, save outputs once (`write_analyzed=True`) and point
displayers at a reader (a `DataReader` subclass, e.g. `MagSpecAllEReader`).

### Displayers

All importable from `PyGEECSPlotter.displayers`.

| Folder | Displayers |
|---|---|
| `scalars/` | `ScalarVsParameter` (with fits), `CorrelationHeatmap`, `ScanStatsTable` |
| `lineouts/` | `LineoutWaterfall`, `LineoutMeanPerBin`, `LineoutMeanWaterfall`, `TraceWaterfall`, `TraceMeanPerBin`, `TraceMeanWaterfall`, `OpticalSpectrum{Waterfall,MeanPerBin,MeanWaterfall}` |
| `images/` | `MeanImagePerBin`, `SampledImages`, `RepresentativeImagePerBin`, `PeakWidthImages` (bases: `ImageGridDisplayer`, `ShotSelectionGrid`) |
| `hybrid/` | `ShotCompare`, `MultiDiagnosticAlignment`, `FocalScanDisplayer`, `MultiScanScalarVsParameter`, `MultiScanLineoutMean` |
| `magspec/` | `MagSpecAllEWaterfall`, `MagSpecAllESpectrumPerBin`, `MagSpecAllESpectrumMeanWaterfall`, `MagSpecAllESampledShots`, `MagSpecAllERepresentativePerBin` |

Single-scan displayers run with `scan.display_scan(d, save=True)`; multi-scan
ones with `display_scans(d, [scan_a, scan_b], save=True)`.

## Data layout

```
<root>/Y2026/05-May/26_0514/
  ├── analysis/
  │   ├── sN.txt                       # sfile
  │   ├── ScanNNN/<diag>_analyzed/     # analyzer outputs
  │   └── ScanNNN_<diag>_averaged.png  # backgrounds
  └── scans/ScanNNN/
      ├── <diagnostic>/ScanNNN_<diagnostic>_SSS.png
      ├── ScanDataScanNNN.txt
      └── ScanInfoScanNNN.ini
```

Data roots are always passed in as parameters; nothing in the library assumes
a drive letter.

## Adding an analyzer or displayer

- **Analyzer**: subclass `ImageAnalyzer` for 2-D images (or add a pipeline step:
  a `_step_<name>` method plus an entry in `STEPS`), `DataReader` for
  load-only views, `MultiDiagnosticAnalyzer` when one shot needs several
  diagnostics, else `DiagnosticAnalyzer`. Return `(data, results, aux)`, and
  choose `aux`'s shape so an existing displayer family fits. Put reusable
  maths in `utils/analysis_1d_utils.py` / `analysis_2d_utils.py`.
- **Displayer**: subclass the family member closest to what you need (e.g.
  `LineoutWaterfall`, `ShotSelectionGrid`) in the matching `displayers/`
  folder, then import it and add it to `__all__` in `displayers/__init__.py`.

## Upgrading from 0.0.x

Modules moved into subpackages in 0.1.0. The old flat import paths
(`PyGEECSPlotter.scan_data_analysis`, `PyGEECSPlotter.image_analysis`,
`PyGEECSPlotter.displayers.optical_spectrum`, `from PyGEECSPlotter.utils import
get_lineout_width`, ...) still work but emit a `DeprecationWarning` naming the
new location; `PyGEECSPlotter/_legacy.py` has the full map. Analyzer static
methods (`ImageAnalyzer.centroid`, ...) are kept as aliases of the functions in
`utils/`.

One behaviour change: `OpticalSpectrumAnalyzer`'s spectral width columns
(`fwhm (nm)`, `width at ... (nm)`, `lambda_b/lambda_r ...`) had their edges
swapped, so widths were negative; they are now correct.
