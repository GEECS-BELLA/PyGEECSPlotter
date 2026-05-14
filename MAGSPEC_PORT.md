# MagSpec Port — Planning Document

Port of BELLA magnetic-spectrometer analysis from Kei Nakamura's MATLAB code
(`D:\Server\10_Software\matlab\Kei\Bella\bellaLiveMagspc2.m`) into the
PyGEECSPlotter framework, as the first real-world multi-input pipeline.

This file is a working plan — updated each phase, deleted (or moved to
`docs/`) when the port is complete.

## Goal

Per shot, take `N` magnetic-spectrometer CCD camera images plus the Hall-probe
field (read from the sfile), produce one stitched 2-D array with **energy
[MeV/c × B]** on one axis and **divergence [mrad]** on the other, values in
**pC / MeV / mrad**. Save the standard backwards-compatible outputs
(`Scan###_quickESpec_###.txt`, `Scan###_quickEDiv_###.txt`,
`Scan###_quickE_###.png` integer-aC PNG).

## Scope (phased)

| phase | contents | PR | status |
|---|---|---|---|
| **0** | Planning doc; agree boundaries / open questions. | merged with Phase 1 | ✓ |
| **1** | Calibration loaders + axis/geometry. New package `PyGEECSPlotter/magspec/`. | #11 | ✓ |
| **2** | Per-camera image processing — bg, rotate, ROI, low-pass, vignette, c2c, incident-angle compensation. | #12 | ✓ |
| **3** | Multi-camera stitch + B-field scaling. `MagSpecAnalyzer(MultiDiagnosticAnalyzer)`. Backwards-compatible outputs. | this one | ✓ |
| **4** | ICT / phosphor / live-watcher. Later, not in this porting effort. | later | — |

## Key design decisions

1. **Variable camera count.** The matlab code is hardcoded for 10 magspec cameras
   (3 front + 7 side) + 1 phosphor. The python port treats the camera list as
   data, not as a hardcoded loop. Each camera entry carries an explicit
   `screen` field (`'front'` or `'side'`); window grouping is derived from
   `screen`, not from camera index.

2. **One `MagSpecAnalyzer`, subclass of `MultiDiagnosticAnalyzer`.** Inputs
   list = `[(CAM-TEA-MagSpec*, '.png'), ...]` for the cameras actually in use
   that day. Calibrations loaded once at construction. Hall-probe field read
   per-shot from `context[<hall_probe_column>]`.

3. **Calibrations are plain dataclasses, not nested structs.** `CameraCalibration`,
   `TrajectoryCalibration`, `LanexCalibration`. Loaders return these.

4. **No `*` kwarg-only markers** anywhere (project convention).

5. **Calibration file format unchanged.** Existing tab-separated text files
   (`*camCalib.txt`, `*trjCalib*A0.txt`, `*lanexCalib.txt`) read with pandas /
   numpy. Discovery follows the existing convention: pick the most recent
   file whose date is `<=` the experiment day.

6. **Backwards-compatible per-shot outputs.** Same `Scan###_quickESpec_###.txt`,
   `Scan###_quickEDiv_###.txt`, `Scan###_quickE_###.png` (16-bit integer aC
   with the existing `Comment` field).

7. **Skip ICT + phosphor for now.** Their calibration paths and pipelines
   land in Phase 4. Phase 1-3 must run without them.

## Decisions

User-confirmed:

- **Q1 — screen assignment.** Add a `screen` column to the camera calibration
  `.txt` files (values `'front'` / `'side'`). The python loader reads it
  into `CameraCalibration.screen`. Window grouping uses this field. (Matlab
  files will need a one-time `screen` column added for the python port to
  consume; see open task at the bottom.)
- **Q2 — Hall-probe column.** Parameter on `MagSpecAnalyzer`, default
  `'HALLPROBE-TEA-MAGSPEC Field'`. No auto-detection.
- **Q7 — phosphor entry in camera calib.** Silently ignored on load
  (loader returns only the magspec cameras).

Proposed defaults for the rest (will use these unless overridden):

- **Q3 — output units.** Column header `'Energy (MeV)'` in
  `Scan###_quickESpec_###.txt` (consistent with "MeV on one axis"). Internal
  arrays still in `MeV/c × B [T]` = `MeV/c`; at output we multiply by `c=1`
  (i.e. quote as energy in MeV, equivalent for relativistic e-).
- **Q4 — angle range.** Default `(-1.3, 1.3)` mrad to match latest matlab;
  configurable via `MagSpecAnalyzer(angle_range=(low, high))`.
- **Q5 — lanex calibration path.** Same date-suffix discovery as camera /
  trajectory calibs; matches the day on the data. If a single dated lanex
  file exists, it gets picked; multiple dated files behave the same way
  as camera calibs (newest before experiment day wins).
- **Q6 — calibration directory.** Explicit `calib_dir` parameter on
  `MagSpecAnalyzer`, with a sensible default that mirrors the matlab
  convention (e.g. `<experiment_root>/camCalib` or similar — finalised in
  Phase 1).

## Pre-Phase-1 task (data prep, user action)

Before Phase 1 code can load real data: add a `screen` column to the
existing `*camCalib.txt` files marking each camera as `front` or `side`,
and the phosphor entry as `phosphor` (so the loader can identify and skip
it). I'll spec the exact column name and accepted values in Phase 1.

## Architectural sketch

```
PyGEECSPlotter/magspec/
├── __init__.py            # re-exports
├── calibrations.py        # CameraCalibration, TrajectoryCalibration,
│                          # LanexCalibration dataclasses + loaders
├── geometry.py            # per-camera + uniform-axis derivation
├── image_processing.py    # per-camera bg/rotate/ROI/lowpass/vignette/c2c
├── stitch.py              # multi-camera → uniform momentum & angle
├── magspec_analyzer.py    # MagSpecAnalyzer(MultiDiagnosticAnalyzer)
└── outputs.py             # backwards-compatible .txt + .png writers
```

## Function port map

Magspec-relevant matlab functions. **Common helpers** (`fLogReadV07`,
`fGet3NmbStringV01`, `fAdd2LogFile`, etc.) live in
`D:\Server\10_Software\matlab\Kei\general\generalF\`. Generic ones replaced
by pandas/numpy where possible.

### Phase 1 — calibrations + geometry

| matlab fn | source | python equivalent | notes |
|---|---|---|---|
| `fLogReadV07` | `general/logReader` | `pd.read_csv(sep='\t')` | drop custom parser |
| `fLogClmnFindV01` | `general/logReader` | `df.columns` lookup | trivial |
| `fBellaCalibPathV01` | `bellaF` | `calibrations.discover_calib_path` | date-suffix glob |
| `fBellaCamCalibV02` | `bellaF` | `calibrations.load_camera_calibration` | returns list[CameraCalibration] |
| `fBellaTrjCalibFSV04` | `bellaF` | `calibrations.load_trajectory_calibration` | front/side split |
| `fLanexClbOutV01` | `general` | `calibrations.load_lanex_calibration` | per-setN lookup |
| `fLanexClbV02` | `general` | `calibrations.compute_c2c_and_vignette` | c2c + vignette matrix per camera |
| `fBellaLanexV02` | `bellaF` | wrapper above | orchestrates lanex + camera |
| `fBellaAxisTri` | `bellaF` | `geometry.compute_camera_axis` | per-camera x/y info |
| `fBellaAxisAllV04` | `bellaF` | `geometry.compute_all_axes` | loops cameras; uses `screen` field instead of hardcoded 1-3/4-N split |
| `fBellaAnglMapV01` | `bellaF` | `geometry.compute_angle_maps` | per-camera angle / dAngle maps |
| `fBellaUaYV02` | `bellaF` | `geometry.uniform_angle_axis` | defaults to (-1.3, 1.3) mrad; configurable |
| `fBellaUmXV03` | `bellaF` | `geometry.uniform_momentum_axes` | per-window momentum axis + per-camera binning info |

### Phase 2 — per-camera image processing

| matlab fn | python equivalent | notes |
|---|---|---|
| `f12bitPngOpnV04` | `image_processing.read_12bit_png` | check saturation against 4095 |
| `fTrexTonyBgV01` | `image_processing.subtract_background` | inspect first |
| `fImageRotV02` | `image_processing.rotate_image` | scipy.ndimage.affine_transform — already in `ImageAnalyzer.rotate_around` |
| `fXrayOutV10` | `image_processing.lowpass_filter` | iterative low-pass / xray rejection |
| `fBellaImgV02` | `image_processing.process_camera_image` | top-level per-camera pipeline |
| `fBellaBgV03` | `image_processing.load_or_build_background` | from saved averaged PNG or compute from bg-scan |

### Phase 3 — stitch + analyzer

| matlab fn | python equivalent | notes |
|---|---|---|
| `fBellaUaV01` | `stitch.resample_to_uniform_angle` | per-camera angle resample |
| `fBellaMmtBinV01` | `stitch.bin_in_momentum` | per-camera momentum binning |
| `fBellaUmV01` | `stitch.resample_to_uniform_momentum` | window-level uniform momentum |
| `fBellaUamCmbV03` | `stitch.combine_window` | per-window stitch |
| `fBellaImgDirTr` | replaced by `scan.add_file_list_to_scan_data` per camera (done by `MultiDiagnosticAnalyzer.register_with_scan`) | |
| `fSpotAnalysisV01` | `magspec_analyzer._spot_analysis` | scalar outputs (peak, mean, sum, FWHM, ...) |
| `fBellaETxtSaveV02` | `outputs.write_energy_spectrum_txt` | backwards compatible |
| `fBellaATxtSaveV01` | `outputs.write_angle_distribution_txt` | backwards compatible |
| (integer PNG with comment) | `outputs.write_integer_aC_png` | matches matlab's `imwrite(...,'Comment',...)` |

### Out of scope (Phase 4 or later)

`fBellaICT`, `fBellaPhosSv`, `fGet3NmbStringV01`, `fBellaRootDirV01`,
`fDirPathV01`, `fAdd2LogFile`, `fTxtOutV01`, the live-mode polling loop,
the optional 10-GeV info figures.

## Phase 1 detailed plan

1. **`calibrations.py`** — dataclasses + loaders for camera, trajectory
   (front/side), and lanex calibrations. Tab-separated text input. Returns
   typed objects. Date-suffix file discovery.
2. **`geometry.py`** — per-camera x/y axis derivation (`fBellaAxisTri`
   port), angle maps, uniform-angle Y axis, uniform-momentum X axes per
   window. Windows determined by `screen` field, not camera index.
3. **Unit verification.** Pick a representative day, generate the same
   axis arrays in matlab and python, compare.

No image processing in Phase 1, no analyzer subclass, no stitching.
Just calibration → numbers.
