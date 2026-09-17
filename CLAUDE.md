# PyGEECSPlotter — agent handoff & architecture guide

This file is auto-loaded into every Claude Code session. It explains what the
repo is, how it's organised, the conventions you must follow, and where the
work currently stands so you can pick up and extend it. Read it fully before
making changes.

---

## 1. What this is

PyGEECSPlotter analyses data from the **BELLA laser–plasma accelerator** at
LBNL. Data is saved by the GEECS control system in a fixed per-day / per-scan
directory structure. The library loads a scan's **scalar data** (an "sfile"),
lets you filter / bin / post-analyze / correlate / plot it, and lets you
**post-analyze per-shot diagnostic data** (camera images, spectra, …), write
new scalars back to the sfile, and display scan-level summaries.

The author/user is **Alex Picksley** (apicksley@lbl.gov); this is a research
codebase, so match the surrounding style and keep changes practical.

---

## 2. Working conventions (IMPORTANT — follow exactly)

- **PR target is `sbir_post_analysis`, NOT `main`.** Branch off
  `origin/sbir_post_analysis` and open PRs against it. `sbir_post_analysis` is
  the active working line and is a strict superset of `main`.
- **No `*` keyword-only markers in function signatures.** Project-wide
  convention. Write `def f(self, data, bg=None, context=None)`, never
  `def f(self, data, *, bg=None, context=None)`. Applies everywhere — base
  classes, subclasses, helpers.
- **Workflow:** one feature per branch → implement → commit → push → PR.
  Each merged PR bumps the branch. The user merges/deletes branches themselves.
- **Worktrees:** the user's primary checkout is
  `D:\Users\apicksley\Documents\PyGEECSPlotter` (on `sbir_post_analysis`).
  Claude sessions run in linked worktrees under `.claude/worktrees/`. To make
  a change to a branch that's checked out elsewhere (e.g. `sbir_post_analysis`
  in the main checkout), use an **isolated temp worktree**
  (`git worktree add`), commit there, and fast-forward push — never disturb the
  user's main checkout.
- **Commit attribution:** end commit messages with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; end PR
  descriptions with the Claude Code generated line.
- **Testing:** there is no `python` on PATH. Use the conda env interpreter
  `C:\Users\apicksley\.conda\envs\claude_test\python.exe` (Python 3.12) to
  byte-compile and import-test. For headless plots use
  `matplotlib.use("Agg")`. Put scratch scripts in the session scratchpad dir,
  not in the repo.

---

## 3. Domain model & data layout

- **Scan** — a run of the accelerator. Either a **"no scan"** (N consecutive
  shots, `Bin # = 1` for all, `scan_parameter = 'Shotnumber'`) or a **"scan"**
  (an N-shot loop where a control parameter — motor position, gas pressure,
  B-field, … — is stepped; `Bin #` increments each step).
- **Shot** — one acquisition. ~100 typical, up to a few thousand per scan.
- **sfile** — `analysis/sN.txt`, tab-separated table of scalars, one row per
  shot. Loaded into `ScanDataAnalyzer.data` (a pandas DataFrame).
- **Diagnostics** — per-shot files in `scans/ScanNNN/<diagnostic>/`, named
  `ScanNNN_<diagnostic>_SSS.png` (etc.). Analyzed outputs go under
  `analysis/ScanNNN/<diagnostic>_analyzed/`.
- **Directory shape** (per experiment root, e.g. `N:\data`):
  ```
  <root>/Y2026/05-May/26_0514/
    ├── analysis/
    │   ├── sN.txt                      # sfile (scalars)
    │   ├── ScanNNN/                    # per-scan analysis outputs
    │   └── ScanNNN_<diag>_averaged.png # backgrounds, etc.
    └── scans/
        └── ScanNNN/
            ├── <diagnostic>/ScanNNN_<diagnostic>_SSS.png
            ├── ScanDataScanNNN.txt
            └── ScanInfoScanNNN.ini
  ```

---

## 4. Architecture

### 4a. `ScanDataAnalyzer` (`scan_data_analysis.py`)
The hub. Loads the sfile, holds scalar data as `self.data`, exposes:
- **Filtering** via a non-destructive boolean mask: `filter_scan_data`,
  `filter_scan_data_by_array`, `reset_filters`, and `active_data` (the property
  returning only masked-in rows). `save_mask` / `restore_mask` bracket temporary
  filters.
- **Binning** of `temp Bin number` (the column that drives all per-bin work):
  `rebin(method, scan_parameter=None, **kwargs)` and `reset_bins()`. Methods:
  `unique`, `rounding`, `zscore`, `kmeans`, `edges`, `quantile`, `width`
  (see `binning.py`). Also accepts a callable.
- **Per-shot iteration**: `_iter_shots(analyzer, bg=None, show_progress=True,
  rows=None)` — the single shell that loads each shot's file, resolves the
  per-row background, and calls `analyzer.analyze_data`. Pass `rows=` a subset
  of `active_data` to process only selected shots. **Everything per-shot goes
  through this** (analyze_scan, mean_std_diagnostic, aggregate_per_bin, the
  shot-selection displayers).
- **Scan analysis**: `analyze_scan(analyzer, …)` runs the analyzer over all
  active shots, optionally displays/writes per shot, accumulates scalar results
  into a DataFrame, and merges them back into the sfile.
- **Aggregation**: `mean_std_diagnostic`, `aggregate_per_bin`,
  `compute_bin_summary`.
- **Displaying**: `display_scan(displayer, save=False, …)` dispatches to a
  `ScanDisplayer`.

### 4b. `DiagnosticAnalyzer` (`diagnostic_analyzer.py`) — per-shot pipeline base
Contract (unified signatures, no `*`):
- `load_data(filename) -> data`
- `analyze_data(data, bg=None, context=None) -> (data, results, aux)` where
  `results` is a dict of scalars for the sfile and `aux` is auxiliary per-shot
  output (e.g. lineouts).
- `display_data(data, return_dict=None, title=None, fig=None, ax=None) -> (fig, ax)`
- `write_analyzed_data(data, analysis_dir, scan, shot_num, context=None)`
- `register_with_scan(scan, remove_missing_files=True)` — adds the
  `<diagnostic> file_list` / `file_exists` columns; overridden for multi.

Subclasses: `ImageAnalyzer` (`image_analysis.py`, the workhorse — bg, ROI,
lowpass, centroids, super-Gaussian fits, fluence, lineouts),
`ScaledImageAnalyzer`, `ColumnMathAnalyzer` (`column_math_analysis.py`,
computes new sfile columns from existing ones), `OpticalSpectrumAnalyzer`
(`spectrum_analysis.py`).

### 4c. `MultiDiagnosticAnalyzer` (`multi_diagnostic_analyzer.py`)
For per-shot pipelines that consume **several diagnostics at once** (one
diagnostic's analysis can depend on another's). Declares `inputs` as a list of
`(diagnostic, file_ext)` tuples and composes per-input `sub_analyzers`.
`load_data` receives/returns a `{name: data}` dict; `analyze_data` gets dict
`data` and dict `bg`. `_iter_shots` detects multi analyzers and routes bg via
`_resolve_bg_for_multi`. Worked example: `CombinedVisNIRSpectrum`
(`combined_vis_nir_spectra.py`) stitches a VIS + NIR spectrometer per shot.

### 4d. Displayers (`displayers/`) — scan-level views
`ScanDisplayer` (`scan_displayer.py`) is the base: `display(scan, fig, ax)`
returns `(fig, ax)`; `save(fig, scan, …)` writes under the scan's analysis dir.
Used via `scan.display_scan(SomeDisplayer(...), save=True)`.

Concrete displayers:
- `ScalarVsParameter` — scatter/errorbar of a scalar column vs the scan parameter.
- `CorrelationHeatmap` — pairwise correlation of selected columns.
- `MultiDiagnosticAlignment` — one frame per diagnostic side-by-side
  (alignment/beam-quality check); `shot_selector` picks first/last/int/callable.
- **Image-grid family** (share a base — see below).

**Image-grid family** (all render a grid of per-shot images):
```
ImageGridDisplayer(ScanDisplayer)          # image_grid.py — layout, render loop,
│                                          #   use_analyzer_display vs imshow,
│                                          #   suppress_labels (default True),
│                                          #   blank panels, suptitle.
│                                          #   Subclasses implement _collect_panels(scan).
├── MeanImagePerBin                        # mean_image_per_bin.py — pixel-wise mean per bin
└── ShotSelectionGrid(ImageGridDisplayer)  # shot_selection_grid.py — pick one representative
    │                                      #   row per panel, load+show that shot via
    │                                      #   _iter_shots(rows=...). Subclasses do _select_rows(scan).
    ├── SampledImages                      # sampled_images.py — N evenly-spaced shots across the scan
    └── RepresentativeImagePerBin          # representative_image_per_bin.py — one real shot per bin,
                                           #   mode='first'|'last'|'max'|'min' (max/min need a `parameter` col)
```

### 4e. Supporting modules
- `binning.py` — the 7 binning strategies (+ callable) used by `rebin`.
- `navigation_utils.py` — path resolution (top_dir, sfile paths, analysis dirs,
  scan discovery). Imported with `from ... import *` by scan_data_analysis.
- `utils.py` — misc helpers (`super_gaussian`, `merge_dicts_overwrite`,
  `get_lineout_width`, controls parsing, etc.).
- `ni_imread.py` — 12-bit / IMAQ PNG reader (`read_imaq_image`).
- `plotting.py`, `geecs_cmaps.py`, `pix_cmaps.py` — plotting config + colormaps
  (some added via `main`'s upstream fixes).
- `sfile_utils.py` — sfile helpers (from `main`).
- `widgets.py` — ipywidgets selectors (GUI file/date pickers).

---

## 5. Project history (the arc so far)

Delivered on `sbir_post_analysis` (each was its own PR):
1. **Displayers package** — split the old `scan_display.py` into `displayers/`;
   added `MultiDiagnosticAlignment`.
2. **Binning** — `rebin`/`reset_bins` with 7 methods on `ScanDataAnalyzer`.
3. **Analyzer framework** — `DiagnosticAnalyzer` + `MultiDiagnosticAnalyzer`
   base classes; refactored `analyze_scan` around `_iter_shots` (fixed an
   O(n²) concat); retired `ScanDataOverview` (replaced by `aggregate_per_bin` +
   `MeanImagePerBin`); ported `OpticalSpectrumAnalyzer` + `CombinedVisNIRSpectrum`.
4. **MagSpec port** (see §6) — parked on the `magspec` branch, not on `sbir`.
5. **`SampledImages`** displayer + the `_iter_shots(rows=)` subset arg.
6. **Image-grid refactor** — unified `MeanImagePerBin` + `SampledImages` under
   `ImageGridDisplayer`; added `RepresentativeImagePerBin` and shared
   `suppress_labels`.
7. **Merged `main` into `sbir`** — brought upstream fixes (float→int shot/scan
   fix, colormaps, plotting helpers, `sfile_utils.py`, nav utils).

---

## 6. Current state & open threads

- **`sbir_post_analysis`** is the source of truth: all framework work + all of
  `main`. It is 22 ahead of `main`, 0 behind, so `sbir → main` is a clean
  fast-forward whenever the user wants it (they've deferred that).
- **`magspec` branch (parked)** — a full port of the MATLAB BELLA magnetic
  spectrometer analysis (Kei Nakamura's `bellaLiveMagspc2.m`) lives here, NOT
  on `sbir`. It's `PyGEECSPlotter/magspec/` (calibrations, geometry,
  image_processing, backgrounds, stitch, analysis, outputs, `MagSpecAnalyzer`)
  + `MAGSPEC_PORT.md` (the function-by-function port map & decisions) +
  `Examples/test_magspec_analyzer.py` (a 7-step test harness). Status: code
  complete + code-audited against MATLAB, but **not yet validated on real
  data** — the outstanding task is running the test harness on a real day and
  comparing quickE outputs to MATLAB. Requires the day's `*camCalib.txt` to
  have `screen` (front/side/phosphor) and `diagnostic` columns added. Phase 4
  (ICT, phosphor, live mode) was intentionally scoped out.
- **`claude/wip-wavefront-and-utils` branch (WIP)** — the user's in-progress
  work: a `WavefrontAnalyzer(ImageAnalyzer)` using the Imagine Optic Wavekit
  SDK, plus date-range / multi-day scan-selection utilities. Committed to keep
  it safe; predates the `main` merge so it may hit small conflicts in
  `scan_data_analysis.py` / `navigation_utils.py` when continued.
- **Memory:** the user's Claude memory records the two conventions in §2
  (no `*` markers; PR against `sbir_post_analysis`).

---

## 7. Recipes

**Add a new per-shot analyzer:** subclass `ImageAnalyzer` (or
`DiagnosticAnalyzer`), set `diagnostic` / `file_ext`, implement `analyze_data`
returning `(data, results_dict, aux_dict)`. Use it via
`scan.load_scan_data(analyzer=a); scan.analyze_scan(a, write_columns_to_sfile=True)`.

**Add a new grid displayer:** subclass `ShotSelectionGrid` and implement
`_select_rows(scan) -> [(label, positional_index_into_active_data), ...]`; or
subclass `ImageGridDisplayer` and implement `_collect_panels(scan) ->
[(label, data, return_dict), ...]`. Register it in `displayers/__init__.py`.

**Add a new scan-level plot:** subclass `ScanDisplayer`, implement
`display(scan, fig, ax)`; read `scan.active_data`; return `(fig, ax)`.

**Typical usage:**
```python
from PyGEECSPlotter.scan_data_analysis import ScanDataAnalyzer
from PyGEECSPlotter.image_analysis import ImageAnalyzer
from PyGEECSPlotter.displayers import (
    ScalarVsParameter, CorrelationHeatmap, MeanImagePerBin,
    SampledImages, RepresentativeImagePerBin,
)

a = ImageAnalyzer(diagnostic='CAM-HPD-CCD', file_ext='.png', analyzer_dict={...})
scan = ScanDataAnalyzer(sfilename=r'N:\data\...\analysis\s20.txt')
scan.load_scan_data(analyzer=a)
scan.filter_scan_data('some col', lo, hi)
scan.rebin('kmeans', n_bins=8, scan_parameter='pressure_measured')

scan.analyze_scan(a, write_columns_to_sfile=True)          # write scalars back to sfile
scan.display_scan(MeanImagePerBin(a), save=True)           # mean image per bin
scan.display_scan(SampledImages(a, n_samples=12))          # evenly-spaced shots
scan.display_scan(RepresentativeImagePerBin(a, mode='max', parameter='CAM-HPD-CCD max_counts'))
scan.display_scan(CorrelationHeatmap([...]))
```

---

## 8. Gotchas

- `temp Bin number` is the working bin column (initialised from the sfile's
  `Bin #` at load; changed by `rebin`). Everything "per bin" reads it. NaN /
  unbinned rows get bin `0`.
- Filtering is a mask; `self.data` is never mutated. Read `active_data`.
- The per-shot contract returns a **3-tuple** `(data, results, aux)` — older
  MATLAB-ported / legacy code sometimes returned 2 tuples; keep it 3.
- `analyze_data` for single-diagnostic analyzers gets an ndarray/None; for
  `MultiDiagnosticAnalyzer` it gets a dict keyed by diagnostic name.
- Windows paths + a shared network drive (`N:\data`). Don't hardcode absolute
  data paths into library code; take them as parameters.
