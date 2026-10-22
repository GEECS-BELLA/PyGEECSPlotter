# PyGEECSPlotter examples

Worked examples on real BELLA scans, one per common workflow. Each sets
`DATA_ROOT` (and the date / scan) at the top. Change those for your own
data; nothing else is machine-specific.

| | Example | Shows |
|---|---|---|
| 01 | [Scan basics](scripts/01_scan_basics.py) | load an sfile, bins and rebinning, filters, scalar vs parameter with fits, stats tables, correlations |
| 02 | [Camera images](scripts/02_camera_images.py) | the `ImageAnalyzer` step pipeline, one shot then the whole scan, image grids, focal scan summary |
| 03 | [Lineouts](scripts/03_lineouts.py) | lineout waterfalls and per-bin mean lineouts, centroid alignment |
| 04 | [FROG traces](scripts/04_frog_traces.py) | `FrogAnalyzer`, trace waterfalls and per-bin means |
| 05 | [Optical spectra](scripts/05_optical_spectra.py) | `OpticalSpectrumAnalyzer`, backgrounds and response curves, VIS + NIR stitching, camera energy correction |
| 06 | [Several diagnostics, one shot](scripts/06_several_diagnostics_one_shot.py) | `ShotCompare` with scalar / stats panels; the laser alignment overview |
| 07 | [Comparing scans](scripts/07_comparing_scans.py) | multi-scan scalar and lineout overlays |
| 08 | [Batch: new scans of a day](scripts/08_batch_new_scans.py) | run a pipeline on every scan not yet analysed |
| 09 | [Column math](scripts/09_column_math.py) | renaming columns and computed columns at load |
| 10 | [Magspec allE](scripts/10_magspec_alle.py) | magnetic spectrometer analysis and views (needs the private `pw_py_magspec`) |
| 11 | [Filtering and binning](scripts/11_filtering_and_binning.py) | every kind of filter rule, removing / temporary filters, bins and `rebin` methods, re-analysing a subset |

Start with 01 and 02; the rest each add one idea. 11 goes deeper into filters and bins.

## Scripts and notebooks

`scripts/` holds the examples as plain Python with `# %%` cell markers. Run
them whole, or cell by cell in VS Code ("Run Cell") or Spyder. `notebooks/`
holds the same examples as Jupyter notebooks, generated from the scripts
without outputs:

```
python Examples/build_notebooks.py        # rebuild all notebooks
python Examples/build_notebooks.py 05     # just one
```

Edit the script, not the notebook, then rebuild. Don't commit notebooks with
outputs.

## Safe by default

Every example keeps results in memory: `write_columns_to_sfile=False`, and
figures are shown, not saved. Add `save=True` to a `display_scan` call to save
a figure under `analysis/ScanNNN/<output_subdir>/`. Setting
`WRITE_SFILE = True` writes new columns into the shared sfile, so only do it
on purpose.

## legacy/

The previous examples, kept for reference with outputs stripped. Their
import lines are updated, but they predate the current API, so prefer the
numbered examples.
