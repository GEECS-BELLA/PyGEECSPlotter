# allE Python port: what it is and how it was checked

This package is a Python port of Kei Nakamura's MATLAB magnetic-spectrometer
chain for the triangle chamber. It covers `bellaMagspcTri.m` (the 10 raw
cameras → `highE` / `lowE` / `frontSL`) and `bellaMagspecViewTri.m` →
`fBellaSShotTri.m` (→ the `allE` image, `allESpec` / `allEDiv` and the
`MSAnalysis` scalars). The PyGEECSPlotter wrapper is
`PyGEECSPlotter.magspec_alle_analysis.MagSpecAllEAnalyzer`.

Each Python function names the MATLAB function it ports in its docstring.
The code uses the same calibration files from `Calibrations/ESMCalib`,
picked by the same "newest file dated on or before the experiment day" rule,
and the same `ScanNNN<camera>_averaged.png` backgrounds.

## How it was checked

Reference: 26_0521 Scan023, 1000 shots. The MATLAB outputs are in
`analysis/Scan023/{highE,lowE,frontSL,allE}` and `Scan023_MSAnalysis.txt`.
Each stage was checked on its own, and then the full chain.

| Check | Result |
|---|---|
| Stage 1 on 19 shots across the scan | `highE` / `lowE` images identical to the MATLAB files on most shots |
| Stage 2 fed MATLAB's own `highE` / `lowE` / `frontSL` | `allE` image within 1 count; spectra and scalars agree to ~1e-6 |
| Full chain, 998 shots through `ScanDataAnalyzer` | 13 of 16 scalars agree to a median of 1e-6 or better; charge, energy and all momentum statistics are within 1 % on every shot |

The largest differences over the 998 shots:

- charge: at most 0.2 %
- energy: at most 0.3 %
- mean and std momentum: under 0.1 %
- max momentum: at most 0.7 %
- angle statistics: 1–2 shots differ by 1–5 %
- `xRayBase_fC/mm`: 52 shots differ by more than 1 %
- `charge6to8GeV_pC`: 72 shots differ by more than 1 %

These remaining differences have known causes:

- **x-ray baseline fit.** Python fits `a·exp(-b·|x-d|^c)` with SciPy's
  `curve_fit`, MATLAB with `fit`. On weak-background shots the fit is
  poorly conditioned, and the two optimisers stop at different points on a
  flat cost surface (same cost, different parameters). This causes
  `xRayBase_fC/mm` to differ by more than 1 % on about 5 % of shots.
  `charge6to8GeV_pC` and `pkChrDen6to8GeV_pC` pick up the same differences:
  they are ~1e-4 pC quantities sitting on that baseline.
- **Hot-pixel-filter ties.** A shot or two where one pixel sits exactly at
  the filter threshold goes the other way. This gives ~0.3 % differences in
  one camera's charge. On shots with a weak or double-peaked divergence,
  the same small change can move the angle peak by one bin, which is where
  the few percent-level angle differences come from.
- Two sfile shots have no camera files and are skipped.

## MATLAB behaviours that were reproduced deliberately

These mattered for exact agreement, or they are quirks worth knowing about:

- **`interp1(..., 'cubic')`** on the non-uniform trajectory tables is a
  *not-a-knot spline* in MATLAB, not pchip. Getting this wrong shifted the
  momentum axis by ~0.3 % at the high-energy end.
- **12-bit PNG opening** divides by `2^(16 - sBIT)` and *rounds* (MATLAB
  integer division); it does not truncate. sBIT varies from file to file.
- **Stage 1 → stage 2 file round trip.** Stage 2 reads integer-aC PNGs and
  tables with 5 or 8 significant figures. `emulate_quantization=True` (the
  default) reproduces this; set it to False to keep full precision.
- **`fXrayOutV10`** averages its 4 neighbours in MATLAB's order of
  operations. This matters because exact ties are common on integer images.
- **A missing camera** runs the background frame itself through the chain,
  as MATLAB does.
- **`xSizeF = xSizeF<1 + xSizeF*(xSizeF>=1)`** in `fBellaSShotTri` parses
  as `xSizeF < (1 + …)`, so it is always 1. This is kept as written and
  commented, so `mmtRes_%` matches.

## Settings inferred from the reference outputs

- **`roi = [0.01, 5] GeV/c`.** The repo copy has `[0.1, 14]`, but the
  reference `allESpec` spans 0.0101–4.998 GeV/c. The reference
  `MSAnalysis` also has an extra `allE charge factor_aC/count` column, so
  the reference outputs were made by a slightly newer script than the repo
  copy. `roi` is an `analyzer_dict` option.

## Not ported

These are figure-only parts of `fBellaSShotTri`: the lanex / EBeam-profile
and front-screen panels (`infoE`), `logE` / `lnrE` and the `maxMmt` figure.
The quickE, turbo-ICT and transverse-profile analyses are also not ported.
