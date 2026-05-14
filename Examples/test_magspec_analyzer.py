"""
Stand-alone test harness for ``PyGEECSPlotter.magspec.MagSpecAnalyzer``.

Edit the CONFIG block below to point at one of your days, then run::

    python Examples/test_magspec_analyzer.py

The script is structured as a series of numbered STEPs. Each step prints
what it found. If something looks wrong, comment out later steps and
inspect the earlier ones individually — no step depends on a later one.

What to look for:
  * STEP 1: calibration files are discovered and parsed; camera count matches
    your day; ``screen`` values are ``'front'`` or ``'side'``.
  * STEP 2: each camera gets a c2c factor [fC/count] and a vignette matrix
    sized to the analysis ROI.
  * STEP 3: axes / angle maps / uniform momentum windows have the expected
    shapes; ``windows['front'].mmt`` length matches your momentum resolution.
  * STEP 4: backgrounds load or get built; per-camera shape matches the raw
    image shape.
  * STEP 5: a single shot runs through the full pipeline and the
    ``quickESpec/quickEDiv/quickE`` outputs land in
    ``<analysis_dir>/Scan{NNN}/quickE/``. Compare a couple of values against
    matlab for the same shot.
  * STEP 6: end-to-end ``analyze_scan`` over all shots, optionally with
    ``write_columns_to_sfile=True``.
  * STEP 7: one-shot pcolormesh display, saved to PNG for sanity.

Notes on backwards-compat vs matlab:
  - Matlab ``bellaLiveMagspc2.m`` only stitches the **front** window. This
    script stitches every screen present in the camera-calibration file.
    If both front + side cameras are configured, you'll get TWO triplets
    of output files per shot (``..._front_...`` and ``..._side_...``) where
    matlab writes only one (no suffix).
  - ``output_diagnostic='MagSpec'`` (the default) prefixes every result
    column in the sfile with ``MagSpec ``. Pass ``output_diagnostic=None``
    if you want bare matlab-style column names.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make the in-repo PyGEECSPlotter importable when running from a worktree.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyGEECSPlotter.scan_data_analysis import ScanDataAnalyzer
from PyGEECSPlotter.magspec import (
    MagSpecAnalyzer,
    compute_c2c_and_vignette,
    discover_calib_path,
    load_camera_calibration,
    load_lanex_calibration_table,
    load_trajectory_calibration,
)


# ====================================================================
# CONFIG — EDIT THESE TO POINT AT ONE OF YOUR DAYS
# ====================================================================

# Top experiment directory. Same convention as ScanDataAnalyzer's
# ``top_dir``: contains ``scans/`` and ``analysis/`` subdirs.
TOP_DIR = r"N:\data\Y2025\08-Aug\25_0827"

# Day string in YY_MMDD format (matches the matlab convention).
DAY = "25_0827"

# Where the daily calibration files live. Matlab convention is
# ``<TOP_DIR>/camCalib/`` but it varies. The directory must contain at
# least one ``*camCalib.txt``, ``*trjCalib*A0.txt``, and ``*lanexCalib.txt``
# with a YYMMDD prefix.
CALIB_DIR = os.path.join(TOP_DIR, "camCalib")

# Scan numbers
BG_SCAN_NUM = 4           # background scan
DATA_SCAN_NUM = 20        # the scan you actually want to analyse

# Which single shot to spot-check in STEP 5
SPOT_CHECK_SHOT = 1

# Hall-probe column in the sfile. Matlab uses
#   'HALLPROBE-TEA-MAGSPEC Field'  for post-2022 data
#   'BELLA_HallProbe Field'        for pre-2022 data
HALL_PROBE_COLUMN = "HALLPROBE-TEA-MAGSPEC Field"

# If True, run STEP 6 (full scan + sfile column write). False keeps it
# read-only.
RUN_FULL_SCAN = True
WRITE_COLUMNS_TO_SFILE = True

# Where to save the STEP 7 display PNG
DISPLAY_OUT = os.path.join(_HERE, f"magspec_test_Scan{DATA_SCAN_NUM:03d}_shot{SPOT_CHECK_SHOT:03d}.png")


# ====================================================================
# Helpers
# ====================================================================
def _banner(step: int, label: str) -> None:
    print()
    print("=" * 72)
    print(f"STEP {step}: {label}")
    print("=" * 72)


def _derived_paths(top_dir: str, data_scan: int):
    scan_dir = os.path.join(top_dir, "scans")
    analysis_dir = os.path.join(top_dir, "analysis")
    sfile = os.path.join(analysis_dir, f"s{data_scan}.txt")
    return scan_dir, analysis_dir, sfile


# ====================================================================
# STEP 1 — calibration discovery + parsing
# ====================================================================
def step1_calibrations():
    _banner(1, "Calibration discovery + parsing")

    cam_path, cam_date = discover_calib_path(CALIB_DIR, "*camCalib.txt", DAY)
    trj_path, trj_date = discover_calib_path(CALIB_DIR, "*trjCalib*A0.txt", DAY)
    lanex_path, lanex_date = discover_calib_path(CALIB_DIR, "*lanexCalib.txt", DAY)

    print(f"camCalib   : {cam_path}  (date {cam_date})")
    print(f"trjCalib   : {trj_path}  (date {trj_date})")
    print(f"lanexCalib : {lanex_path}  (date {lanex_date})")

    cameras = load_camera_calibration(cam_path)
    print(f"\n{len(cameras)} non-phosphor camera(s) loaded:")
    for cam in cameras:
        print(
            f"  {cam.diagnostic:25s}  screen={cam.screen:<5s}  "
            f"setN={cam.set_n}  fov={cam.fov:.2f}mm  "
            f"ROI=({cam.x_end - cam.x_start + 1}x{cam.y_end - cam.y_start + 1})  "
            f"rot={cam.rot:.2f}°"
        )

    front_traj, side_traj = load_trajectory_calibration(trj_path)
    print(
        f"\nTrajectory: front {len(front_traj.mmt)} pts "
        f"[{front_traj.mmt.min():.1f}, {front_traj.mmt.max():.1f}] MeV/c; "
        f"side {len(side_traj.mmt)} pts "
        f"[{side_traj.mmt.min():.1f}, {side_traj.mmt.max():.1f}] MeV/c"
    )

    lanex = load_lanex_calibration_table(lanex_path)
    print(f"\nLanex table: {len(lanex)} set(s) — set_n values: {sorted(lanex.keys())}")
    return cameras, front_traj, side_traj, lanex


# ====================================================================
# STEP 2 — per-camera c2c + vignette
# ====================================================================
def step2_lanex_corrections(cameras, lanex):
    _banner(2, "Per-camera c2c + vignette matrices")
    for i, cam in enumerate(cameras):
        if cam.set_n not in lanex:
            print(f"  {cam.diagnostic}: NO LANEX ENTRY FOR setN={cam.set_n} — SKIP")
            continue
        c2c, vgnt = compute_c2c_and_vignette(cam, lanex[cam.set_n])
        print(
            f"  [{i}] {cam.diagnostic:25s} c2c={c2c:.4f} fC/count  "
            f"vignette shape={vgnt.shape}  "
            f"vignette range=[{vgnt.min():.3f}, {vgnt.max():.3f}]"
        )


# ====================================================================
# STEP 3 — full analyzer construction + axis sanity
# ====================================================================
def step3_construct_analyzer():
    _banner(3, "Construct MagSpecAnalyzer (loads everything)")
    scan_dir, analysis_dir, _ = _derived_paths(TOP_DIR, DATA_SCAN_NUM)
    mag = MagSpecAnalyzer(
        calib_dir=CALIB_DIR,
        day=DAY,
        scan_dir=scan_dir,
        analysis_dir=analysis_dir,
        bg_scan_num=BG_SCAN_NUM,
        hall_probe_column=HALL_PROBE_COLUMN,
    )

    print(f"Uniform angle: {len(mag.uniform_angle.angle)} pts  "
          f"d_angle={mag.uniform_angle.d_angle:.4f} mrad  "
          f"range=[{mag.uniform_angle.angle[0]:.2f}, "
          f"{mag.uniform_angle.angle[-1]:.2f}] mrad")
    print(f"Screens present: {sorted(mag.windows.keys())}")
    for screen, w in mag.windows.items():
        print(
            f"  '{screen}' window: {len(w.mmt)} pts  dp={w.dp:.3f} MeV/c/T  "
            f"mmt=[{w.mmt[0]:.1f}, {w.mmt[-1]:.1f}] MeV/c/T"
        )

    print("\nPer-camera angle map and bin info:")
    for i, cam in enumerate(mag.cameras):
        ax = mag.x_axes[i]
        ang = mag.angle_maps[i]
        print(
            f"  [{i}] {cam.diagnostic:25s} "
            f"angle_map={ang.angle.shape}  "
            f"bins={ax.bin_counts.size if ax.bin_counts is not None else '—'}"
        )

    return mag


# ====================================================================
# STEP 4 — backgrounds
# ====================================================================
def step4_backgrounds(mag: MagSpecAnalyzer):
    _banner(4, "Background images")
    for cam in mag.cameras:
        bg = mag.backgrounds.get(cam.diagnostic)
        if bg is None:
            print(f"  MISSING bg for {cam.diagnostic}")
            continue
        print(
            f"  {cam.diagnostic:25s} shape={bg.shape}  "
            f"mean={bg.mean():.2f}  max={bg.max():.2f}"
        )


# ====================================================================
# STEP 5 — single-shot end-to-end + outputs
# ====================================================================
def step5_single_shot(mag: MagSpecAnalyzer):
    _banner(5, f"Single-shot pipeline (scan={DATA_SCAN_NUM}, shot={SPOT_CHECK_SHOT})")
    scan_dir, analysis_dir, sfile = _derived_paths(TOP_DIR, DATA_SCAN_NUM)

    # Build the paths dict for one shot manually
    paths = {}
    for cam in mag.cameras:
        paths[cam.diagnostic] = os.path.join(
            scan_dir,
            f"Scan{DATA_SCAN_NUM:03d}",
            cam.diagnostic,
            f"Scan{DATA_SCAN_NUM:03d}_{cam.diagnostic}_{SPOT_CHECK_SHOT:03d}.png",
        )

    # Read the Hall-probe field from the sfile for that shot
    sfile_df = pd.read_csv(sfile, sep="\t")
    row = sfile_df[sfile_df["Shotnumber"] == SPOT_CHECK_SHOT]
    if len(row) == 0:
        print(f"Shot {SPOT_CHECK_SHOT} not found in {sfile}; aborting STEP 5.")
        return None, None
    context = row.iloc[0].to_dict()
    field_T = context.get(HALL_PROBE_COLUMN, 0.0)
    print(f"  Hall-probe field for shot {SPOT_CHECK_SHOT}: {field_T} T")

    # Run the pipeline
    data = mag.load_data(paths)
    print(f"  Loaded {sum(v is not None for v in data.values())}/{len(data)} camera images")

    stitched, results, aux = mag.analyze_data(data, context=context)
    print(f"\n  Stitched windows: {list(stitched.keys())}")
    for screen, img in stitched.items():
        print(
            f"    '{screen}' window: shape={img.shape}  "
            f"sum={img.sum():.2e} aC  max={img.max():.2e} aC"
        )

    print(f"\n  Scalar results ({len(results)} entries):")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"    {key:50s} = {value:.6g}")
        else:
            print(f"    {key:50s} = {value}")

    # Save the backwards-compatible outputs
    shot_analysis_dir = os.path.join(analysis_dir, f"Scan{DATA_SCAN_NUM:03d}")
    os.makedirs(shot_analysis_dir, exist_ok=True)
    mag.write_analyzed_data(
        stitched, shot_analysis_dir, DATA_SCAN_NUM, SPOT_CHECK_SHOT, context=context,
    )
    print(f"\n  Wrote quickE outputs under {os.path.join(shot_analysis_dir, 'quickE')}")
    return stitched, aux


# ====================================================================
# STEP 6 — full scan analyze_scan
# ====================================================================
def step6_full_scan(mag: MagSpecAnalyzer):
    if not RUN_FULL_SCAN:
        print("\n[STEP 6 skipped — RUN_FULL_SCAN=False]")
        return
    _banner(6, f"Full analyze_scan over scan {DATA_SCAN_NUM}")
    _, _, sfile = _derived_paths(TOP_DIR, DATA_SCAN_NUM)

    scan = ScanDataAnalyzer(sfilename=sfile)
    scan.load_scan_data(analyzer=mag)
    print(f"  Loaded sfile with {len(scan.data)} rows; "
          f"{int(scan._mask.sum())} active after file-existence mask.")

    add_columns_df = scan.analyze_scan(
        mag,
        display_data=False,
        write_columns_to_sfile=WRITE_COLUMNS_TO_SFILE,
        write_analyzed=True,
    )
    print(f"  add_columns_df shape: "
          f"{add_columns_df.shape if add_columns_df is not None else None}")
    if add_columns_df is not None and len(add_columns_df) > 0:
        print(f"  Result columns: {list(add_columns_df.columns)}")


# ====================================================================
# STEP 7 — display one shot
# ====================================================================
def step7_display(mag: MagSpecAnalyzer, stitched, aux):
    if stitched is None:
        print("\n[STEP 7 skipped — no stitched data from STEP 5]")
        return
    _banner(7, "Display the spot-check shot")
    fig, _ = mag.display_data(
        stitched,
        return_dict=aux,
        title=f"Scan{DATA_SCAN_NUM:03d} shot {SPOT_CHECK_SHOT:03d}",
    )
    fig.savefig(DISPLAY_OUT, dpi=150)
    print(f"  Saved display: {DISPLAY_OUT}")


# ====================================================================
# Main
# ====================================================================
def main():
    cameras, front_traj, side_traj, lanex = step1_calibrations()
    step2_lanex_corrections(cameras, lanex)
    mag = step3_construct_analyzer()
    step4_backgrounds(mag)
    stitched, aux = step5_single_shot(mag)
    step6_full_scan(mag)
    step7_display(mag, stitched, aux)

    print("\nAll steps complete. If outputs look wrong, comment out later "
          "steps and re-run to isolate.")


if __name__ == "__main__":
    main()
