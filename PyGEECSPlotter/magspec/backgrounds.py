# Background-image loading / building for the BELLA magspec port.
#
# Ports the matlab `fBellaBgV03`, generalised for variable camera count.
# The matlab version hardcoded 10 magspec + 1 phosphor diagnostic names;
# here we drive everything off the caller's list of CameraCalibrations.

from __future__ import annotations

import glob
import os
from typing import Dict, Sequence

import cv2
import numpy as np

from PyGEECSPlotter.ni_imread import read_imaq_image
from PyGEECSPlotter.magspec.calibrations import CameraCalibration


def _averaged_bg_path(analysis_dir: str, bg_scan_num: int, diagnostic: str) -> str:
    """
    Path to the averaged-background PNG for one diagnostic.

    Matches the matlab naming convention from ``fBellaBgV03``:
    ``Scan###<diagnostic>_averaged.png`` (note: no separator between the
    scan-number and the diagnostic name).
    """
    return os.path.join(
        analysis_dir,
        f"Scan{bg_scan_num:03d}{diagnostic}_averaged.png",
    )


def _shot_glob(scan_dir: str, scan_num: int, diagnostic: str) -> str:
    return os.path.join(
        scan_dir,
        f"Scan{scan_num:03d}",
        diagnostic,
        f"Scan{scan_num:03d}_{diagnostic}_*.png",
    )


def _load_averaged_backgrounds(
    cameras: Sequence[CameraCalibration],
    paths: Dict[str, str],
) -> Dict[str, np.ndarray]:
    bg_images: Dict[str, np.ndarray] = {}
    for cam in cameras:
        img = read_imaq_image(paths[cam.diagnostic])
        if img is None:
            raise IOError(f"Failed to read averaged background: {paths[cam.diagnostic]!r}.")
        bg_images[cam.diagnostic] = img.astype(np.float64)
    return bg_images


def _build_averaged_backgrounds(
    cameras: Sequence[CameraCalibration],
    scan_dir: str,
    analysis_dir: str,
    bg_scan_num: int,
    save: bool,
) -> Dict[str, np.ndarray]:
    bg_images: Dict[str, np.ndarray] = {}
    for cam in cameras:
        diag = cam.diagnostic
        pattern = _shot_glob(scan_dir, bg_scan_num, diag)
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No background shots found for {diag!r} in bg scan "
                f"{bg_scan_num} (pattern: {pattern!r})."
            )

        accum: np.ndarray = np.zeros((0, 0))
        for k, path in enumerate(files):
            arr = read_imaq_image(path)
            if arr is None:
                raise IOError(f"Failed to read {path!r}.")
            arr = arr.astype(np.float64)
            accum = arr if k == 0 else accum + arr
        avg = accum / len(files)
        bg_images[diag] = avg

        if save:
            os.makedirs(analysis_dir, exist_ok=True)
            out_path = _averaged_bg_path(analysis_dir, bg_scan_num, diag)
            int_img = np.round(avg).clip(0, 65535).astype(np.uint16)
            cv2.imwrite(out_path, int_img)

    return bg_images


def load_or_build_background(
    cameras: Sequence[CameraCalibration],
    scan_dir: str,
    analysis_dir: str,
    bg_scan_num: int,
    save: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Return per-camera averaged background images, building them if absent.

    Ports ``fBellaBgV03``. Generalised to use the caller-supplied list of
    ``CameraCalibration`` instead of the matlab hardcoded diagnostic list.

    Parameters
    ----------
    cameras : sequence of CameraCalibration
        Cameras to load / build backgrounds for. Phosphor entries should
        already be filtered out by ``load_camera_calibration``.
    scan_dir : str
        Parent directory holding ``Scan###/<diagnostic>/`` per-shot subdirs.
    analysis_dir : str
        Directory for averaged-background PNGs (also where they're checked
        for first).
    bg_scan_num : int
        Scan number designated as the background scan.
    save : bool, optional
        If True (default) and we have to build the averages, also write
        ``Scan{NNN}<diagnostic>_averaged.png`` to ``analysis_dir`` for
        next time (matlab does this).

    Returns
    -------
    dict
        ``{diagnostic: bg_image}`` (float64, full sensor size).
    """
    paths = {
        cam.diagnostic: _averaged_bg_path(analysis_dir, bg_scan_num, cam.diagnostic)
        for cam in cameras
    }

    if all(os.path.exists(p) for p in paths.values()):
        try:
            return _load_averaged_backgrounds(cameras, paths)
        except IOError:
            pass  # Fall through to rebuild from raw shots.

    return _build_averaged_backgrounds(
        cameras, scan_dir, analysis_dir, bg_scan_num, save,
    )
