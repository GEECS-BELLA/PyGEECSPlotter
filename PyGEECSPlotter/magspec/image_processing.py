# Per-camera image processing for the BELLA magspec port.
#
# Ports of the matlab helpers `fTrexTonyBgV01`, `fXrayOutV10`,
# `fImageRotV02`, plus the per-camera pipeline `fBellaImgV02`.
#
# The 12-bit PNG reader (`f12bitPngOpnV04`) is supplied by the existing
# `PyGEECSPlotter.ni_imread.read_imaq_image`, which handles the same
# sBIT-aware right-shift; we reuse it directly rather than re-porting.
#
# The pivot rotation (`fImageRotV02`) is similarly supplied by the
# existing `ImageAnalyzer.rotate_around` (scipy.ndimage.affine_transform
# with the same bilinear interpolation).

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from PyGEECSPlotter.image_analysis import ImageAnalyzer
from PyGEECSPlotter.ni_imread import read_imaq_image
from PyGEECSPlotter.magspec.calibrations import CameraCalibration


# ----------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------
def count_saturated_pixels(image: np.ndarray, threshold: int = 4095) -> int:
    """
    Count pixels equal to the saturation threshold (default 4095 for 12-bit).
    Matches matlab ``sum(sum(imgR == 4095))`` from ``fBellaImgV02``.
    """
    return int(np.count_nonzero(image == threshold))


def subtract_background_iterative(
    image: np.ndarray,
    bg: np.ndarray,
    max_iter: int = 100,
) -> np.ndarray:
    """
    "Tony's" iterative background subtraction. Ports ``fTrexTonyBgV01``.

    If the background sums to more than the image, returns zeros
    (high-background protection). Otherwise:

      1. Compute ``image - bg``.
      2. Sum the negative-valued pixels' magnitudes; zero them out.
      3. Redistribute that total over the positive pixels (subtract
         ``ngtCnt / n_positive`` from each), zero out any new negatives.
      4. Repeat until the negative-total is ≤ 1 or ``max_iter`` is hit.
      5. If ``max_iter`` is hit before convergence, also return zeros.

    Parameters
    ----------
    image, bg : np.ndarray
        Same-shaped 2-D arrays.
    max_iter : int, optional
        Iteration cap, matching matlab's hardcoded 100.

    Returns
    -------
    np.ndarray
        Background-subtracted image, same shape and dtype as ``image``.
    """
    if image.shape != bg.shape:
        raise ValueError(f"image and bg shape mismatch: {image.shape} vs {bg.shape}.")

    img = np.asarray(image, dtype=np.float64)
    bg = np.asarray(bg, dtype=np.float64)

    if bg.sum() > img.sum():
        return np.zeros_like(img)

    work = img - bg
    neg_mask = work < 0
    pos_mask = work > 0
    neg_total = -work[neg_mask].sum()
    work[neg_mask] = 0.0

    for _ in range(max_iter - 1):
        if neg_total <= 1:
            return work

        n_pos = int(pos_mask.sum())
        if n_pos == 0:
            return np.zeros_like(img)

        # Redistribute neg-total uniformly across positive pixels
        work = work - pos_mask * (neg_total / n_pos)
        neg_mask = work < 0
        pos_mask = work > 0
        neg_total = -work[neg_mask].sum()
        work[neg_mask] = 0.0

    # Did not converge within max_iter → high-background protection
    if neg_total > 1:
        return np.zeros_like(img)
    return work


def lowpass_xray_filter(
    image: np.ndarray,
    factor: float,
    pit: int,
    min_counts: float,
    n_iter: int,
) -> Tuple[np.ndarray, float]:
    """
    Iterative 4-neighbour X-ray rejection / low-pass filter.
    Ports ``fXrayOutV10``.

    At each iteration, every pixel is compared with the average of its
    four neighbours at distance ``pit``. A pixel is replaced with that
    local average if either:

      - ``factor > 0`` and ``current - factor * avg > 0`` and
        ``input >= min_counts``  (typical X-ray rejection)
      - ``factor < 0`` and ``current * factor + avg > 0`` and
        ``input >= min_counts``  (matlab "negative factor" mode)

    The edges of the image are padded by ``pit`` pixels using edge
    replication; corners use zeros (matches the matlab original exactly).
    The padded image is rebuilt at every iteration so corrections
    propagate inward.

    Returns
    -------
    image_out : np.ndarray
    count_removed : float
        ``sum(image) - sum(image_out)`` — the total counts taken out.
    """
    img_in = np.asarray(image, dtype=np.float64)
    out = img_in.copy()
    if factor == 0 or n_iter <= 0:
        return out, 0.0

    pit = int(pit)
    szy, szx = out.shape

    def _pad(arr: np.ndarray) -> np.ndarray:
        crn = np.zeros((pit, pit), dtype=np.float64)
        top = arr[:pit, :]
        bot = arr[-pit:, :]
        left = arr[:, :pit]
        right = arr[:, -pit:]
        return np.block([
            [crn, top, crn],
            [left, arr, right],
            [crn, bot, crn],
        ])

    pd = _pad(out)
    for _ in range(int(n_iter)):
        # 4-neighbour average at distance `pit`
        ref_top = 0.25 * pd[:szy, pit:pit + szx]
        ref_bot = 0.25 * pd[2 * pit:2 * pit + szy, pit:pit + szx]
        ref_left = 0.25 * pd[pit:pit + szy, :szx]
        ref_right = 0.25 * pd[pit:pit + szy, 2 * pit:2 * pit + szx]
        ref = ref_top + ref_bot + ref_left + ref_right

        if factor > 0:
            mask = ((out - factor * ref) > 0) & (img_in >= min_counts)
        else:
            mask = ((out * factor + ref) > 0) & (img_in >= min_counts)

        out = np.where(mask, ref, out)
        pd = _pad(out)

    return out, float(img_in.sum() - out.sum())


def rotate_around_center(image: np.ndarray, angle_deg: float, camera: CameraCalibration) -> np.ndarray:
    """
    Rotate ``image`` by ``angle_deg`` around the FULL-IMAGE centre
    ``(camera.width / 2, camera.height / 2)``.

    Thin wrapper around ``ImageAnalyzer.rotate_around``. The matlab original
    (``fImageRotV02`` invoked from ``fBellaImgV02``) always pivots on the
    centre of the un-cropped sensor image, not the analysis ROI.
    """
    x0 = camera.width / 2.0
    y0 = camera.height / 2.0
    return ImageAnalyzer.rotate_around(image, angle_deg, x0, y0)


# ----------------------------------------------------------------------
# Per-camera pipeline
# ----------------------------------------------------------------------
def process_camera_image(
    image: Optional[np.ndarray],
    camera: CameraCalibration,
    bg: np.ndarray,
    lowpass_params: Tuple[float, int, float, int],
    c2c: float,
    vignette: np.ndarray,
    inc_angle: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Run one camera's full per-shot pipeline. Ports ``fBellaImgV02`` for a
    single camera (the matlab function loops over a list; here a caller
    loops outside and calls this once per camera).

    Pipeline:

      1) Saturation count (raw image, ≥ 4095).
      2) Background subtract (``subtract_background_iterative``).
      3) Rotate around the full-image centre by ``camera.rot``.
      4) Crop to the analysis ROI (``y_start..y_end``, ``x_start..x_end``).
      5) Low-pass / X-ray filter with ``lowpass_params``.
      6) Multiply by the camera's ``vignette`` correction.
      7) Apply ``cos(inc_angle)`` per column (broadcast across rows).
      8) Multiply by ``c2c`` (counts → fC).
      9) Multiply by ``camera.sensitivity``.

    Parameters
    ----------
    image : np.ndarray or None
        Raw 12-bit image, full sensor size. ``None`` means the file was
        missing for this shot; the function then short-circuits to a
        zero-filled ROI array.
    camera : CameraCalibration
    bg : np.ndarray
        Full-size background image (same shape as ``image``).
    lowpass_params : (factor, pit, min_counts, n_iter)
        Passed straight through to ``lowpass_xray_filter``.
    c2c : float
        Counts-to-charge factor [fC / count] from
        ``compute_c2c_and_vignette``.
    vignette : np.ndarray
        Vignette correction shaped to the analysis ROI, same source.
    inc_angle : np.ndarray
        Per-column incident angle [deg], length =
        ``camera.x_end - camera.x_start + 1``.

    Returns
    -------
    processed : np.ndarray
        Per-pixel charge density on the analysis ROI [fC, scaled by
        ``camera.sensitivity``].
    saturated : int
        Number of saturated pixels in the raw image (0 if image was None).
    """
    if image is None:
        roi_h = camera.y_end - camera.y_start + 1
        roi_w = camera.x_end - camera.x_start + 1
        return np.zeros((roi_h, roi_w), dtype=np.float64), 0

    img = np.asarray(image, dtype=np.float64)

    # 1) saturation count BEFORE any processing
    saturated = count_saturated_pixels(img)

    # 2) background subtraction (full image)
    img = subtract_background_iterative(img, bg)

    # 3) rotation around full-image centre
    img = rotate_around_center(img, camera.rot, camera)

    # 4) ROI crop (matlab 1-indexed inclusive → numpy [a-1:b])
    img = img[camera.y_start - 1:camera.y_end, camera.x_start - 1:camera.x_end]

    # 5) X-ray / low-pass filter
    img, _ = lowpass_xray_filter(img, *lowpass_params)

    # 6) vignette compensation
    img = img * vignette

    # 7) incident-angle compensation: cos(deg) broadcast across rows
    cos_angle = np.cos(np.deg2rad(inc_angle))   # shape (W_roi,)
    img = img * cos_angle[np.newaxis, :]

    # 8-9) c2c + per-camera sensitivity → charge density
    img = c2c * img * camera.sensitivity

    return img, saturated
