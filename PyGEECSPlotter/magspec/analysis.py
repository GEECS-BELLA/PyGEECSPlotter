# Scalar-stat extraction from the stitched 2-D spectrum.
#
# Ports `fSpotAnalysisV01` + its helpers (`fGetFwhmV04`, `fGetRmsV01`).
# Used by MagSpecAnalyzer to produce the per-shot scalars that get
# written back to the sfile.

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _rms(x: np.ndarray, counts: np.ndarray) -> Tuple[float, float]:
    """Weighted (mean, std) of ``x`` with weights ``counts``.

    Ports ``fGetRmsV01``. The std formula uses the variance-times-sum^2
    intermediate to mirror matlab byte-for-byte:

        rms = sqrt(N * sum(x²·c) − sum(x·c)²) / N
    where N = sum(c).
    """
    total = float(counts.sum())
    if total == 0:
        return 0.0, 0.0
    e_sum = float((x * counts).sum())
    e2_sum = float((x * x * counts).sum())
    mean = e_sum / total
    variance_times_total_squared = total * e2_sum - e_sum * e_sum
    if variance_times_total_squared < 0:
        variance_times_total_squared = 0.0
    rms = np.sqrt(variance_times_total_squared) / total
    return rms, mean


def _fwhm_inner(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    """
    FWHM measured "from inside" of the half-max region — the inner
    crossings (walking from peak outward, find first point below
    half-max; FWHM is distance between those two inner crossings).
    Ports the ``fwhmIn`` return of ``fGetFwhmV04``, plus the peak index.
    """
    if y.sum() == 0 or x.size <= 3:
        return 0.0, 0

    max_v = float(y.max())
    peak_idx = int(np.argmax(y))
    half = 0.5 * max_v

    right_y = y[peak_idx:]
    left_y = y[:peak_idx + 1]

    # Right side, from peak: first point below half-max, then step back one
    below_right = np.where(right_y < half)[0]
    if below_right.size == 0:
        # never drops below half-max → use right-edge index
        above_right_from_edge = np.where(right_y > half)[0]
        k1 = above_right_from_edge[-1] if above_right_from_edge.size > 0 else 0
    else:
        k1 = int(below_right[0]) - 1
        if k1 < 0:
            k1 = 0
    v1 = x[peak_idx + k1]

    # Left side, from peak: last point below half-max, then step forward one
    below_left = np.where(left_y < half)[0]
    if below_left.size == 0:
        above_left_from_edge = np.where(left_y > half)[0]
        k3 = above_left_from_edge[0] if above_left_from_edge.size > 0 else len(left_y) - 1
    else:
        k3 = int(below_left[-1]) + 1
        if k3 >= len(left_y):
            k3 = len(left_y) - 1
    v3 = x[k3]

    return float(abs(v1 - v3)), peak_idx


# ----------------------------------------------------------------------
# Public
# ----------------------------------------------------------------------
def spot_analysis(x: np.ndarray, y: np.ndarray, image: np.ndarray) -> Dict[str, float]:
    """
    Compute the 8 standard spot-analysis scalars for a 2-D image with
    axes ``x`` (columns) and ``y`` (rows).

    Ports ``fSpotAnalysisV01``.

    Returns
    -------
    dict
        ``{'peak_x', 'peak_y', 'mean_x', 'mean_y',
           'fwhm_x', 'fwhm_y', 'std_x', 'std_y'}``.
        Values use the same units as the supplied ``x`` and ``y`` axes
        (so callers control whether momentum is in MeV/c or GeV/c).
    """
    img = np.asarray(image, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Peak location (matlab: row argmax via column-of-row-max)
    col_max = img.max(axis=0)              # per-column max
    peak_xi = int(np.argmax(col_max))
    peak_yi = int(np.argmax(img[:, peak_xi]))

    # FWHM along the peak's row / column
    fwhm_x, peak_xi_from_fwhm = _fwhm_inner(x, img[peak_yi, :])
    peak_x = float(x[peak_xi_from_fwhm]) if x.size > 0 else 0.0

    fwhm_y, peak_yi_from_fwhm = _fwhm_inner(y, img[:, peak_xi])
    peak_y = float(y[peak_yi_from_fwhm]) if y.size > 0 else 0.0

    # RMS / mean over the projection sums
    std_x, mean_x = _rms(x, img.sum(axis=0))
    std_y, mean_y = _rms(y, img.sum(axis=1))

    return {
        'peak_x': peak_x,
        'peak_y': peak_y,
        'mean_x': mean_x,
        'mean_y': mean_y,
        'fwhm_x': fwhm_x,
        'fwhm_y': fwhm_y,
        'std_x': std_x,
        'std_y': std_y,
    }
