# Multi-camera stitch onto uniform (angle, momentum) axes.
#
# Ports the matlab `fBellaUaV01` (resample to uniform angle),
# `fBellaMmtBinV01` (per-camera pixel binning along momentum),
# `fBellaUmV01` (resample to uniform momentum), and `fBellaUamCmbV03`
# (per-window combine orchestrator). Generalised over an arbitrary
# number of cameras per screen.

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from PyGEECSPlotter.magspec.calibrations import CameraCalibration
from PyGEECSPlotter.magspec.geometry import (
    AngleMap,
    CameraXAxis,
    UniformAngleAxis,
    UniformMomentumWindow,
)


# ----------------------------------------------------------------------
# Per-camera resampling
# ----------------------------------------------------------------------
def resample_to_uniform_angle(
    image: np.ndarray,
    angle_map: AngleMap,
    uniform_angle: UniformAngleAxis,
) -> np.ndarray:
    """
    Resample a per-camera image onto the common uniform-angle Y axis.
    Ports ``fBellaUaV01``.

    Each input column is mapped onto ``uniform_angle.angle`` using its
    own ``angle_map.angle[:, col]`` as the input axis. Counts are
    normalised by ``d_angle`` before interpolation and un-normalised
    afterwards. A final per-column scaling enforces column-sum
    conservation against the original image.

    Returns
    -------
    np.ndarray
        Image shaped ``(len(uniform_angle.angle), image.shape[1])``.
    """
    img = np.asarray(image, dtype=np.float64)
    szy, szx = img.shape
    target = np.asarray(uniform_angle.angle, dtype=np.float64)
    da = float(uniform_angle.d_angle)

    # Normalise by d_angle (matlab `img./dAngl`)
    norm = img / np.asarray(angle_map.d_angle, dtype=np.float64)

    # Zero top/bottom rows of the normalised image — matlab does this so
    # the interpolation has zero-valued endpoints to anchor to.
    norm[0, :] = 0.0
    norm[-1, :] = 0.0

    # Pad with one zero row above and below; extend the angle map past the
    # uniform-angle endpoints so the interpolation never has to extrapolate.
    pad_row = np.zeros((1, szx))
    pad_a_high = np.full((1, szx), float(target.max()) + da)
    pad_a_low = np.full((1, szx), float(target.min()) - da)
    img_pad = np.vstack([pad_row, norm, pad_row])
    angle_pad = np.vstack([pad_a_low, np.asarray(angle_map.angle, dtype=np.float64), pad_a_high])

    out = np.zeros((target.size, szx), dtype=np.float64)
    for col in range(szx):
        x_in = angle_pad[:, col]
        y_in = img_pad[:, col]
        order = np.argsort(x_in)  # interp wants monotonic input
        out[:, col] = np.interp(target, x_in[order], y_in[order])

    out *= da  # un-normalise

    # Column-sum conservation
    sum_orig = img.sum(axis=0)
    sum_new = out.sum(axis=0)
    sum_orig = np.where(sum_orig == 0, 1.0, sum_orig)
    sum_new = np.where(sum_new == 0, 1.0, sum_new)
    out *= (sum_orig / sum_new)[np.newaxis, :]
    return out


def bin_in_momentum(image: np.ndarray, bin_counts: np.ndarray) -> np.ndarray:
    """
    Per-camera pixel binning along the momentum (column) axis.
    Ports ``fBellaMmtBinV01``.

    ``bin_counts[k]`` gives how many consecutive input columns are summed
    into output column ``k``. The total of ``bin_counts`` should equal
    the input image width (or less; remaining columns are ignored).
    """
    img = np.asarray(image, dtype=np.float64)
    bin_counts = np.asarray(bin_counts, dtype=int)
    szy = img.shape[0]
    szx_out = bin_counts.size
    out = np.zeros((szy, szx_out), dtype=np.float64)
    cursor = 0
    for k in range(szx_out):
        c = int(bin_counts[k])
        out[:, k] = img[:, cursor:cursor + c].sum(axis=1)
        cursor += c
    return out


def resample_to_uniform_momentum(
    image: np.ndarray,
    mmt_binned: np.ndarray,
    dp_binned: np.ndarray,
    window: UniformMomentumWindow,
) -> np.ndarray:
    """
    Resample a (column-concatenated) per-window image onto the window's
    uniform-momentum X axis. Ports ``fBellaUmV01``.

    Each input column is normalised by its ``dp_binned`` width before
    interpolation, then un-normalised by multiplying with ``window.dp``.

    Parameters
    ----------
    image : np.ndarray
        Shape ``(angle_resolution, total_concatenated_columns)``.
    mmt_binned : np.ndarray
        Per-column momentum values for ``image`` (one entry per input
        column). Need not be monotonic; sorted internally.
    dp_binned : np.ndarray
        Per-column dp widths (one entry per input column).
    window : UniformMomentumWindow
    """
    img = np.asarray(image, dtype=np.float64)
    mmt = np.asarray(mmt_binned, dtype=np.float64)
    dp = np.asarray(dp_binned, dtype=np.float64)

    # Normalise per column: img / dp (broadcast across rows)
    norm = img / dp[np.newaxis, :]

    order = np.argsort(mmt)
    mmt_sorted = mmt[order]

    target = np.asarray(window.mmt, dtype=np.float64)
    szy = img.shape[0]
    out = np.zeros((szy, target.size), dtype=np.float64)
    for row in range(szy):
        row_vals = norm[row, :][order]
        out[row, :] = window.dp * np.interp(target, mmt_sorted, row_vals)
    return out


# ----------------------------------------------------------------------
# Per-window orchestrator
# ----------------------------------------------------------------------
def combine_window(
    screen: str,
    indices: Sequence[int],
    images: Sequence[np.ndarray],
    angle_maps: Sequence[AngleMap],
    x_axes: Sequence[CameraXAxis],
    uniform_angle: UniformAngleAxis,
    window: UniformMomentumWindow,
) -> np.ndarray:
    """
    Build the stitched 2-D spectrum for one screen / window.

    Ports the inner body of ``fBellaUamCmbV03``. For each camera in this
    window: resample its processed image to uniform angle, bin along
    momentum to the window dp, then concatenate all cameras column-wise
    and resample onto the window's uniform momentum axis. Multiply by
    1000 (fC → aC), and zero out the first / last column for clean
    edges.

    Parameters
    ----------
    screen : str
        Label for this window (e.g. ``'front'``); only used in error
        messages.
    indices : sequence of int
        Indices into the parallel ``images`` / ``angle_maps`` /
        ``x_axes`` lists that belong to this screen.
    images : sequence of np.ndarray
        Per-camera processed images from ``process_camera_image``.
    angle_maps, x_axes : sequence
        Per-camera (full, not just this screen) angle maps and x-axes.
    uniform_angle : UniformAngleAxis
    window : UniformMomentumWindow

    Returns
    -------
    np.ndarray
        Shape ``(len(uniform_angle.angle), len(window.mmt))``, in aC per
        (uniform mrad bin × uniform MeV/c bin).
    """
    if not indices:
        raise ValueError(f"No cameras assigned to window {screen!r}.")

    angle_then_binned: List[np.ndarray] = []
    dp_list: List[np.ndarray] = []
    mmt_list: List[np.ndarray] = []

    for i in indices:
        x = x_axes[i]
        if x.bin_counts is None or x.dp_binned is None or x.mmt_binned is None:
            raise RuntimeError(
                f"x_axes[{i}] is missing binning info. "
                f"Call uniform_momentum_axes(...) first."
            )

        ua = resample_to_uniform_angle(images[i], angle_maps[i], uniform_angle)
        binned = bin_in_momentum(ua, x.bin_counts)
        angle_then_binned.append(binned)
        dp_list.append(np.asarray(x.dp_binned, dtype=np.float64))
        mmt_list.append(np.asarray(x.mmt_binned, dtype=np.float64))

    concat = np.concatenate(angle_then_binned, axis=1)
    dp_all = np.concatenate(dp_list)
    mmt_all = np.concatenate(mmt_list)

    out = resample_to_uniform_momentum(concat, mmt_all, dp_all, window) * 1000.0  # fC → aC
    out[:, 0] = 0.0
    out[:, -1] = 0.0
    return out


def stitch_all_windows(
    cameras: Sequence[CameraCalibration],
    images: Sequence[np.ndarray],
    angle_maps: Sequence[AngleMap],
    x_axes: Sequence[CameraXAxis],
    uniform_angle: UniformAngleAxis,
    windows: Dict[str, UniformMomentumWindow],
) -> Dict[str, np.ndarray]:
    """
    Stitch one image per screen present in ``cameras``. Ports the top
    level of ``fBellaUamCmbV03``, generalised so any number of screens
    (typically ``'front'`` and ``'side'``) is supported transparently.

    Returns
    -------
    dict
        ``{screen: stitched_2d_image}``.
    """
    if not (len(cameras) == len(images) == len(angle_maps) == len(x_axes)):
        raise ValueError(
            "cameras / images / angle_maps / x_axes must be parallel lists "
            "(got lengths %d, %d, %d, %d)."
            % (len(cameras), len(images), len(angle_maps), len(x_axes))
        )

    by_screen: Dict[str, List[int]] = {}
    for i, cam in enumerate(cameras):
        by_screen.setdefault(cam.screen, []).append(i)

    out: Dict[str, np.ndarray] = {}
    for screen, indices in by_screen.items():
        if screen not in windows:
            raise KeyError(
                f"windows dict is missing screen {screen!r}. "
                f"Got: {sorted(windows)}."
            )
        out[screen] = combine_window(
            screen, indices, images, angle_maps, x_axes,
            uniform_angle, windows[screen],
        )
    return out
