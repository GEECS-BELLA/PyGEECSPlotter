# Axis / geometry derivations for the BELLA magnetic spectrometer port.
#
# Ports of `fBellaAxisTri`, `fBellaAxisAllV04`, `fBellaAnglMapV01`,
# `fBellaUaYV02`, `fBellaUmXV03`. Generalized for variable camera count:
# screen membership comes from `CameraCalibration.screen` rather than from
# the hardcoded "cameras 1-3 front, 4-N side" rule in the matlab original.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import interp1d

from PyGEECSPlotter.magspec.calibrations import (
    CameraCalibration,
    TrajectoryCalibration,
)


# ----------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------
@dataclass
class CameraXAxis:
    """Per-camera x-axis info. Ports the matlab ``x(i)`` struct."""

    pixel: np.ndarray           # analysis-ROI pixel indices (1-indexed)
    mm: np.ndarray              # screen-space position [mm]
    dx: float                   # uniform pixel size [mm]
    inc_angle: np.ndarray       # incident angle to screen [deg]
    path: np.ndarray            # total path length [m]
    div_fy: np.ndarray          # y diverging factor (rms)
    acceptance: np.ndarray      # half-angle acceptance [mrad]
    mmt: np.ndarray             # momentum [MeV/c per T] (normalized to 1 T)
    dp: np.ndarray              # |d(mmt)/dpixel| [MeV/c]
    resolution: np.ndarray      # momentum resolution [%/mrad]
    dispersion: np.ndarray      # dp / dx [MeV/c per mm]
    # Filled in later by uniform_momentum_axes:
    dp_binned: Optional[np.ndarray] = None     # binned dp [MeV/c]
    mmt_binned: Optional[np.ndarray] = None    # binned momentum [MeV/c]
    bin_counts: Optional[np.ndarray] = None    # pixel count per bin


@dataclass
class CameraYAxis:
    """Per-camera y-axis info. Ports the matlab ``y(i)`` struct."""

    pixel: np.ndarray           # analysis-ROI pixel indices (1-indexed)
    mm: np.ndarray              # y position [mm], zero at yCntr
    dy: float                   # uniform pixel size [mm] (= x dx)


# ----------------------------------------------------------------------
# Per-camera axis
# ----------------------------------------------------------------------
def compute_camera_axis(
    camera: CameraCalibration,
    trajectory: TrajectoryCalibration,
    acceptance_mm: float,
    sign: int,
) -> Tuple[CameraXAxis, CameraYAxis]:
    """
    Build x/y axis info for one camera. Ports ``fBellaAxisTri``.

    Parameters
    ----------
    camera : CameraCalibration
    trajectory : TrajectoryCalibration
        Front or side trajectory calibration, matching ``camera.screen``.
    acceptance_mm : float
        Window whole-width acceptance [mm] for this screen (matlab's ``accp``).
    sign : int
        ``+1`` for front, ``-1`` for side. Controls direction of the
        ``mm`` axis sweep across the FOV (matlab ``signS``).
    """
    # Analysis-ROI pixel indices (matlab 1..width, 1-indexed)
    pixel_full = np.arange(1, camera.width + 1, dtype=int)
    pixel = pixel_full[camera.x_start - 1:camera.x_end]

    # Screen-space x (or z) in mm
    mm_full = np.linspace(camera.left_pos,
                          camera.left_pos + sign * camera.fov,
                          camera.width)
    mm = mm_full[camera.x_start - 1:camera.x_end]
    dx = float(mm[1] - mm[0])

    # Interpolate trajectory calibration onto the camera's mm axis.
    # Matlab uses cubic; SciPy ``interp1d(kind='cubic')`` with
    # ``fill_value='extrapolate'`` to mirror matlab's extrapolation behavior.
    def _interp(values: np.ndarray) -> np.ndarray:
        f = interp1d(trajectory.screen_pos, values, kind='cubic',
                     fill_value='extrapolate', assume_sorted=False)
        return f(mm)

    inc_angle = _interp(trajectory.inc_angle)
    path = _interp(trajectory.path)
    div_fy = _interp(trajectory.div_fy)
    mmt = _interp(trajectory.mmt)
    resolution = _interp(trajectory.resolution)

    # Acceptance (half-angle, screen-side) [mrad]:
    #   accp_mrad = 0.5 * acceptance_mm / (path * div_fy)
    acceptance = 0.5 * acceptance_mm / (path * div_fy)

    # dp per pixel: |diff(mmt)| with edge-extension averaging
    diff1 = np.diff(mmt)
    diff1_pad_left = np.concatenate(([diff1[0]], diff1))
    diff1_pad_right = np.concatenate((diff1, [diff1[-1]]))
    dp = np.abs(0.5 * (diff1_pad_right + diff1_pad_left))

    # Dispersion [MeV/c per mm]
    dispersion = dp / dx

    x = CameraXAxis(
        pixel=pixel,
        mm=mm,
        dx=dx,
        inc_angle=inc_angle,
        path=path,
        div_fy=div_fy,
        acceptance=acceptance,
        mmt=mmt,
        dp=dp,
        resolution=resolution,
        dispersion=dispersion,
    )

    # Y axis: matches matlab fBellaAxisTri tail
    y_pixel_full = np.arange(1, camera.height + 1, dtype=int)
    y_pixel = y_pixel_full[camera.y_start - 1:camera.y_end]
    y_mm = y_pixel.astype(float) * dx
    y_mm = y_mm - dx * camera.y_center
    y = CameraYAxis(pixel=y_pixel, mm=y_mm, dy=dx)

    return x, y


# ----------------------------------------------------------------------
# All-cameras orchestrator
# ----------------------------------------------------------------------
def compute_all_axes(
    cameras: Sequence[CameraCalibration],
    front_trajectory: TrajectoryCalibration,
    side_trajectory: TrajectoryCalibration,
    acceptance_mm: Optional[Dict[str, float]] = None,
) -> Tuple[List[CameraXAxis], List[CameraYAxis]]:
    """
    Compute x and y axis info for every camera in ``cameras``.

    Cameras with ``screen == 'front'`` use ``front_trajectory`` and sign ``+1``;
    cameras with ``screen == 'side'`` use ``side_trajectory`` and sign ``-1``.
    Generalises ``fBellaAxisAllV04`` for variable camera count.

    Parameters
    ----------
    cameras : sequence of CameraCalibration
    front_trajectory, side_trajectory : TrajectoryCalibration
    acceptance_mm : dict, optional
        ``{'front': float, 'side': float}`` whole-width window acceptances
        in mm. Defaults to ``{'front': 33, 'side': 40}`` (matlab default).

    Returns
    -------
    x_axes, y_axes : parallel lists, one per camera, same order as ``cameras``.
    """
    if acceptance_mm is None:
        acceptance_mm = {'front': 33.0, 'side': 40.0}

    x_axes: List[CameraXAxis] = []
    y_axes: List[CameraYAxis] = []
    for cam in cameras:
        if cam.screen == 'front':
            traj, sign = front_trajectory, +1
        elif cam.screen == 'side':
            traj, sign = side_trajectory, -1
        else:
            raise ValueError(
                f"Camera {cam.diagnostic!r} has unknown screen={cam.screen!r}; "
                f"expected 'front' or 'side'."
            )
        if cam.screen not in acceptance_mm:
            raise KeyError(
                f"acceptance_mm missing screen {cam.screen!r}; "
                f"provide {{'front': ..., 'side': ...}}."
            )
        x, y = compute_camera_axis(cam, traj, acceptance_mm[cam.screen], sign)
        x_axes.append(x)
        y_axes.append(y)

    return x_axes, y_axes


# ----------------------------------------------------------------------
# Angle maps
# ----------------------------------------------------------------------
@dataclass
class AngleMap:
    """Per-camera angle map. Ports the matlab ``anglC{i}`` / ``dAnglC{i}`` cells."""

    angle: np.ndarray           # mrad, shape (height_roi, width_roi)
    d_angle: np.ndarray         # mrad, same shape


def compute_angle_maps(
    x_axes: Sequence[CameraXAxis],
    y_axes: Sequence[CameraYAxis],
) -> List[AngleMap]:
    """
    Build per-camera (angle, d_angle) 2-D maps. Each pixel ``(row, col)``
    on the camera's analysis ROI gets the angle of the corresponding
    electron trajectory at that pixel [mrad], computed as

        angle = 1000 * atan(0.001 * y_mm / divFY / path)

    Ports ``fBellaAnglMapV01``.
    """
    if len(x_axes) != len(y_axes):
        raise ValueError("x_axes and y_axes must be parallel lists.")

    maps: List[AngleMap] = []
    for x, y in zip(x_axes, y_axes):
        # path / y mesh, then apply divFY
        path_mesh, y_mesh = np.meshgrid(x.path, y.mm)
        divfy_mesh, _ = np.meshgrid(x.div_fy, y.mm)
        y_with_div = y_mesh / divfy_mesh
        angle = 1000.0 * np.arctan(0.001 * y_with_div / path_mesh)  # [mrad]

        # d_angle: diff along rows, then average between consecutive rows
        # to keep the shape consistent with `angle`
        d_a = np.diff(angle, axis=0)
        d_a = 0.5 * (np.vstack([d_a[:1, :], d_a]) +
                     np.vstack([d_a, d_a[-1:, :]]))

        maps.append(AngleMap(angle=angle, d_angle=d_a))

    return maps


# ----------------------------------------------------------------------
# Uniform angle axis (Y)
# ----------------------------------------------------------------------
@dataclass
class UniformAngleAxis:
    """Uniform angle axis used to resample every camera onto a common Y."""

    angle: np.ndarray           # mrad
    d_angle: float              # mrad / pixel


def uniform_angle_axis(
    n_resolution: int,
    angle_range: Tuple[float, float] = (-1.3, 1.3),
) -> UniformAngleAxis:
    """
    Uniform angle axis [mrad] spanning ``angle_range`` with ``n_resolution``
    samples. Default range matches the latest matlab fBellaUaYV02
    (``[-1.3, 1.3]`` mrad, fixed).

    Earlier matlab versions derived the range from the per-camera angle
    maps; that mode is not provided here. If you need it, take
    ``(min/max of [m.angle.min() for m in maps])`` and pass it in.
    """
    if n_resolution < 2:
        raise ValueError(f"n_resolution must be >= 2, got {n_resolution!r}.")
    low, high = angle_range
    if not (high > low):
        raise ValueError(f"angle_range must be (low, high) with high > low; got {angle_range}.")
    angle = np.linspace(low, high, int(n_resolution))
    return UniformAngleAxis(angle=angle, d_angle=float(angle[1] - angle[0]))


# ----------------------------------------------------------------------
# Uniform momentum axis (X)
# ----------------------------------------------------------------------
@dataclass
class UniformMomentumWindow:
    """One stitched momentum window. Ports the matlab ``xA(k)`` entries."""

    screen: str                 # 'front' or 'side'
    mmt: np.ndarray             # [MeV/c per T] (1 T normalisation)
    dp: float                   # uniform pixel pitch [MeV/c per T]
    acceptance: np.ndarray      # interpolated onto the window axis
    inc_angle: np.ndarray
    dispersion: np.ndarray


def _greedy_pixel_binning(
    camera_dp: np.ndarray,
    camera_mmt: np.ndarray,
    window_dp: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Greedy per-camera pixel binning so each output bin has total dp ≈ window_dp.

    Mirrors the inner loop of ``fBellaUmXV03``:
      - walk pixels left-to-right
      - accumulate dp into the current bin
      - when the running sum exceeds the window dp, advance to the next bin
        (and push the offending pixel forward if the current bin already
        had > 1 pixel — to avoid making the bin grow indefinitely)

    Returns
    -------
    dp_binned, mmt_binned, bin_counts : np.ndarray
        Same length, with one entry per output bin.
    """
    width = len(camera_dp)
    dp_bin = np.zeros(width)
    bin_counts = np.zeros(width, dtype=int)
    j = 0
    for i in range(width):
        dp_bin[j] += camera_dp[i]
        bin_counts[j] += 1
        if dp_bin[j] / window_dp > 1:
            if bin_counts[j] > 1:
                dp_bin[j] -= camera_dp[i]
                bin_counts[j] -= 1
                if j + 1 < width:
                    dp_bin[j + 1] = camera_dp[i]
                    bin_counts[j + 1] = 1
            j += 1
            if j >= width:
                break

    zero_idx = np.where(bin_counts == 0)[0]
    if zero_idx.size == 0:
        # Every pixel became its own bin → no binning happened
        return camera_dp.copy(), camera_mmt.copy(), bin_counts.copy()

    cutoff = int(zero_idx[0])
    dp_binned = dp_bin[:cutoff].copy()
    counts_out = bin_counts[:cutoff].copy()
    mmt_binned = np.empty(cutoff, dtype=float)
    i = 0
    for k in range(cutoff):
        c = int(counts_out[k])
        mmt_binned[k] = float(np.mean(camera_mmt[i:i + c]))
        i += c
    return dp_binned, mmt_binned, counts_out


def uniform_momentum_axes(
    cameras: Sequence[CameraCalibration],
    x_axes: Sequence[CameraXAxis],
    momentum_resolutions: Dict[str, int],
) -> Dict[str, UniformMomentumWindow]:
    """
    Build one ``UniformMomentumWindow`` per screen represented in ``cameras``
    (``'front'`` and/or ``'side'``), and fill each ``x_axes[i].dp_binned`` /
    ``.mmt_binned`` / ``.bin_counts`` in place.

    Ports ``fBellaUmXV03``, generalised so any number of cameras can be
    assigned to a given screen.

    Parameters
    ----------
    cameras : sequence of CameraCalibration
    x_axes : sequence of CameraXAxis (parallel to ``cameras``; mutated in place)
    momentum_resolutions : dict
        ``{'front': int, 'side': int}`` — output pixel count for each window's
        momentum axis.

    Returns
    -------
    windows : dict
        ``{screen: UniformMomentumWindow}``.
    """
    if len(cameras) != len(x_axes):
        raise ValueError("cameras and x_axes must be parallel lists.")

    # Group cameras by screen, preserving order.
    by_screen: Dict[str, List[int]] = {}
    for i, cam in enumerate(cameras):
        by_screen.setdefault(cam.screen, []).append(i)

    for screen in by_screen:
        if screen not in momentum_resolutions:
            raise KeyError(
                f"momentum_resolutions is missing entry for screen {screen!r}. "
                f"Got: {sorted(momentum_resolutions)}."
            )

    windows: Dict[str, UniformMomentumWindow] = {}
    for screen, indices in by_screen.items():
        screen_x = [x_axes[i] for i in indices]

        max_p = float(max(np.max(x.mmt) for x in screen_x))
        min_p = float(min(np.min(x.mmt) for x in screen_x))
        n = int(momentum_resolutions[screen])
        if n < 2:
            raise ValueError(f"momentum_resolutions[{screen!r}] must be >= 2, got {n!r}.")

        mmt = np.linspace(min_p, max_p, n)
        dp_window = float(mmt[1] - mmt[0])

        # Concatenate all per-camera arrays and sort by mmt for np.interp
        all_mmt = np.concatenate([x.mmt for x in screen_x])
        order = np.argsort(all_mmt)
        sorted_mmt = all_mmt[order]

        def _interp_concat(values_list: List[np.ndarray]) -> np.ndarray:
            cat = np.concatenate(values_list)[order]
            return np.interp(mmt, sorted_mmt, cat)

        acceptance = _interp_concat([x.acceptance for x in screen_x])
        inc_angle = _interp_concat([x.inc_angle for x in screen_x])
        dispersion = _interp_concat([x.dispersion for x in screen_x])

        windows[screen] = UniformMomentumWindow(
            screen=screen,
            mmt=mmt,
            dp=dp_window,
            acceptance=acceptance,
            inc_angle=inc_angle,
            dispersion=dispersion,
        )

        # Per-camera binning info (mutates x_axes[i])
        for i in indices:
            dp_b, mmt_b, counts = _greedy_pixel_binning(
                x_axes[i].dp, x_axes[i].mmt, dp_window,
            )
            x_axes[i].dp_binned = dp_b
            x_axes[i].mmt_binned = mmt_b
            x_axes[i].bin_counts = counts

    return windows
