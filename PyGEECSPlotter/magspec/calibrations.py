# Calibration loaders for the BELLA magnetic spectrometer port.
#
# Ports of the matlab calibration helpers (`fBellaCamCalibV02`,
# `fBellaTrjCalibFSV04`, `fLanexClbOutV01`, `fLanexClbV02`,
# `fBellaCalibPathV01`) into typed python dataclasses plus pandas-based
# loaders. See MAGSPEC_PORT.md for the function-by-function map.

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------
@dataclass
class CameraCalibration:
    """One row of `*camCalib.txt`. One entry per magspec camera.

    Matches the matlab `camClb` struct produced by `fBellaCamCalibV02`,
    plus a `screen` field (added for the python port) that says which
    spectrometer screen this camera looks at: ``'front'`` or ``'side'``.
    Phosphor cameras are filtered out at load time (`screen='phosphor'`).
    """

    diagnostic: str             # e.g. 'CAM-TEA-MagSpecA'
    screen: str                 # 'front' or 'side'
    fov: float                  # field of view [mm]
    y_offset: int               # sensor ROI y offset [pixel]
    height: int                 # sensor ROI height [pixel]
    x_offset: int               # sensor ROI x offset [pixel]
    width: int                  # sensor ROI width [pixel]
    left_pos: float             # screen-space left edge [mm]
    y_center: float             # y center pixel
    y_start: int                # analysis ROI y start (1-indexed in matlab)
    y_end: int                  # analysis ROI y end
    x_start: int                # analysis ROI x start
    x_end: int                  # analysis ROI x end
    rot: float                  # in-plane rotation [deg]
    sensitivity: float          # relative sensitivity multiplier
    set_n: int                  # lanex-calibration set index
    # damage / vignetting holes — up to 4, each (x1, x2, y1, y2). Zero
    # arrays mean "no hole".
    holes: List[Tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class TrajectoryCalibration:
    """Trajectory calibration for one screen (front or side).

    Ports the per-screen half of matlab's `fBellaTrjCalibFSV04` output.
    All arrays are sorted by ``mmt`` and have the same length.
    """

    mmt: np.ndarray             # momentum [MeV/c]
    screen_pos: np.ndarray      # screen-space position [mm]
    inc_angle: np.ndarray       # incident angle to screen [deg]
    path: np.ndarray            # total path length [m]
    div_fy: np.ndarray          # y diverging factor (rms)
    resolution: np.ndarray      # momentum resolution [%/mrad]


@dataclass
class LanexCalibration:
    """One row of `*lanexCalib.txt` (one entry per lanex `setN`).

    Ports `fLanexClbOutV01`. Captures the FOV slope+offset, the radial
    vignette polynomial (4th + 2nd + 0th order), the per-camera-distance
    sensitivity polynomial, and the full sensor size used for the radial
    map.
    """

    set_n: int
    fov_slope: float
    fov_offset: float
    vignette_4: float
    vignette_2: float
    vignette_0: float
    sense_2: float
    sense_1: float
    sense_0: float
    width: int                  # full sensor width [pixel]
    height: int                 # full sensor height [pixel]


# ----------------------------------------------------------------------
# Path discovery
# ----------------------------------------------------------------------
def discover_calib_path(calib_dir: str, pattern: str, day: str) -> Tuple[str, int]:
    """
    Find the calibration file matching ``pattern`` whose YYMMDD prefix is
    the latest date not exceeding ``day``.

    Ports matlab's ``fBellaCalibPathV01``.

    Parameters
    ----------
    calib_dir : str
        Directory holding the calibration files.
    pattern : str
        Glob pattern (e.g. ``'*camCalib.txt'``).
    day : str
        Experiment day as ``'YY_MMDD'`` (e.g. ``'25_0827'``).

    Returns
    -------
    path : str
        Full path to the chosen file.
    calib_date : int
        Six-digit YYMMDD date of the chosen file.
    """
    day_int = int(day[:2] + day[3:])
    candidates = sorted(glob.glob(os.path.join(calib_dir, pattern)))
    if not candidates:
        raise FileNotFoundError(
            f"No calibration files matching {pattern!r} in {calib_dir!r}."
        )

    valid: List[Tuple[int, str]] = []
    for path in candidates:
        name = os.path.basename(path)
        match = re.match(r'(\d{6})', name)
        if match is None:
            continue
        file_day = int(match.group(1))
        if file_day <= day_int:
            valid.append((file_day, path))

    if not valid:
        raise FileNotFoundError(
            f"No {pattern!r} in {calib_dir!r} with date <= {day_int}."
        )

    valid.sort()
    file_day, path = valid[-1]
    return path, file_day


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _read_calib_table(path: str) -> pd.DataFrame:
    """Tab-separated calibration file → DataFrame, with whitespace-stripped columns."""
    df = pd.read_csv(path, sep='\t', engine='python')
    df.columns = df.columns.str.strip()
    return df


# ----------------------------------------------------------------------
# Camera calibration
# ----------------------------------------------------------------------
def load_camera_calibration(path: str) -> List[CameraCalibration]:
    """
    Load ``*camCalib.txt``.

    Returns a list of ``CameraCalibration`` (one per row), silently skipping
    any row whose ``screen`` field is ``'phosphor'``.

    The file must include two extra columns added for the python port:
    ``screen`` (``'front'`` / ``'side'`` / ``'phosphor'``) and ``diagnostic``
    (e.g. ``'CAM-TEA-MagSpecA'``). See ``MAGSPEC_PORT.md`` for details.

    Ports matlab's ``fBellaCamCalibV02``.
    """
    df = _read_calib_table(path)

    for required in ('screen', 'diagnostic'):
        if required not in df.columns:
            raise KeyError(
                f"Camera calibration file is missing the {required!r} column. "
                f"See MAGSPEC_PORT.md for the python-port column additions."
            )

    cameras: List[CameraCalibration] = []
    for _, row in df.iterrows():
        screen = str(row['screen']).strip().lower()
        if screen == 'phosphor':
            continue
        if screen not in ('front', 'side'):
            raise ValueError(
                f"Unknown screen value {screen!r} in camera calibration. "
                f"Expected 'front', 'side', or 'phosphor'."
            )
        cameras.append(_camera_row_to_dataclass(row, screen))

    return cameras


def _camera_row_to_dataclass(row: pd.Series, screen: str) -> CameraCalibration:
    holes: List[Tuple[int, int, int, int]] = []
    for i in (1, 2, 3, 4):
        keys = (f'hole{i} x1', f'hole{i} x2', f'hole{i} y1', f'hole{i} y2')
        if all(k in row.index for k in keys):
            vals = tuple(int(row[k]) for k in keys)
            if any(vals):
                holes.append(vals)  # type: ignore[arg-type]

    diag = str(row['diagnostic']).strip()
    if not diag or diag.lower() == 'nan':
        raise ValueError(
            "Empty 'diagnostic' value in camera calibration row "
            f"(screen={screen!r}). Every non-phosphor row needs a diagnostic name."
        )

    return CameraCalibration(
        diagnostic=diag,
        screen=screen,
        fov=float(row['FOV [mm]']),
        y_offset=int(row['ROI Y offset']),
        height=int(row['ROI height']),
        x_offset=int(row['ROI X offset']),
        width=int(row['ROI width']),
        left_pos=float(row['Left edge [mm]']),
        y_center=float(row['Y center pixel']),
        y_start=int(row['Y Start']),
        y_end=int(row['Y End']),
        x_start=int(row['X Start']),
        x_end=int(row['X End']),
        rot=float(row['rot [deg]']),
        sensitivity=float(row['sensitivity']),
        set_n=int(row['setN']),
        holes=holes,
    )


# ----------------------------------------------------------------------
# Trajectory calibration
# ----------------------------------------------------------------------
def load_trajectory_calibration(
    path: str,
) -> Tuple[TrajectoryCalibration, TrajectoryCalibration]:
    """
    Load ``*trjCalib*A0.txt``.

    Returns ``(front_calib, side_calib)``. The file uses a ``side logic``
    column: rows where ``side logic == 1`` are side-screen rows (come first),
    rows where ``side logic == 0`` are front-screen.

    Ports matlab's ``fBellaTrjCalibFSV04``.
    """
    df = _read_calib_table(path)
    if 'side logic' not in df.columns:
        raise KeyError("Trajectory calibration file is missing 'side logic' column.")

    side_logic = df['side logic'].astype(int)
    transition = side_logic[side_logic == 0].index
    if len(transition) == 0:
        raise ValueError("Trajectory calibration has no front-screen rows (side logic == 0).")
    k_first_front = int(transition[0])

    side_df = df.iloc[:k_first_front].reset_index(drop=True)
    front_df = df.iloc[k_first_front:].reset_index(drop=True)
    return _make_trj_calib(front_df, 'front'), _make_trj_calib(side_df, 'side')


def _make_trj_calib(df: pd.DataFrame, screen: str) -> TrajectoryCalibration:
    pos_col = f'{screen} screen [m]'
    angle_col = f'bending angle at {screen} screen [dgr]'
    for c in (pos_col, angle_col, 'momentum [MeV/c]', 'total path [m]',
              'y conv fct rms', 'momentum rsl [%/mrad]'):
        if c not in df.columns:
            raise KeyError(f"Trajectory calibration missing column {c!r}.")
    return TrajectoryCalibration(
        mmt=df['momentum [MeV/c]'].to_numpy(dtype=float),
        screen_pos=1000.0 * df[pos_col].to_numpy(dtype=float),   # m → mm
        inc_angle=df[angle_col].to_numpy(dtype=float),
        path=df['total path [m]'].to_numpy(dtype=float),
        div_fy=df['y conv fct rms'].to_numpy(dtype=float),
        resolution=df['momentum rsl [%/mrad]'].to_numpy(dtype=float),
    )


# ----------------------------------------------------------------------
# Lanex calibration
# ----------------------------------------------------------------------
def load_lanex_calibration_table(path: str) -> Dict[int, LanexCalibration]:
    """
    Load ``*lanexCalib.txt``. Returns a dict keyed by ``set_n``.

    Each row corresponds to one lanex set (camera-to-screen distance config).
    Callers pick the entry matching their camera's ``set_n``.

    Ports matlab's ``fLanexClbOutV01`` (which selects a single set_n).
    """
    df = _read_calib_table(path)
    out: Dict[int, LanexCalibration] = {}
    for idx, row in df.iterrows():
        n = int(row['setN']) if 'setN' in df.columns else (int(idx) + 1)  # type: ignore[arg-type]
        out[n] = LanexCalibration(
            set_n=n,
            fov_slope=float(row['FOV slope']),
            fov_offset=float(row['FOV offset']),
            vignette_4=float(row['vignette 4']),
            vignette_2=float(row['vignette 2']),
            vignette_0=float(row['vignette 0']),
            sense_2=float(row['sensitivity 2']),
            sense_1=float(row['sensitivity 1']),
            sense_0=float(row['sensitivity 0']),
            width=int(row['full width']),
            height=int(row['full height']),
        )
    return out


# ----------------------------------------------------------------------
# Counts-to-charge + vignette
# ----------------------------------------------------------------------
def compute_c2c_and_vignette(
    camera: CameraCalibration,
    lanex: LanexCalibration,
    lanex_kind: str = 'front',
) -> Tuple[float, np.ndarray]:
    """
    Per-camera counts-to-charge factor [fC/count @ 1 GeV] and radial vignette
    compensation matrix sized for the camera's analysis ROI.

    Ports matlab's ``fLanexClbV02``.

    Parameters
    ----------
    camera : CameraCalibration
    lanex : LanexCalibration
        The lanex-table entry matching ``camera.set_n``.
    lanex_kind : {'front', 'back'}, optional
        Lanex screen type (``'front'`` = Lanex Fast Front (thin),
        ``'back'`` = LFB (thick)). Note: unrelated to spectrometer
        front-vs-side screen.

    Returns
    -------
    c2c : float
        Counts-to-charge factor at 1 GeV [fC/count].
    vignette : np.ndarray
        Multiplicative vignette correction (= 1 / polynomial) shaped to
        the analysis ROI ``(y_end - y_start + 1, x_end - x_start + 1)``.
    """
    screen_factor = 1.0 if lanex_kind.lower() == 'back' else 1.98

    z = (camera.fov - lanex.fov_offset) / lanex.fov_slope
    als_r = lanex.sense_2 * z ** 2 + lanex.sense_1 * z + lanex.sense_0
    c2c = screen_factor * als_r / 146.0

    # Radial map over the full sensor (matlab: meshgrid(1:W, 1:H))
    xx, yy = np.meshgrid(
        np.arange(1, lanex.width + 1, dtype=float),
        np.arange(1, lanex.height + 1, dtype=float),
    )
    # Matlab: aaa - W/2 + 0.5  (operator order: subtract, then add)
    xx -= lanex.width / 2.0 - 0.5
    yy -= lanex.height / 2.0 - 0.5
    r = np.sqrt(xx ** 2 + yy ** 2)

    # Sensor ROI: matlab uses [yOffset+1 : yOffset+height] (1-indexed inclusive)
    y_st = camera.y_offset
    y_ed = camera.y_offset + camera.height
    x_st = camera.x_offset
    x_ed = camera.x_offset + camera.width
    vgnt = r[y_st:y_ed, x_st:x_ed]

    # Analysis ROI (matlab 1-indexed inclusive both ends → numpy [a-1:b])
    vgnt = vgnt[camera.y_start - 1:camera.y_end, camera.x_start - 1:camera.x_end]

    poly = lanex.vignette_4 * vgnt ** 4 + lanex.vignette_2 * vgnt ** 2 + lanex.vignette_0
    return c2c, 1.0 / poly
