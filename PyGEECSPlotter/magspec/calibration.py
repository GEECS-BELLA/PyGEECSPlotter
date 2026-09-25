# Calibration loading for the BELLA triangle-chamber magnetic spectrometer.
# Ports fBellaCalibPathV01, fBellaCamCalibV02, fBellaLanexV02 /
# fLanexClbOutV01 / fLanexClbV02, fBellaTrjCalibAngle and
# fBellaTrjCalibFSAngl. All files live in the ESMCalib directory, which is
# passed in by the caller (never hardcoded).

import fnmatch
import os
from dataclasses import dataclass

import numpy as np

from PyGEECSPlotter.magspec.io import log_column, read_log
from PyGEECSPlotter.magspec.matlab_compat import find_first, interp1, mround

# Post-2022 camera names, in the MATLAB processing order (fBellaImgDirTr).
MAGSPEC_CAMERAS = [
    'CAM-TEA-MagSpecA', 'CAM-TEA-MagSpecB', 'CAM-TEA-MagSpecC',
    'CAM-TEA-MagSpecD', 'CAM-TEA-MagSpecG', 'CAM-TEA-MagSpecH',
    'CAM-TEA-MagSpecI', 'CAM-TEA-MagSpecJ', 'CAM-TEA-MagSpecK',
    'CAM-TEA-MagSpecL',
]

_CAM_FIELDS = {
    'fov': 'FOV [mm]', 'yOffset': 'ROI Y offset', 'height': 'ROI height',
    'xOffset': 'ROI X offset', 'width': 'ROI width', 'leftPos': 'Left edge [mm]',
    'yCntr': 'Y center pixel', 'ySt': 'Y Start', 'xSt': 'X Start', 'yEd': 'Y End',
    'xEd': 'X End', 'rot': 'rot [deg]', 'sns': 'sensitivity', 'setN': 'setN',
}
_INT_FIELDS = ('yOffset', 'height', 'xOffset', 'width', 'ySt', 'xSt', 'yEd', 'xEd', 'setN')


def day_number(day):
    """'26_0521' -> 260521 (``str2num([day(1:2), day(4:end)])``)."""
    return int(day[:2] + day[3:])


def pick_dated_file(calib_dir, pattern, day):
    """``fBellaCalibPathV01``: newest ``YYMMDD<...>`` file matching
    ``pattern`` dated on or before ``day``. Returns (path, yymmdd)."""
    names = sorted(n for n in os.listdir(calib_dir) if fnmatch.fnmatchcase(n, pattern))
    dates = np.array([int(n[:6]) for n in names])
    k = np.flatnonzero(dates <= day_number(day))
    if k.size == 0:
        raise FileNotFoundError(f'no {pattern} in {calib_dir} dated <= {day}')
    k = k[-1]
    return os.path.join(calib_dir, names[k]), int(dates[k])


@dataclass
class CamCalib:
    """One row of ``*camCalib.txt`` (``fBellaCamCalibV02``). ROI indices are
    kept 1-based and inclusive, exactly as in the file."""
    fov: float
    yOffset: int
    height: int
    xOffset: int
    width: int
    leftPos: float
    yCntr: float
    ySt: int
    xSt: int
    yEd: int
    xEd: int
    rot: float
    sns: float
    setN: int


def load_cam_calib(path):
    df = read_log(path)
    cams = []
    for i in range(len(df)):
        vals = {}
        for key, col in _CAM_FIELDS.items():
            v = log_column(df, col)[i]
            vals[key] = int(v) if key in _INT_FIELDS else float(v)
        cams.append(CamCalib(**vals))
    return cams


def lanex_c2c_vignette(cam, lanex_row, screen='front'):
    """``fLanexClbV02``: counts->fC factor (for 1 GeV) and vignette
    compensation matrix over the analysis ROI."""
    screen_f = 1.0 if screen == 'back' else 1.98
    z = (cam.fov - lanex_row['FOV offset']) / lanex_row['FOV slope']
    als_r = lanex_row['sensitivity 2'] * z ** 2 + lanex_row['sensitivity 1'] * z + lanex_row['sensitivity 0']
    c2c = screen_f * als_r / 146

    w, h = int(lanex_row['full width']), int(lanex_row['full height'])
    aaa, bbb = np.meshgrid(np.arange(1, w + 1, dtype=float), np.arange(1, h + 1, dtype=float))
    aaa = aaa - w / 2 + 0.5
    bbb = bbb - h / 2 + 0.5
    r = np.sqrt(aaa ** 2 + bbb ** 2)
    r = r[cam.yOffset:cam.yOffset + cam.height, cam.xOffset:cam.xOffset + cam.width]
    r = r[cam.ySt - 1:cam.yEd, cam.xSt - 1:cam.xEd]
    v = lanex_row['vignette 4'] * r ** 4 + lanex_row['vignette 2'] * r ** 2 + lanex_row['vignette 0']
    return c2c, 1.0 / v


def load_lanex_table(path):
    df = read_log(path)
    cols = ['FOV slope', 'FOV offset', 'vignette 4', 'vignette 2', 'vignette 0',
            'sensitivity 2', 'sensitivity 1', 'sensitivity 0', 'full width', 'full height']
    return {c: log_column(df, c) for c in cols}


def lanex_row(table, set_n):
    """Row ``setN`` (1-based) of the lanex table (``fLanexClbOutV01``)."""
    return {k: float(v[set_n - 1]) for k, v in table.items()}


@dataclass
class TrajectoryAngleCalib:
    """``fBellaTrjCalibAngle`` output: tables indexed [angle, momentum]."""
    mmt: np.ndarray        # normalised momentum [MeV/c/T]
    angl: np.ndarray       # input angle [mrad]
    lgc: np.ndarray        # side logic
    sScrn: np.ndarray      # side screen [mm]
    fScrn: np.ndarray      # front screen [mm]
    bndAnglS: np.ndarray
    bndAnglF: np.ndarray
    tPath: np.ndarray
    xDivF: np.ndarray
    yDivF: np.ndarray
    rsl: np.ndarray


def load_trajectory_angle_calib(calib_dir, day, crrnt=400):
    _, day_out = pick_dated_file(calib_dir, f'*trjCalib{crrnt}A0.txt', day)
    pattern = f'{day_out}trjCalib{crrnt}A*.txt'
    names = sorted(n for n in os.listdir(calib_dir) if fnmatch.fnmatchcase(n, pattern))
    names = names[:-1]  # MATLAB drops the last one (the 'Geecs' file)
    # fGetPartStrV03(name, 'A', '.'): text between the 'A' and the '.'
    angles = np.array([float(n[n.index('A') + 1:n.index('.')]) for n in names])
    order = np.argsort(angles, kind='stable')
    first = read_log(os.path.join(calib_dir, names[0]))
    mmt = log_column(first, 'momentum [MeV/c]')
    cols = {
        'lgc': 'side logic', 'sScrn': 'side screen [m]', 'fScrn': 'front screen [m]',
        'bndAnglS': 'bending angle at side screen [dgr]',
        'bndAnglF': 'bending angle at front screen [dgr]',
        'tPath': 'total path [m]', 'xDivF': 'x conv fct rms', 'yDivF': 'y conv fct rms',
        'rsl': 'momentum rsl [%/mrad]',
    }
    tables = {k: np.zeros((len(names), mmt.size)) for k in cols}
    for jj, ii in enumerate(order):
        df = read_log(os.path.join(calib_dir, names[ii]))
        for k, c in cols.items():
            tables[k][jj] = log_column(df, c)
    tables['sScrn'] *= 1000
    tables['fScrn'] *= 1000
    return TrajectoryAngleCalib(mmt=mmt, angl=angles[order] / 10, **tables)


@dataclass
class ScreenTrajectory:
    """Front- or side-screen trajectory calibration at one input angle."""
    mmt: np.ndarray
    screen: np.ndarray
    incAgl: np.ndarray
    path: np.ndarray
    divFX: np.ndarray
    divFY: np.ndarray
    rsl: np.ndarray


def trajectory_at_angle(trj, angl):
    """``fBellaTrjCalibFSAngl``: interpolate the angle tables at ``angl``
    [mrad] and split into (front, side) calibrations."""
    def at(tab):
        return interp1(trj.angl, tab, np.array([angl]))[0]
    side_l = mround(at(trj.lgc))
    s_scrn, f_scrn = at(trj.sScrn), at(trj.fScrn)
    inc_s, inc_f = at(trj.bndAnglS), at(trj.bndAnglF)
    path, div_x, div_y, rsl = at(trj.tPath), at(trj.xDivF), at(trj.yDivF), at(trj.rsl)
    k = find_first(side_l == 0)
    side = ScreenTrajectory(trj.mmt[:k], s_scrn[:k], inc_s[:k], path[:k],
                            div_x[:k], div_y[:k], rsl[:k])
    front = ScreenTrajectory(trj.mmt[k:], f_scrn[k:], inc_f[k:], path[k:],
                             div_x[k:], div_y[k:], rsl[k:])
    return front, side


@dataclass
class ResolutionCalib:
    """400A0 trajectory table used for the momentum resolution in
    ``fBellaSShotTri`` (nMmt in GeV/c/T)."""
    nMmt: np.ndarray
    xConv: np.ndarray
    path: np.ndarray
    rsl: np.ndarray


class MagSpecCalibration:
    """All static calibration for one experiment day.

    Parameters
    ----------
    calib_dir : str
        The ``Calibrations/ESMCalib`` directory.
    day : str
        Experiment day string, e.g. ``'26_0521'``; selects the newest
        dated calibration files on or before that day.
    crrnt : int
        Nominal magnet current used to pick the trajectory tables.
    mgs_chg : float
        Charge correction applied to the bottom-screen cameras (4-10).
    lanex_file : str
        Lanex calibration file name inside ``calib_dir``.
    """

    def __init__(self, calib_dir, day, crrnt=400, mgs_chg=0.64,
                 lanex_file='200301lanexCalib.txt'):
        self.calib_dir = calib_dir
        self.day = day
        self.crrnt = crrnt

        self.cam_calib_path, _ = pick_dated_file(calib_dir, '*camCalib.txt', day)
        all_cams = load_cam_calib(self.cam_calib_path)
        self.cams = all_cams[:len(MAGSPEC_CAMERAS)]

        self.lanex_path = os.path.join(calib_dir, lanex_file)
        lanex = load_lanex_table(self.lanex_path)
        c2c, vgnt = [], []
        for cam in all_cams:
            c, v = lanex_c2c_vignette(cam, lanex_row(lanex, cam.setN))
            c2c.append(c)
            vgnt.append(v)
        c2c = np.array(c2c)
        c2c[3:-1] *= mgs_chg  # c2c(4:end-1) in MATLAB (11 rows in the file)
        self.c2c = c2c[:len(MAGSPEC_CAMERAS)]
        self.vignette = vgnt[:len(MAGSPEC_CAMERAS)]

        # full-width vignette of camera A for the x-ray (frontSL) lineout
        cam_x = CamCalib(**{**self.cams[0].__dict__, 'xSt': 1, 'xEd': self.cams[0].width})
        _, self.vignette_xray = lanex_c2c_vignette(cam_x, lanex_row(lanex, cam_x.setN))

        self.trajectory = load_trajectory_angle_calib(calib_dir, day, crrnt)

        path0, _ = pick_dated_file(calib_dir, f'*trjCalib{crrnt}A0.txt', day)
        df = read_log(path0)
        self.resolution = ResolutionCalib(
            nMmt=0.001 * log_column(df, 'momentum [MeV/c]'),
            xConv=log_column(df, 'x conv fct fwhm'),
            path=log_column(df, 'total path [m]'),
            rsl=log_column(df, 'momentum rsl [%/mrad]'),
        )

    def trajectory_at_angle(self, angl):
        return trajectory_at_angle(self.trajectory, angl)
