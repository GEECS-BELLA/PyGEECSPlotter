# Per-camera axes and the stitched momentum / angle axes.
# Ports fBellaAxisTri, fBellaAxisAllV04, fBellaAnglMapV01, fBellaUaYV02 and
# fBellaUmXV03. Momenta here are *normalised* (MeV/c per tesla); the field
# is applied later.

from dataclasses import dataclass, field

import numpy as np

from PyGEECSPlotter.magspec.calibration import CamCalib
from PyGEECSPlotter.magspec.matlab_compat import interp1


@dataclass
class CamXAxis:
    pixel: np.ndarray
    mm: np.ndarray
    dx: float
    incAgl: np.ndarray
    path: np.ndarray
    divFY: np.ndarray
    accp: np.ndarray
    mmt: np.ndarray
    dp: np.ndarray
    rsl: np.ndarray
    dsp: np.ndarray
    # filled by momentum_bins
    dpB: np.ndarray = field(default=None)
    bin: np.ndarray = field(default=None)
    mmtB: np.ndarray = field(default=None)


@dataclass
class CamYAxis:
    pixel: np.ndarray
    dy: float
    mm: np.ndarray


def axis_tri(cam, trj, accp, sign_s):
    """``fBellaAxisTri``: x (dispersion) and y axes for one camera."""
    xx = np.arange(1, cam.width + 1, dtype=float)
    pixel = xx[cam.xSt - 1:cam.xEd]
    mm_full = np.linspace(cam.leftPos, cam.leftPos + sign_s * cam.fov, cam.width)
    mm = mm_full[cam.xSt - 1:cam.xEd]
    dx = mm[1] - mm[0]
    inc = interp1(trj.screen, trj.incAgl, mm, 'cubic')
    path = interp1(trj.screen, trj.path, mm, 'cubic')
    div_fy = interp1(trj.screen, trj.divFY, mm, 'cubic')
    acc = 0.5 * accp / (path * div_fy)
    mmt = interp1(trj.screen, trj.mmt, mm, 'cubic')
    d1 = np.diff(mmt)
    dp = np.abs(0.5 * (np.r_[d1, d1[-1]] + np.r_[d1[0], d1]))
    rsl = interp1(trj.screen, trj.rsl, mm, 'cubic')
    dsp = dp / dx
    x = CamXAxis(pixel, mm, dx, inc, path, div_fy, acc, mmt, dp, rsl, dsp)

    yy = np.arange(1, cam.height + 1, dtype=float)
    ypix = yy[cam.ySt - 1:cam.yEd]
    ymm = ypix * dx - dx * cam.yCntr
    return x, CamYAxis(ypix, dx, ymm)


def axis_all(cams, trj_front, trj_side, accp=(33, 40)):
    """``fBellaAxisAllV04``: cameras 1-3 on the front screen, 4-10 on the
    side (bottom) screen, plus the full-width camera-1 axis for x-ray."""
    xs, ys = [], []
    for ii, cam in enumerate(cams):
        if ii < 3:
            x, y = axis_tri(cam, trj_front, accp[0], 1)
        else:
            x, y = axis_tri(cam, trj_side, accp[1], -1)
        xs.append(x)
        ys.append(y)
    cam_x = CamCalib(**{**cams[0].__dict__, 'xSt': 1, 'xEd': cams[0].width})
    x_xr, _ = axis_tri(cam_x, trj_front, accp[0], 1)
    return xs, ys, x_xr


def angle_maps(xs, ys):
    """``fBellaAnglMapV01``: per-camera angle map [mrad] and its row step."""
    angl_c, dangl_c = [], []
    for x, y in zip(xs, ys):
        path, ymm = np.meshgrid(x.path, y.mm)
        div_fy, _ = np.meshgrid(x.divFY, y.mm)
        ymm = ymm / div_fy
        ang = 1000 * np.arctan(0.001 * ymm / path)
        da = np.diff(ang, axis=0)
        da = 0.5 * (np.vstack([da[:1], da]) + np.vstack([da, da[-1:]]))
        angl_c.append(ang)
        dangl_c.append(da)
    return angl_c, dangl_c


@dataclass
class AngleAxis:
    angl: np.ndarray
    da: float


def uniform_angle_axis(agl_rsl=256, edges=(-1.3, 1.3)):
    """``fBellaUaYV02`` (fixed edges). MATLAB builds ``edgA(1):da:edgA(2)``;
    the colon operator is reproduced (not linspace) to keep the last point
    identical."""
    da = (edges[1] - edges[0]) / (agl_rsl - 1)
    n = int(np.floor((edges[1] - edges[0]) / da + 1e-10)) + 1
    # MATLAB colon: symmetric computation from both ends
    k = np.arange(n)
    ang = np.where(k < n / 2, edges[0] + k * da, edges[1] - (n - 1 - k) * da)
    return AngleAxis(ang, da)


@dataclass
class WindowAxis:
    mmt: np.ndarray
    dp: float
    accp: np.ndarray
    incAgl: np.ndarray
    dsp: np.ndarray


def _window(xs, idx, mmt_rsl):
    hi = xs[idx[0]].mmt[0]
    lo = xs[idx[-1]].mmt[-1]
    mmt = np.linspace(lo, hi, mmt_rsl)
    cat = lambda attr: np.concatenate([getattr(xs[i], attr) for i in idx])
    xm = cat('mmt')
    return WindowAxis(mmt=mmt, dp=mmt[1] - mmt[0],
                      accp=interp1(xm, cat('accp'), mmt),
                      incAgl=interp1(xm, cat('incAgl'), mmt),
                      dsp=interp1(xm, cat('dsp'), mmt))


def momentum_bins(xs, cams, mmt_rsl=(1024, 1024)):
    """``fBellaUmXV03``: stitched window axes and the per-camera pixel
    binning that makes each bin's dp just under the window's uniform dp.
    Mutates ``xs`` (adds ``dpB``, ``bin``, ``mmtB``) and returns the two
    window axes."""
    windows = [_window(xs, [0, 1, 2], mmt_rsl[0]), _window(xs, list(range(3, 10)), mmt_rsl[1])]
    for indx in range(10):
        wdp = windows[0 if indx < 3 else 1].dp
        width = cams[indx].xEd - cams[indx].xSt + 1
        dp = xs[indx].dp
        dp_a = np.zeros(width + 2)
        bin_a = np.zeros(width + 2, dtype=int)
        j = 0
        for i in range(width):
            dp_a[j] += dp[i]
            bin_a[j] += 1
            if dp_a[j] / wdp > 1:
                if bin_a[j] > 1:
                    dp_a[j] -= dp[i]
                    dp_a[j + 1] = dp[i]
                    bin_a[j] -= 1
                    bin_a[j + 1] = 1
                j += 1
        zeros = np.flatnonzero(bin_a == 0)
        pp = int(zeros[0]) if zeros.size else None
        x = xs[indx]
        if pp is None:
            x.dpB, x.bin, x.mmtB = dp.copy(), bin_a[:width].copy(), x.mmt.copy()
        else:
            x.dpB, x.bin = dp_a[:pp].copy(), bin_a[:pp].copy()
            mmt_b = np.empty(pp)
            i = 0
            for k in range(pp):
                mmt_b[k] = np.mean(x.mmt[i:i + x.bin[k]])
                i += x.bin[k]
            x.mmtB = mmt_b
    return windows
