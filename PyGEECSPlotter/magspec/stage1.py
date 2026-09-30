# Stage 1 of the triangle-chamber magspec analysis (bellaMagspcTri.m):
# ten raw camera frames -> charge-calibrated, angle/momentum-uniform
# "highE" (front screen, cams A-C) and "lowE" (side screen, cams D, G-L)
# images [aC], plus the camera-A front space-linear lineout used for the
# x-ray background estimate. Ports fBellaImgV02, fBellaUaV01,
# fBellaMmtBinV01, fBellaUmV01, fBellaUamCmbV03, fBellaXrayLineTri and
# fBellaInAngl2.

from dataclasses import dataclass

import numpy as np

from PyGEECSPlotter.magspec.axes import (angle_maps, axis_all, momentum_bins,
                                         uniform_angle_axis)
from PyGEECSPlotter.magspec.image_proc import rotate_image, tony_bg_subtract, xray_out
from PyGEECSPlotter.magspec.matlab_compat import interp1

LOW_PASS = (2, 1, 1, 3)          # fctX1, pitX1, minX1, itr1
ACCEPTANCE_MM = (33, 40)         # window full width, front and side [mm]
ANGLE_RESOLUTION = 256
MOMENTUM_RESOLUTION = (1024, 1024)


def ebeam_y_angle(context, max_angle=1.8, max_div=1.32, max_mean_peak=0.35):
    """``fBellaInAngl2``: e-beam vertical input angle [mrad] from the
    EBeam-profile scalars, or 0 if any quality cut fails."""
    def g(key):
        return float(context[key])
    xa, ya = g('EBeamPrf peak angle x [mrad]'), g('EBeamPrf peak angle y [mrad]')
    xm, ym = g('EBeamPrf mean angle x [mrad]'), g('EBeamPrf mean angle y [mrad]')
    xs, ys = g('EBeamPrf std div x [mrad]'), g('EBeamPrf std div y [mrad]')
    xf, yf = g('EBeamPrf fwhm div x [mrad]'), g('EBeamPrf fwhm div y [mrad]')
    ok = (abs(xa) < max_angle and abs(ya) < max_angle and abs(xm) < max_angle
          and abs(ym) < max_angle and xs < max_div and ys < max_div
          and xf < 2 and xf != 0 and yf < 2 and yf != 0
          and abs(xa - xm) < max_mean_peak and abs(ya - ym) < max_mean_peak)
    return ya if ok else 0.0


def process_camera_images(raw, bg, calib, xs):
    """``fBellaImgV02``. ``raw``/``bg``: lists of 12-bit frames (``raw[i]``
    may be None for a missing file). Returns (images [fC], index of the
    last missing camera (1-based, 0 if none), saturated-pixel counts,
    full-frame bg-subtracted rotated camera-A image or None)."""
    imgs, sat = [], np.zeros(len(calib.cams), dtype=int)
    missing, img_x = 0, None
    for i, cam in enumerate(calib.cams):
        if raw[i] is not None:
            r = np.asarray(raw[i], float)
            sat[i] = int(np.sum(r == 4095))
            r = tony_bg_subtract(r, bg[i])
            r = rotate_image(r, cam.rot, (cam.width / 2, cam.height / 2))
            if i == 0:
                img_x = r
        else:
            # MATLAB feeds the background frame itself through the chain
            missing = i + 1
            r = np.asarray(bg[i], float)
        img = r[cam.ySt - 1:cam.yEd, cam.xSt - 1:cam.xEd]
        img, _ = xray_out(img, LOW_PASS)
        img = img * calib.vignette[i]
        cos_inc = np.cos(np.pi * xs[i].incAgl / 180)[None, :]
        img = calib.c2c[i] * img * cos_inc
        imgs.append(img * cam.sns)
    return imgs, missing, sat, img_x


def uniform_angle(img, angl, dangl, y):
    """``fBellaUaV01``: resample each column from its angle map onto the
    uniform angle axis ``y``, conserving each column's charge. MATLAB does
    this as one 1-D interpolation over column-stacked, biased angles."""
    da = y[1] - y[0]
    szy, szx = img.shape
    sum_o = img.sum(axis=0)
    img = img / dangl
    img[0, :] = 0
    img[-1, :] = 0
    pad = np.zeros((1, szx))
    img = np.vstack([pad, img, pad])
    angl = np.vstack([pad + np.min(y) - da, angl, pad + np.max(y) + da])
    bias_s = (np.max(y) + da) - (np.min(y) - da) + da
    col_bias = bias_s * np.arange(szx)[None, :]
    angl = angl + col_bias
    u = y[:, None] + col_bias
    out = interp1(angl.ravel(order='F'), img.ravel(order='F'), u.ravel(order='F'))
    out = out.reshape((y.size, szx), order='F') * da
    sum_o = sum_o + (sum_o == 0)
    sum_t = out.sum(axis=0)
    sum_t = sum_t + (sum_t == 0)
    return out * (sum_o / sum_t)[None, :]


def momentum_bin(img, bins):
    """``fBellaMmtBinV01``: sum adjacent columns into momentum bins."""
    edges = np.r_[0, np.cumsum(bins)]
    return np.stack([img[:, edges[k]:edges[k + 1]].sum(axis=1)
                     for k in range(len(bins))], axis=1)


def uniform_momentum(img, mmt_b, dp_b, window):
    """``fBellaUmV01``: dp-normalise and interpolate each row onto the
    window's uniform momentum axis."""
    img = img / np.asarray(dp_b)[None, :]
    out = interp1(mmt_b, img.T, window.mmt)  # (n_mmt, n_rows)
    return window.dp * out.T


def combine_window(imgs, angl_c, dangl_c, xs, window, y, idx):
    """One window of ``fBellaUamCmbV03`` -> uniform image [aC]."""
    parts = [momentum_bin(uniform_angle(imgs[i], angl_c[i], dangl_c[i], y), xs[i].bin)
             for i in idx]
    img = np.hstack(parts)
    dp_b = np.concatenate([xs[i].dpB for i in idx])
    mmt_b = np.concatenate([xs[i].mmtB for i in idx])
    out = 1000 * uniform_momentum(img, mmt_b, dp_b, window)
    out[:, 0] = 0
    out[:, -1] = 0
    return out


def xray_lineout(img_x, calib, x_xr, y0, fld):
    """``fBellaXrayLineTri``: camera-A full-width lineouts for the x-ray
    background. Returns (image [fC] flipped up-down, outX 4xN
    [mm, fC, fC/mm, MeV/c], outY 3xM [mm, fC, fC/mm])."""
    cam = calib.cams[0]
    dx_out = x_xr.mm[1] - x_xr.mm[0]
    if img_x is not None:
        im = img_x[cam.ySt - 1:cam.yEd, :]
        for prm in ((2, 3, 1, 2), (2, 2, 1, 2), (2, 1, 1, 2)):
            im, _ = xray_out(im, prm)
        im = im * calib.vignette_xray
        im = cam.sns * calib.c2c[0] * im
        chg, chg_y = im.sum(axis=0), im.sum(axis=1)
        lin_mev = fld * x_xr.mmt
    else:
        im = np.zeros(1)
        chg = np.zeros_like(x_xr.mm)
        chg_y = np.zeros_like(y0.mm)
        lin_mev = np.zeros_like(x_xr.mm)
    im = np.flipud(np.atleast_2d(im))
    out_x = np.vstack([x_xr.mm, chg, chg / dx_out, lin_mev])
    out_y = np.vstack([y0.mm, chg_y, chg_y / dx_out])
    return im, out_x, out_y


@dataclass
class Stage1Result:
    high: np.ndarray        # highE image [aC], 256 x 1024
    low: np.ndarray         # lowE image [aC], 256 x 1024 (not yet flipped)
    windows: list           # WindowAxis for high (0) and low (1), normalised mmt
    angle: object           # AngleAxis
    field: float            # [T]
    ey_angle: float         # [mrad]
    front_x: np.ndarray     # frontSLX table (4 x N)
    front_y: np.ndarray     # frontSLY table (3 x M)
    front_img: np.ndarray   # frontSL image [fC]
    cam_charge_pC: np.ndarray
    saturated: np.ndarray
    missing: int


def run_stage1(raw, bg, calib, field_T, ey_angle):
    """Full stage 1 for one shot (the shot loop body of bellaMagspcTri)."""
    trj_f, trj_s = calib.trajectory_at_angle(ey_angle)
    xs, ys, x_xr = axis_all(calib.cams, trj_f, trj_s, ACCEPTANCE_MM)
    angl_c, dangl_c = angle_maps(xs, ys)
    y_axis = uniform_angle_axis(ANGLE_RESOLUTION)
    windows = momentum_bins(xs, calib.cams, MOMENTUM_RESOLUTION)
    imgs, missing, sat, img_x = process_camera_images(raw, bg, calib, xs)
    high = combine_window(imgs, angl_c, dangl_c, xs, windows[0], y_axis.angl, [0, 1, 2])
    low = combine_window(imgs, angl_c, dangl_c, xs, windows[1], y_axis.angl, list(range(3, 10)))
    front_img, fx, fy = xray_lineout(img_x, calib, x_xr, ys[0], field_T)
    chg = np.array([1e-3 * np.nansum(im) for im in imgs])
    return Stage1Result(high, low, windows, y_axis, field_T, ey_angle, fx, fy,
                        front_img, chg, sat, missing)
