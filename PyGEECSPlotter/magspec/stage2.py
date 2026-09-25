# Stage 2 of the triangle-chamber magspec analysis (fBellaSShotTri.m,
# sMode = 1): highE + lowE + frontSL -> x-ray-background-subtracted,
# stitched full-range "allE" image, its spectrum / divergence lineouts and
# the scan scalars. Figure-only parts of the MATLAB (lanex / EBeam-profile
# panels, infoE, logE, lnrE, maxMmt figures) are not ported.

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from PyGEECSPlotter.magspec.image_proc import (get_fwhm, get_rms, low_pass_line,
                                               smooth_array, xray_out)
from PyGEECSPlotter.magspec.matlab_compat import interp1, nearest_index

MAX_E_THRESHOLD = 1e6    # maxETh [fC^3]
XRAY_FIT_START_MM = 60   # brd1
XRAY_FIT_END_MM = 170    # brd4
XRAY_ROI_END_MM = 218    # brd5
ALLE_POINTS = 2048


@dataclass
class ProcessedWindow:
    """One stage-1 window as stage 2 sees it (``fBellaReadPrcImgV01``):
    image [aC] and the axis columns of the ``*Spec`` text file."""
    img: np.ndarray
    mmt: np.ndarray      # [GeV/c]
    nMmt: np.ndarray     # [GeV/c/T]
    accp: np.ndarray     # [mrad]
    dsp: np.ndarray      # [MeV/mm] at the shot's field


def _xray_model(x, a, b, c, d):
    return a * np.exp(-b * np.abs(x - d) ** c)


def xray_base(front_mm, front_sig, front_mmt):
    """frontSL fit ``a*exp(-b*|x-d|^c)`` (fig4 block of fBellaSShotTri).
    Returns (xRayBI [fC/mm], xRayB [fC/mm^2], fit parameters)."""
    sig = smooth_array(front_sig, 3)
    i1 = nearest_index(front_mm, XRAY_FIT_START_MM)
    i4 = nearest_index(front_mm, XRAY_FIT_END_MM)
    if i1 >= i4:
        i4 = i1 + 50
    i5 = nearest_index(front_mm, XRAY_ROI_END_MM)
    x, y = front_mm[i1:i4 + 1], sig[i1:i4 + 1]
    ok = np.isfinite(x) & np.isfinite(y)   # prepareCurveData drops NaN/Inf
    lo = [sig[i1] / 2, 0, 0, front_mm[i1] - 1]
    p0 = [sig[i1], 0.001, 0.5357, front_mm[i1]]
    hi = [sig[i1] * 1.5, 5, 5, front_mm[i1] + 1]
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    p0 = np.clip(p0, lo, hi)
    try:
        popt, _ = curve_fit(_xray_model, x[ok], y[ok], p0=p0, bounds=(lo, hi),
                            method='trf', maxfev=20000)
    except (RuntimeError, ValueError):
        popt = p0
    bi = _xray_model(front_mm[i5], *popt)
    return bi, bi / (163 * 0.2), popt


def _find_last(mask):
    idx = np.flatnonzero(mask)
    return int(idx[-1]) if idx.size else None


@dataclass
class Stage2Result:
    alle: np.ndarray            # allE image [aC], n_angle x 2048
    alle_mmt: np.ndarray        # uniform momentum axis of allE [GeV/c]
    angle: np.ndarray           # [mrad]
    spec: pd.DataFrame          # allESpec table
    div: pd.DataFrame           # allEDiv table
    density: np.ndarray         # ROI image [pC/mrad/(GeV/c)] on mmtR
    mmtR: np.ndarray            # ROI momentum axis [GeV/c] (non-uniform)
    accpR: np.ndarray           # acceptance on mmtR [mrad]
    scalars: dict


def run_stage2(high, low, angle, front_mm, front_sig, front_mmt, resolution, roi=(0.01, 5.0)):
    """Port of ``fBellaSShotTri`` (sMode = 1, roi given).

    ``high``/``low`` are :class:`ProcessedWindow` (low not yet flipped),
    ``angle`` the uniform angle axis [mrad], ``front_*`` the frontSLX
    columns, ``resolution`` a :class:`ResolutionCalib`.
    """
    img1 = high.img.astype(float).copy()
    img2 = np.flipud(low.img.astype(float))
    ya = np.asarray(angle, float)
    da = ya[1] - ya[0]
    mmt = np.r_[low.mmt, high.mmt]
    n_mmt = np.r_[low.nMmt, high.nMmt]
    accp = np.r_[low.accp, high.accp]
    dp1 = high.mmt[1] - high.mmt[0]
    dp2 = low.mmt[1] - low.mmt[0]
    dx_mm = 1000 * dp1 / high.dsp
    dx_mm = smooth_array(low_pass_line(dx_mm, 1.04, 0.001, 1), 5)

    idx1 = nearest_index(mmt, roi[0])
    idx2 = nearest_index(mmt, roi[1])
    mmt_r = mmt[idx1:idx2 + 1]
    n_mmt_r = n_mmt[idx1:idx2 + 1]
    accp_r = accp[idx1:idx2 + 1]

    # x-ray background from the front-screen space-linear lineout
    xray_bi, xray_b, _ = xray_base(front_mm, front_sig, front_mmt)
    acp_cmp = high.accp / ya[-1]
    xray_m = np.tile(dx_mm / acp_cmp, (ya.size, 1))
    xray_m = 1000 * xray_m * xray_bi / ya.size        # [aC]
    img1 = img1 - xray_m
    img1 = img1 * (img1 >= 0)

    img = np.hstack([img2, img1])                       # [aC]
    img_r = img[:, idx1:idx2 + 1]
    img_r, _ = xray_out(img_r, (1, 2, 10, 6))

    spc_rt = np.sum(img_r ** 3, axis=0)
    kk = _find_last(spc_rt > MAX_E_THRESHOLD)
    max_eng = 0.0 if kk is None else mmt_r[kk]

    div_r = 1e-6 * img_r.sum(axis=1) / da               # [pC/mrad]
    chg = 1e-6 * img_r.sum()
    eng = 1e-6 * np.sum(img_r.sum(axis=0) * mmt_r)
    i6 = nearest_index(mmt_r, 9)
    i8 = nearest_index(mmt_r, 11)
    chg68 = 1e-6 * img_r[:, i6:i8 + 1].sum()

    img = np.hstack([img2 / dp2, img1 / dp1])           # [aC/GeV] (pre-filter, as MATLAB)
    img_r = img[:, idx1:idx2 + 1]
    spc_r = 1e-6 * img_r.sum(axis=0)                    # [pC/GeV]
    img_r = 1e-6 * img_r / da                           # [pC/mrad/(GeV/c)]
    cd68 = np.max(img_r[:, i6:i8 + 1])

    # allE: uniform momentum, renormalised to the ROI charge [aC]
    uni_mmt = np.linspace(mmt_r[0], mmt_r[-1], ALLE_POINTS)
    uni = interp1(mmt_r, img_r.T, uni_mmt).T
    uni = 1e6 * uni * chg / np.nansum(uni)

    spec = pd.DataFrame({
        'Momentum_GeV/c': uni_mmt,
        'Charge_fC': 0.001 * np.nansum(uni, axis=0),
    })
    spec['ChargeDen_pC/GeV/c'] = 1e-3 * spec['Charge_fC'] / (uni_mmt[1] - uni_mmt[0])
    div = pd.DataFrame({'Angle_mrad': ya, 'Charge_fC': 0.001 * np.nansum(uni, axis=1)})
    div['ChargeDen_fC/mrad'] = div['Charge_fC'] / (ya[1] - ya[0])

    std_agl, mean_agl = get_rms(ya, div_r)
    fwhm_agl, _, _, _, ipa = get_fwhm(ya, div_r)
    std_mmt, mean_mmt = get_rms(mmt_r, spc_r)
    fwhm_mmt, _, _, _, ipm = get_fwhm(mmt_r, spc_r)
    _, _, _, _, ipe = get_fwhm(mmt_r, spc_r * mmt_r)

    fwhm_l, fwhm_lo, _, _, _ = get_fwhm(ya, img_r[:, ipm])
    fwhm_l = 0.5 * (fwhm_lo + fwhm_l)
    res = resolution
    at = np.array([n_mmt_r[ipm]])
    x_conv = interp1(res.nMmt, res.xConv, at)[0]
    path = interp1(res.nMmt, res.path, at)[0]
    x_size = fwhm_l * path * x_conv
    with np.errstate(divide='ignore'):   # fwhm 0 -> Inf, as in MATLAB
        x_size_f = 0.2 / x_size
    # MATLAB: xSizeF = xSizeF<1 + xSizeF*(xSizeF>=1) parses as
    # xSizeF < (1 + ...), a logical -- reproduced as written.
    x_size_f = float(x_size_f < 1 + x_size_f * (x_size_f >= 1))
    mmt_res = interp1(res.nMmt, res.rsl, at)[0] * fwhm_l * x_size_f

    scalars = {
        'charge_pC': chg, 'energy_mJ': eng,
        'meanMomentum_GeV/c': mean_mmt, 'peakMomentum_GeV/c': mmt_r[ipm],
        'stdMomentum_GeV/c': std_mmt, 'fwhmMomentum_GeV/c': fwhm_mmt,
        'meanAngle_mrad': mean_agl, 'peakAngle_mrad': ya[ipa],
        'stdAngle_mrad': std_agl, 'fwhmAngle_mrad': fwhm_agl,
        'mmtRes_%': mmt_res, 'maxMomentum_GeV/c': max_eng,
        'energyPeakMmt_GeV/c': mmt_r[ipe],
        'charge6to8GeV_pC': chg68, 'pkChrDen6to8GeV_pC': cd68,
        'xRayBase_fC/mm^2': xray_b, 'xRayBase_fC/mm': xray_bi,
    }
    return Stage2Result(uni, uni_mmt, ya, spec, div, img_r, mmt_r, accp_r, scalars)
