# Image / line processing primitives from the magspec MATLAB code.
# Ports fTrexTonyBgV01, fImageRotV02, fXrayOutV10, fSmoothAryV01,
# fLowPassLineV01, fGetRmsV01 and fGetFwhmV04.

import numpy as np

from PyGEECSPlotter.magspec.matlab_compat import find_first, find_last, interp2, mround


def tony_bg_subtract(img, bg):
    """``fTrexTonyBgV01``: subtract ``bg``, then iteratively spread the
    negative residual over the positive pixels (max 100 iterations)."""
    img = np.asarray(img, float)
    bg = np.asarray(bg, float)
    if bg.sum() > img.sum():
        return np.zeros_like(img)
    a = img - bg
    neg = a < 0
    pos = a > 0
    ngt = -np.sum(a * neg)
    a = a * pos
    kk = 1
    while ngt > 1 and kk < 100:
        nmb = np.sum(pos)
        a = a - pos * (ngt / nmb)
        neg = a < 0
        pos = a > 0
        ngt = -np.sum(a * neg)
        a = a * pos
        kk += 1
    if kk == 100:
        return np.zeros_like(img)
    return a


def rotate_image(img, rot_deg, pvt=None):
    """``fImageRotV02``: rotate about pivot pixel ``pvt`` = (x, y), 1-based,
    by bilinear resampling. Pixels mapping outside the (zero-padded) frame
    are NaN, as with MATLAB ``interp2``."""
    img = np.asarray(img, float)
    szy, szx = img.shape
    if pvt is None:
        pvt = (mround(0.5 * szx), mround(0.5 * szy))
    x_loc, y_loc = np.meshgrid(np.arange(1, szx + 1) - pvt[0], np.arange(1, szy + 1) - pvt[1])
    pad = np.zeros((2 * szy, 2 * szx))
    init_x, init_y = int(mround(0.5 * szx)), int(mround(0.5 * szy))
    pad[init_y - 1:init_y - 1 + szy, init_x - 1:init_x - 1 + szx] = img
    xp = np.arange(1, 2 * szx + 1) - (pvt[0] + init_x - 1)
    yp = np.arange(1, 2 * szy + 1) - (pvt[1] + init_y - 1)
    r = rot_deg * np.pi / 180
    xt = x_loc * np.cos(r) - y_loc * np.sin(r)
    yt = x_loc * np.sin(r) + y_loc * np.cos(r)
    return interp2(xp, yp, pad, xt, yt)


def xray_out(img, prm):
    """``fXrayOutV10``: iterative hot-pixel (x-ray hit) replacement by the
    4-neighbour average. ``prm = [fct, pit, minX, itr]``. Returns
    (image, removed counts)."""
    fct, pit, min_x, itr = prm
    pit, itr = int(pit), int(itr)
    img = np.asarray(img, float)
    cnt = img.sum()
    szy, szx = img.shape
    # padded frame: the image in the centre, its first/last ``pit`` rows and
    # columns copied outward (not mirrored), zero corners -- as MATLAB's
    # [crn,edgT,crn; edgL,img,edgR; crn,edgB,crn]
    pd_img = np.zeros((szy + 2 * pit, szx + 2 * pit))
    core = pd_img[pit:pit + szy, pit:pit + szx]

    def refresh_edges():
        pd_img[:pit, pit:pit + szx] = core[:pit, :]
        pd_img[pit + szy:, pit:pit + szx] = core[-pit:, :]
        pd_img[pit:pit + szy, :pit] = core[:, :pit]
        pd_img[pit:pit + szy, pit + szx:] = core[:, -pit:]

    core[...] = img
    refresh_edges()
    valid = img >= min_x
    # neighbour views (top, bottom, left, right) and reusable work arrays
    views = (pd_img[:szy, pit:pit + szx], pd_img[2 * pit:2 * pit + szy, pit:pit + szx],
             pd_img[pit:pit + szy, :szx], pd_img[pit:pit + szy, 2 * pit:2 * pit + szx])
    ref = np.empty((szy, szx))
    tmp = np.empty((szy, szx))
    hit = np.empty((szy, szx), dtype=bool)
    if fct != 0:
        for _ in range(itr):
            # ref = 0.25*top + 0.25*bottom + 0.25*left + 0.25*right, summed
            # left to right as MATLAB does: exact ties in the comparison below
            # are common on integer images, so the rounding must match
            np.multiply(views[0], 0.25, out=ref)
            for v in views[1:]:
                np.multiply(v, 0.25, out=tmp)
                np.add(ref, tmp, out=ref)
            if fct > 0:
                np.multiply(ref, fct, out=tmp)
                np.subtract(core, tmp, out=tmp)      # core - fct*ref
            else:
                np.multiply(core, fct, out=tmp)
                np.add(tmp, ref, out=tmp)            # core*fct + ref
            np.greater(tmp, 0, out=hit)
            np.logical_and(hit, valid, out=hit)
            np.copyto(core, ref, where=hit)
            refresh_edges()
    out = core.copy()
    return out, cnt - out.sum()


def smooth_array(ary, dgr):
    """``fSmoothAryV01``: centred moving average of odd width ``dgr``;
    the first/last ``round((dgr-1)/2)`` points are left unchanged."""
    ary = np.asarray(ary, float).ravel()
    end_n = int(mround((dgr - 1) / 2))
    half = (dgr - 1) // 2
    n = ary.size
    acc = np.zeros(n)
    for s in range(-half, half + 1):
        shifted = np.zeros(n)
        if s >= 0:
            shifted[s:] = ary[:n - s] if s else ary
        else:
            shifted[:n + s] = ary[-s:]
        acc += shifted
    smt = acc / dgr
    if end_n:
        smt[:end_n] = ary[:end_n]
        smt[-end_n:] = ary[-end_n:]
    return smt


def low_pass_line(v, fct, min_v, itr):
    """``fLowPassLineV01``: spike removal against the mean of the two
    neighbours (distance 1, then distance 2)."""
    v = np.asarray(v, float).ravel().copy()
    s, e = v[0], v[-1]
    for _ in range(itr):
        c1 = np.r_[s, v, e]
        c2 = 0.5 * (np.r_[v, e, e] + np.r_[s, s, v])
        hit = (c1 > c2 * fct) & (c1 > min_v)
        v = np.where(hit, c2, c1)[1:-1]
    for _ in range(itr):
        c1 = np.r_[s, s, v, e, e]
        c2 = 0.5 * (np.r_[v, e, e, e, e] + np.r_[s, s, s, s, v])
        hit = (c1 > c2 * fct) & (c1 > min_v)
        v = np.where(hit, c2, c1)[2:-2]
    return v


def get_rms(x, cnts):
    """``fGetRmsV01``: (std, mean) of ``x`` weighted by ``cnts``."""
    x = np.asarray(x, float).ravel()
    c = np.asarray(cnts, float).ravel()
    tot = c.sum()
    e = np.sum(x * c)
    e2 = np.sum(x * x * c)
    return np.sqrt(tot * e2 - e ** 2) / tot, e / tot


def get_fwhm(x, y):
    """``fGetFwhmV04``: returns (fwhmIn, fwhmOut, indIn, indOut, indPk),
    indices 0-based."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if np.sum(y) == 0 or x.size <= 3:
        return 0.0, 0.0, (0, 0), (0, 0), 0
    ind_pk = int(np.nanargmax(y)) if not np.all(np.isnan(y)) else 0
    max_v = y[ind_pk]
    rx, ry = x[ind_pk:], y[ind_pk:]
    lx, ly = x[:ind_pk + 1], y[:ind_pk + 1]

    k2 = find_last(ry > 0.5 * max_v)
    k1 = find_first(ry < 0.5 * max_v)
    if k1 is None:
        if k2 is None:
            k1 = k2 = 0
        else:
            k1 = k2
    else:
        k1 -= 1
    k4 = find_first(ly > 0.5 * max_v)
    k3 = find_last(ly < 0.5 * max_v)
    if k3 is None:
        if k4 is None:
            k3 = k4 = 0
        else:
            k3 = k4
    else:
        k3 += 1
    fwhm_in = abs(rx[k1] - lx[k3])
    fwhm_out = abs(rx[k2] - lx[k4])
    return fwhm_in, fwhm_out, (k3, k1 + ind_pk), (k4, k2 + ind_pk), ind_pk
