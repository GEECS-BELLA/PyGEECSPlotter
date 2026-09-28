# Transverse e-beam profile (phosphor screen, CAM-TEA-EBeam_Profile).
# Port of the e-beam part of the live quickE script bellaLiveMagspc3.m
# (which also writes the CAM-TEA-EBeam_ProfileA outputs fBellaSShotTri
# reads for infoE): fNmnlImgPrc, the screen / hole masks, the filter-wheel
# factor, fBellaPhosSv and fSpotAnalysisV01. The ICT part of that script is
# independent and not ported.

from dataclasses import dataclass

import numpy as np

from PyGEECSPlotter.magspec.calibration import (lanex_c2c_vignette, lanex_row,
                                                load_cam_calib, load_lanex_table,
                                                pick_dated_file)
from PyGEECSPlotter.magspec.image_proc import (get_fwhm, get_rms, rotate_image,
                                               tony_bg_subtract, xray_out)

LOW_PASS = (2, 1, 1, 2)     # lPass in bellaLiveMagspc3 (fctX1, pitX1, minX1, itr1)
SCREEN_DISTANCE_M = 11.1    # screen distance from the nominal cap entrance [m]
HOLE_RADIUS_MRAD = 1.15     # holeR: radius of the 1" hole
CAP_LENGTH_M = 0.03         # capL

# Filter-wheel position ('Phosphor-FW') -> transmission correction, for
# data after the 17_0816 filter change (bellaLiveMagspc3 only has this set).
FILTER_FACTORS = {
    6: 1.0,                     # open
    1: 1 / 0.6788,              # blue pass
    2: 10 ** 0.3 / 0.6788,      # blue pass + ND0.3
    3: 1.2805 * 10 / 0.6788,    # blue pass + ND1
    4: 100 / 0.6788,            # blue pass + ND2
}
FILTER_DEFAULT = 10 ** 3 / 0.6788   # any other position: blue pass + ND3
# the raw sfile header; ScanDataAnalyzer renames aliased columns to the alias
FILTER_COLUMNS = ('Phosphor-FW', 'FW-BELLA-General pos.Channel4 Alias:Phosphor-FW')


def filter_position_from(context):
    """Filter-wheel position from an sfile row, under either column name."""
    for col in FILTER_COLUMNS:
        if col in context and context[col] == context[col]:   # present, not NaN
            return context[col]
    raise KeyError(f"no filter-wheel column {FILTER_COLUMNS} in the sfile row; "
                   "pass analyzer_dict['filter_position'] instead")


def filter_factor(position):
    """Transmission correction for a filter-wheel position."""
    try:
        return FILTER_FACTORS.get(int(position), FILTER_DEFAULT)
    except (TypeError, ValueError):
        return FILTER_DEFAULT


class EBeamProfileCalibration:
    """Static calibration for the EBeam-profile screen: the last row of
    ``*camCalib.txt`` plus its lanex row, giving axes, masks and the
    counts-to-charge factor.

    Parameters
    ----------
    calib_dir : str
        The ``Calibrations/ESMCalib`` directory.
    day : str
        Experiment day, e.g. ``'26_0521'``.
    lanex_file : str
        Lanex calibration file name inside ``calib_dir``.
    hole_radius, cap_length :
        ``holeR`` [mrad] and ``capL`` [m] from bellaLiveMagspc3.
    """

    def __init__(self, calib_dir, day, lanex_file='200301lanexCalib.txt',
                 hole_radius=HOLE_RADIUS_MRAD, cap_length=CAP_LENGTH_M):
        path, _ = pick_dated_file(calib_dir, '*camCalib.txt', day)
        cam = load_cam_calib(path)[-1]           # camClb(end)
        self.cam = cam
        lanex = lanex_row(load_lanex_table(f'{calib_dir}/{lanex_file}'), cam.setN)
        self.c2c, _ = lanex_c2c_vignette(cam, lanex)   # fC/count (vignette unused)

        # axes (fBellaLiveMagspc3 'e-beam prf axis info')
        mm_per_px = cam.fov / lanex['full width']
        x_mm = mm_per_px * (np.arange(1, cam.width + 1) - cam.leftPos + cam.xSt)
        y_mm = mm_per_px * (np.arange(1, cam.height + 1) - cam.yCntr + cam.ySt)
        self.x_mm = x_mm[cam.xSt - 1:cam.xEd]
        self.y_mm = y_mm[cam.ySt - 1:cam.yEd]
        L = SCREEN_DISTANCE_M - cap_length
        self.x_mrad = self.x_mm / L
        self.y_mrad = self.y_mm / L
        self.dmrad = self.x_mrad[1] - self.x_mrad[0]

        xm, ym = np.meshgrid(self.x_mrad, self.y_mrad)
        self.hole_radius = hole_radius
        self.hole_mask = (xm / hole_radius) ** 2 + (ym / hole_radius) ** 2 <= 1
        self.screen_mask = (xm / 2.6 - 0.07) ** 2 + (ym / 3.6 - 0.04) ** 2 <= 1

        # damage holes: the first hole1..hole3 rectangles with a nonzero x1
        # (bellaLiveMagspc3 checks hole3, hole2, hole1; hole4 is not used)
        self.damage_holes = []
        holes = list(cam.holes[:3]) if cam.holes else []
        for n in (3, 2, 1):
            if len(holes) >= n and holes[n - 1][0] != 0:
                self.damage_holes = [tuple(int(v) for v in h) for h in holes[:n]]
                break


def process_profile(raw, bg, cal, filter_position):
    """One shot, as in bellaLiveMagspc3 'E-beam prf analysis'.

    ``raw`` / ``bg`` are full 12-bit frames (``open_12bit_png``).
    Returns (image [pC] on the analysis ROI, saturated flag)."""
    cam = cal.cam
    raw = np.asarray(raw, float)
    saturated = bool(np.max(raw) == 4096)   # MATLAB tests 4096 (not 4095)
    roi = (slice(cam.ySt - 1, cam.yEd), slice(cam.xSt - 1, cam.xEd))
    img = tony_bg_subtract(raw[roi], np.asarray(bg, float)[roi])
    img, _ = xray_out(img, LOW_PASS)
    # xCntr = leftPos for the phosphor camera
    img = rotate_image(img, cam.rot, (cam.leftPos, cam.yCntr))
    img = img * cal.screen_mask
    for x1, x2, y1, y2 in cal.damage_holes:
        img[y1 - 1:y2, x1 - 1:x2] = 0
    img = filter_factor(filter_position) * 0.001 * cal.c2c * img
    return img, saturated


def spot_analysis(x, y, img):
    """``fSpotAnalysisV01``: (peak x, peak y, mean x, mean y, fwhm x,
    fwhm y, std x, std y)."""
    img = np.nan_to_num(np.asarray(img, float))
    max_xi = int(np.argmax(img.max(axis=0)))
    max_yi = int(np.argmax(img[:, max_xi]))
    fwhm_x, _, _, _, pk_x = get_fwhm(x, img[max_yi, :])
    fwhm_y, _, _, _, pk_y = get_fwhm(y, img[:, max_xi])
    std_x, mean_x = get_rms(x, img.sum(axis=0))
    std_y, mean_y = get_rms(y, img.sum(axis=1))
    return (x[pk_x], y[pk_y], mean_x, mean_y, fwhm_x, fwhm_y, std_x, std_y)


@dataclass
class ProfileResult:
    image: np.ndarray       # [pC] per pixel, analysis ROI
    x_mrad: np.ndarray
    y_mrad: np.ndarray
    scalars: dict


def analyze_profile(raw, bg, cal, filter_position):
    """Full per-shot EBeam-profile analysis. The scalar names follow the
    MATLAB ``EBeamPrf ...`` columns (without the prefix)."""
    img, sat = process_profile(raw, bg, cal, filter_position)
    img0 = np.nan_to_num(img)
    stats = spot_analysis(cal.x_mrad, cal.y_mrad, img0)
    scalars = {
        'charge [pC]': float(img0.sum()),
        'charge in hole [pC]': float((img0 * cal.hole_mask).sum()),
        'peak angle x [mrad]': stats[0], 'peak angle y [mrad]': stats[1],
        'mean angle x [mrad]': stats[2], 'mean angle y [mrad]': stats[3],
        'fwhm div x [mrad]': stats[4], 'fwhm div y [mrad]': stats[5],
        'std div x [mrad]': stats[6], 'std div y [mrad]': stats[7],
        'mx fluence [pC/mrad2]': float(img0.max() / cal.dmrad ** 2),
        'saturation': int(sat),
    }
    return ProfileResult(img, cal.x_mrad, cal.y_mrad, scalars)
