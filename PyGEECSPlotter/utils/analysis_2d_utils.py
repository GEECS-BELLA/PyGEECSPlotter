"""2-D (image) analysis functions.

Pure array maths used by ``ImageAnalyzer``'s pipeline steps, usable on their
own: ROI handling, thresholding and filtering, centroids and beam centres,
second moments, encircled energy, lineouts, masks, rotations and fits.
``ImageAnalyzer`` keeps every name as a static-method alias, so
``ImageAnalyzer.centroid(...)`` and ``analysis_2d_utils.centroid(...)`` are
the same function.
"""

import numpy as np
from scipy.signal import medfilt2d
from scipy.ndimage import affine_transform
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import PyGEECSPlotter.utils.analysis_1d_utils as a1d


def subtract_background(data, bg, scale=1.0):
    """``data - scale * bg``; raises if the shapes differ (e.g. bg taken before an ROI)."""
    bg = np.asarray(bg, dtype=float)
    if bg.shape != np.shape(data):
        raise ValueError(f"background shape {bg.shape} != image shape {np.shape(data)}; "
                         f"subtract the background before any ROI step.")
    return data - scale * bg


def roi_data(data, roi):
    """
    Crop data to region of interest.
    ROI should be [row_start, row_end, col_start, col_end]; it is
    clipped to the image (see clip_roi), None ends mean the image edge.
    """
    r0, r1, c0, c1 = clip_roi(roi, data.shape)
    return data[r0:r1, c0:c1]


def clip_roi(roi, shape):
    """
    ROI [row_start, row_end, col_start, col_end] clipped to an image of
    `shape`. None ends mean the image edge; negative values count from
    the end, as in Python slicing.
    """
    N, M = shape[:2]
    r0, r1, c0, c1 = roi
    r0, r1, _ = slice(r0, r1).indices(N)
    c0, c1, _ = slice(c0, c1).indices(M)
    return r0, max(r0, r1), c0, max(c0, c1)


@staticmethod    
def extract_subarray(array, x0, y0, n, ny=None):
    """
    n x n (or ny x n) window centred on pixel (x0, y0), zero-padded where
    it runs off the image.
    """
    if ny is None:
        ny = n
    x0, y0 = int(x0), int(y0)
    N, M = array.shape

    # Clip center to nearest valid pixel
    x0 = np.clip(x0, 0, M - 1)
    y0 = np.clip(y0, 0, N - 1)

    row_start = y0 - ny // 2
    row_end   = row_start + ny
    col_start = x0 - n // 2
    col_end   = col_start + n

    row_start_clip = max(0, row_start)
    row_end_clip   = min(N, row_end)
    col_start_clip = max(0, col_start)
    col_end_clip   = min(M, col_end)

    subarray = array[row_start_clip:row_end_clip, col_start_clip:col_end_clip]

    pad_top    = row_start_clip - row_start
    pad_bottom = row_end - row_end_clip
    pad_left   = col_start_clip - col_start
    pad_right  = col_end - col_end_clip

    subarray = np.pad(
        subarray,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0
    )

    return subarray


def compute_data_counts(data):
    """
    Return max, mean, and sum of data ignoring NaNs.
    """
    return {
        'max_counts':  np.nanmax(data),
        'mean_counts': np.nanmean(data),
        'sum_counts':  np.nansum(data)
    }


def compute_n_saturated(data, bit_depth=12, saturation_threshold=None):
    """
    Count the number of saturated pixels in 'data'.
    """
    if saturation_threshold is None:
        saturation_threshold = (2**bit_depth) - 2
    return np.count_nonzero(data >= saturation_threshold)


def calculate_misalignment(data, 
                           target_x, target_y, 
                           measured_x0=None, measured_y0=None, 
                           analyzer_dict={}, 
                           offset_type='measured'):

    if measured_x0 is None and measured_y0 is None:
        measured_x0, measured_y0 = thresholded_centroid(data, 
                                                thresh_min=analyzer_dict.get('centroid_threshold_low', 0.85)
                                               )
        offset_type='measured'

    x_offset = (measured_x0 - target_x)*analyzer_dict.get('dx', 1)
    y_offset = (measured_y0 - target_y)*analyzer_dict.get('dy', 1)
    r_offset = np.sqrt(x_offset**2 + y_offset**2)

    return {f'{offset_type} x_offset' : x_offset, 
            f'{offset_type} y_offset' : y_offset, 
            f'{offset_type} r_offset' : r_offset
           }


def apply_threshold(data, threshold_low, threshold_high, units='auto'):
    """
    Zero‐out pixels outside [threshold_low, threshold_high): pixels equal
    to threshold_low are kept (as in the LabVIEW focus analysis).
    A negative threshold_low or non-positive threshold_high is ignored.

    units: 'auto' (legacy) — low > 1 / high >= 1 are counts, otherwise
    a fraction of the max; 'counts' — always counts; 'frac' — always a
    fraction of the max.
    """
    if units not in ('auto', 'counts', 'frac'):
        raise ValueError(f"apply_threshold: unknown units {units!r}")
    data_copy = np.copy(data)

    if units != 'auto':
        peak = np.nanmax(data) if units == 'frac' else 1.0
        if threshold_low >= 0:
            data_copy *= (data >= threshold_low*peak).astype(np.uint8)
        if threshold_high > 0:
            data_copy *= (data < threshold_high*peak).astype(np.uint8)
        return data_copy

    if threshold_low >= 0:
        if threshold_low > 1:
            data_copy *= (data >= threshold_low).astype(np.uint8)
        else:
            data_copy *= (data >= threshold_low * np.nanmax(data)).astype(np.uint8)

    if threshold_high > 0:
        if threshold_high < 1:
            data_copy *= (data < threshold_high * np.nanmax(data)).astype(np.uint8)
        else:
            data_copy *= (data < threshold_high).astype(np.uint8)

    return data_copy


def remove_outliers(data, threshold=3, mask=None):
    """
    Replace outliers (beyond threshold*std) with NaN.
    Optional 'mask' for partial application.
    """
    mean = np.nanmean(data)
    std_dev = np.nanstd(data)

    lower_bound = mean - threshold*std_dev
    upper_bound = mean + threshold*std_dev

    outliers = (data < lower_bound) | (data > upper_bound)
    if mask is not None:
        outliers = np.logical_and(outliers, mask)

    data_with_nans = np.where(outliers, np.nan, data)
    return data_with_nans


def median_filter(data, N=3):
    """N x N median filter (N odd). NaNs are treated as 0."""
    return medfilt2d(np.nan_to_num(np.asarray(data, dtype=np.float64)), [N, N])


def get_spatial_coords(data, method='center', x0=None, y0=None,
                       dx=1, dy=1, centroid_thresh=0.85, round_centroid=True):
    """
    Compute x, y arrays for data, using one of several 'method's to place origin:
      - 'center', 'pixel', 'manual', 'centroid'

    With method='centroid' and round_centroid=True (default), the axis
    origin is the centroid rounded to the nearest pixel, so every shot's
    axis lies on the same pixel lattice and lineouts from different shots
    line up by a whole-pixel shift. The returned x0, y0 stay sub-pixel.
    """
    if method == 'center':
        x0 = 0.5*data.shape[1]
        y0 = 0.5*data.shape[0]
    elif method == 'pixel':
        x0, y0 = 0, 0
        dx, dy = 1, 1
    elif method == 'manual':
        if x0 is None:
            x0 = 0.5*data.shape[1]
        if y0 is None:
            y0 = 0.5*data.shape[0]
    elif method == 'centroid':
        x0, y0 = thresholded_centroid(data, thresh_min=centroid_thresh)
    else:
        x0, y0 = 0.5*data.shape[1], 0.5*data.shape[0]

    x_origin, y_origin = x0, y0
    if method == 'centroid' and round_centroid:
        x_origin, y_origin = np.round(x0), np.round(y0)
    x = (np.arange(data.shape[1]) - x_origin) * dx
    y = (np.arange(data.shape[0]) - y_origin) * dy
    return x, y, x0, y0


def coords_for_origin(data, origin='centroid', dx=1, dy=1, centroid_thresh=0.85,
                      round_centroid=True):
    """
    Pixel-centre coordinates (x, y) of `data` and the origin (x0, y0, in
    pixels) for an analyzer_dict 'origin' setting:

      'centroid'  thresholded centroid (rounded to a whole pixel for the
                  axes if round_centroid); falls back to 'center' if the
                  image has no signal
      'center'    the image centre
      'pixel'     the image corner, in pixels (dx, dy ignored)
      (x0, y0)    fixed pixel position in this image

    Returns x, y, x0, y0 as get_spatial_coords does.
    """
    if isinstance(origin, str):
        if origin not in ('centroid', 'center', 'pixel'):
            raise ValueError(f"origin must be 'centroid', 'center', 'pixel' or (x0, y0), "
                             f"got {origin!r}")
        method, x0, y0 = origin, None, None
    else:
        method, (x0, y0) = 'manual', origin
    if method == 'centroid':
        with np.errstate(invalid='ignore', divide='ignore'):
            cx, cy = thresholded_centroid(np.asarray(data, float),
                                                 thresh_min=centroid_thresh)
        if not (np.isfinite(cx) and np.isfinite(cy)):
            method = 'center'
    return get_spatial_coords(
        data, method=method, x0=x0, y0=y0, dx=dx, dy=dy,
        centroid_thresh=centroid_thresh, round_centroid=round_centroid)


def get_imshow_extent(x, y):
    """
    imshow extent [left, right, bottom, top] for pixel-centre coordinates
    x, y: the outer pixel edges, half a pixel beyond the first and last
    centres, so each pixel is drawn centred on its coordinate.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    hx = 0.5*(x[1] - x[0]) if x.size > 1 else 0.5
    hy = 0.5*(y[1] - y[0]) if y.size > 1 else 0.5
    return np.array([x[0] - hx, x[-1] + hx, y[0] - hy, y[-1] + hy])


def thresholded_centroid(data, thresh_min=0.85, thresh_max=4095):
    """
    Helper for centroid calculation. 
    Used by get_spatial_coords(method='centroid').
    """
    data_copy = np.copy(data)
    if thresh_min > 1:
        data_copy *= (data > thresh_min).astype(np.uint8)
    else:
        data_copy *= (data > thresh_min * np.nanmax(data)).astype(np.uint8)

    if thresh_max < 1:
        data_copy *= (data < thresh_max * np.nanmax(data)).astype(np.uint8)
    else:
        data_copy *= (data < thresh_max).astype(np.uint8)

    sum_counts = np.nansum(data_copy)
    x0 = np.nansum(np.arange(data_copy.shape[1]) * data_copy) / sum_counts
    y0 = np.nansum(np.arange(data_copy.shape[0]) * data_copy.T) / sum_counts
    return x0, y0


def centroid(data, x0=None, y0=None, r_max=None, n_iter=1):
    """
    Intensity-weighted centroid (x0, y0) in pixels, negative and NaN
    pixels counted as 0. With r_max, the centroid is recomputed n_iter
    times using only pixels within r_max pixels of the previous estimate
    (starting from x0, y0 if given), so isolated far-out pixels can't
    pull it.
    """
    w = np.clip(np.nan_to_num(np.asarray(data, dtype=float)), 0, None)
    N, M = w.shape
    xx, yy = np.arange(M)[None, :], np.arange(N)[:, None]
    total = w.sum()
    if total <= 0:
        return 0.5*M, 0.5*N
    if x0 is None or y0 is None:
        x0, y0 = (w*xx).sum()/total, (w*yy).sum()/total
    if r_max is not None:
        for _ in range(n_iter):
            wm = w*(((xx - x0)**2 + (yy - y0)**2) <= r_max**2)
            total = wm.sum()
            if total <= 0:
                break
            x0, y0 = (wm*xx).sum()/total, (wm*yy).sum()/total
    return x0, y0


def find_beam_center(data, method='centroid', thresh=0.85, presmooth=3):
    """
    Beam centre (x0, y0) in pixels, for placing an automatic ROI.

    method 'centroid': thresholded centroid (_get_centroid with
    thresh_min=thresh); 'peak': position of the maximum. presmooth > 1
    median-filters a copy first so single hot pixels can't move the
    centre (presmooth=0 reproduces the legacy auto_roi exactly). Falls
    back to the image centre if the result isn't finite.
    """
    d = np.asarray(data, dtype=float)
    if presmooth and presmooth > 1:
        d = median_filter(d, int(presmooth) | 1)
    N, M = d.shape
    if method == 'peak':
        if not np.any(np.isfinite(d)):
            return 0.5*M, 0.5*N
        y0, x0 = np.unravel_index(np.nanargmax(d), d.shape)
    elif method == 'centroid':
        with np.errstate(invalid='ignore', divide='ignore'):
            x0, y0 = thresholded_centroid(d, thresh_min=thresh)
    else:
        raise ValueError(f"find_beam_center: unknown method {method!r}")
    if not (np.isfinite(x0) and np.isfinite(y0)):
        return 0.5*M, 0.5*N
    return x0, y0


def apply_radial_mask(data, x0, y0, r_max, fill_value=0):
    """Copy of data with pixels further than r_max pixels from (x0, y0) set to fill_value."""
    N, M = data.shape
    r2 = (np.arange(M)[None, :] - x0)**2 + (np.arange(N)[:, None] - y0)**2
    return np.where(r2 <= r_max**2, data, fill_value)


def apply_elliptical_mask(array, x0, y0, a, b, fill_value=np.nan, invert=False):
    """
    Apply an elliptical mask to a 2D array, setting values inside or outside the ellipse to 'fill_value'
    based on the invert parameter. Optionally, return only the mask as 1s and 0s.

    Parameters:
        array (np.ndarray): The input 2D array to mask.
        x0 (int): The x-coordinate of the ellipse's center.
        y0 (int): The y-coordinate of the ellipse's center.
        a (int): The semi-major axis of the ellipse.
        b (int): The semi-minor axis of the ellipse.
        fill_value (float or np.nan): The value to assign to elements inside or outside the ellipse.
        invert (bool): If True, mask the inside of the ellipse; otherwise, mask the outside.
        return_mask_only (bool): If True, return the mask as a 2D array of 1s and 0s; otherwise, return the masked array.

    Returns:
        np.ndarray: The masked array or the mask (1s and 0s) depending on 'return_mask_only'.
    """
    # Create an index grid
    Y, X = np.ogrid[:array.shape[0], :array.shape[1]]
    # Calculate the distance from the center adjusted for the ellipse
    dist_from_center = ((X - x0)**2 / a**2) + ((Y - y0)**2 / b**2)
    # Determine whether to mask inside or outside the ellipse
    if invert:
        mask = dist_from_center <= 1  # Mask inside
    else:
        mask = dist_from_center > 1  # Mask outside

    # Copy the original array to not modify it
    masked_array = np.copy(array).astype(float)
    # Apply the mask
    masked_array[mask] = fill_value
    return masked_array


def compute_second_moments(data, dx=1, dy=1, x0=None, y0=None, r_max=None, n_iter=1):
    """
    Second-moment (ISO 11146) beam widths.

    Negative and NaN pixels count as 0. The centre is the intensity
    centroid unless x0, y0 (pixels) are given. With r_max (pixels), only
    pixels within r_max of the centre are used, and the centroid is
    re-found inside that circle n_iter times.

    Widths are 2*sigma (sigma = rms width), in dx/dy units, matching the
    LabVIEW focus analysis. Returns a dict:
      'x', 'y'          2*sigma along x and y
      'x45', 'y45'      2*sigma along the +45 deg (x=y) and -45 deg
                        diagonals: sigma45^2 = (sx^2 + sy^2)/2 +- sxy
      'major', 'minor'  2*sigma along the principal axes
      'angle'           deg, major axis from +x
      'cov xy'          the xy covariance sxy, in (dx units)^2
      'x0', 'y0'        centre used, pixels
    """
    w = np.clip(np.nan_to_num(np.asarray(data, dtype=float)), 0, None)
    N, M = w.shape
    cx, cy = centroid(w, x0=x0, y0=y0, r_max=r_max, n_iter=n_iter)
    if x0 is not None and y0 is not None:
        cx, cy = x0, y0
    xx = (np.arange(M)[None, :] - cx)
    yy = (np.arange(N)[:, None] - cy)
    if r_max is not None:
        w = w*((xx**2 + yy**2) <= r_max**2)
    total = w.sum()
    out = {'x0': cx, 'y0': cy}
    if total <= 0:
        for k in ('x', 'y', 'x45', 'y45', 'major', 'minor'):
            out[k] = np.nan
        out.update({'cov xy': np.nan, 'angle': np.nan})
        return out
    X, Y = xx*dx, yy*dy
    sxx = (w*X**2).sum()/total
    syy = (w*Y**2).sum()/total
    sxy = (w*X*Y).sum()/total
    mean = 0.5*(sxx + syy)
    diff = np.sqrt((0.5*(sxx - syy))**2 + sxy**2)
    var = {'x': sxx, 'y': syy, 'x45': mean + sxy, 'y45': mean - sxy,
           'major': mean + diff, 'minor': mean - diff}
    for k, v in var.items():
        out[k] = 2*np.sqrt(max(v, 0))
    out['cov xy'] = sxy
    out['angle'] = np.degrees(0.5*np.arctan2(2*sxy, sxx - syy))
    return out


def compute_encircled_energy(data, dx=1, dy=1, x0=None, y0=None, radii=(), fractions=(0.5, 0.865)):
    """
    Encircled energy about (x0, y0) (pixels; default: centroid).

    Returns 'ee r=<R>' (fraction of the total within radius R, R in dx
    units) for each of `radii`, 'ee radius <p>%' (radius holding a
    fraction p of the total) for each of `fractions`, and
    'peak fraction' (max pixel / total). Negative and NaN pixels count
    as 0.
    """
    w = np.clip(np.nan_to_num(np.asarray(data, dtype=float)), 0, None)
    N, M = w.shape
    if x0 is None or y0 is None:
        x0, y0 = centroid(w)
    r = np.sqrt(((np.arange(M)[None, :] - x0)*dx)**2 + ((np.arange(N)[:, None] - y0)*dy)**2)
    total = w.sum()
    out = {}
    if total <= 0:
        out.update({f'ee r={R:g}': np.nan for R in radii})
        out.update({f'ee radius {100*p:g}%': np.nan for p in fractions})
        out['peak fraction'] = np.nan
        return out
    order = np.argsort(r, axis=None)
    r_sorted = r.ravel()[order]
    cum = np.cumsum(w.ravel()[order])/total
    for R in radii:
        out[f'ee r={R:g}'] = w[r <= R].sum()/total
    for p in fractions:
        out[f'ee radius {100*p:g}%'] = float(np.interp(p, cum, r_sorted))
    out['peak fraction'] = w.max()/total
    return out


def adjust_data_to_fluence(image, energy, dx, dy, spatial_units='microns'):
    """
    Adjust image data to represent fluence in J/cm² based on total energy and pixel size.

    Parameters:
        image (ndarray): 2D image array.
        energy (float): Total energy of the laser pulse (Joules).
        dx (float): Pixel size in x (microns).
        dy (float): Pixel size in y (microns).

    Returns:
        calibrated_image (ndarray): Image calibrated to J/cm².
        fluence (float): Calibration factor in J/cm² per count.
    """

    if spatial_units == 'microns' or spatial_units == 'um':
        dx_cm = dx / 10000.0
        dy_cm = dy / 10000.0
    else:
        dx_cm = dx / 10.0
        dy_cm = dy / 10.0


    # Compute energy per count
    energy_per_count = energy / np.sum(image)

    # Convert to fluence
    fluence = energy_per_count / (dx_cm * dy_cm)

    # Apply scaling
    calibrated_image = image * fluence

    return calibrated_image, fluence


def get_lineouts(data, x0, y0, method='center', Nlo=2):
    """
    Return x_lo, y_lo lineouts from data around (x0,y0).
    method='center' takes vertical+horizontal strips of width Nlo.
    method='all' averages entire dimension.
    """
    if method == 'center':
        x_lo = np.nanmean(data[int(y0 - 0.5*Nlo):int(y0 + 0.5*Nlo), :], axis=0)
        y_lo = np.nanmean(data[:, int(x0 - 0.5*Nlo):int(x0 + 0.5*Nlo)], axis=1)
    elif method == 'all':
        x_lo = np.nanmean(data, axis=0)
        y_lo = np.nanmean(data, axis=1)
    else:
        x_lo = np.nanmean(data, axis=0)
        y_lo = np.nanmean(data, axis=1)
    return x_lo, y_lo


def rotate_around(image, angle, x0, y0):
    """
    Rotate 'image' around (x0, y0) in‐place via affine_transform.
    """
    angle_rad = np.deg2rad(angle)
    cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)

    rot_mat = np.array([[cos_t, sin_t, 0],
                        [-sin_t, cos_t, 0],
                        [0, 0, 1]])
    trans1 = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [-x0, -y0, 1]])
    trans2 = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [x0, y0, 1]])
    transform_matrix = trans1 @ rot_mat @ trans2
    transform_linear = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]

    rotated = affine_transform(image, transform_linear, offset=offset,
                               output_shape=image.shape, order=1)
    return rotated


def rotational_average(image, x0, y0, interpolate_nans=True):
    """
    Compute radial average around (x0, y0).
    """
    Y, X = np.indices(image.shape)
    R = np.sqrt((X - x0)**2 + (Y - y0)**2)
    R = np.round(R).astype(int)

    max_r = np.max(R)
    avg = np.zeros(max_r+1)

    for r in range(max_r+1):
        mask = (R == r)
        avg[r] = np.nanmean(image[mask])

    if interpolate_nans:
        avg = a1d.interpolate_nan_values(avg)
    return avg


def rotational_average_sector(
    image, x0, y0,
    dtheta=360.0, theta0=0.0,
    interpolate_nans=True,
    r_rounding="round",
    return_counts=False,
):
    """
    Radial (rotational) average over a sector (wedge) of angular width dtheta (deg),
    centered at theta0 (deg), with:

    - 0° = up (toward decreasing y)
    - +angle = clockwise

    Example: top-right quadrant -> dtheta=90, theta0=45.
    """
    img = np.asarray(image, dtype=float)

    Y, X = np.indices(img.shape)
    dx = X - x0
    dy = Y - y0

    # Radius
    r = np.sqrt(dx**2 + dy**2)
    if r_rounding == "round":
        R = np.rint(r).astype(int)
    elif r_rounding == "floor":
        R = np.floor(r).astype(int)
    elif r_rounding == "ceil":
        R = np.ceil(r).astype(int)
    else:
        raise ValueError("r_rounding must be 'round', 'floor', or 'ceil'")

    # Angle in degrees, with 0=up and CW positive:
    # Start from arctan2(dy, dx) which is 0 on +x and CCW positive,
    # then convert to (0=up, CW positive) via:
    theta = (np.degrees(np.arctan2(dy, dx)) + 90.0) % 360.0
    theta = (-theta) % 360.0  # flip to CW-positive while keeping 0=up

    # Sector mask
    if dtheta >= 360.0:
        ang_mask = np.ones_like(theta, dtype=bool)
    else:
        d = a1d.angular_difference_deg(theta, theta0)
        ang_mask = np.abs(d) <= (dtheta / 2.0)

    max_r = int(np.nanmax(R))
    avg = np.full(max_r + 1, np.nan)
    counts = np.zeros(max_r + 1, dtype=int)

    for rr in range(max_r + 1):
        mask = (R == rr) & ang_mask
        vals = img[mask]
        if vals.size:
            avg[rr] = np.nanmean(vals)
            counts[rr] = np.count_nonzero(np.isfinite(vals))

    if interpolate_nans:
        avg = a1d.interpolate_nan_values(avg)

    return (avg, counts) if return_counts else avg


def angle_deg_from_points(x0, y0, x1, y1):
    """
    Angle from (x0, y0) -> (x1, y1) in degrees with:
    - 0° = up (negative y direction)
    - +angle = clockwise
    Returns angle in [0, 360).
    """
    dx = x1 - x0
    dy = y1 - y0

    # If the point is identical, angle is undefined
    if np.isscalar(dx) and np.isscalar(dy) and dx == 0 and dy == 0:
        return np.nan

    # Start with standard: atan2(dy, dx) gives 0° on +x and CCW positive
    theta = (np.degrees(np.arctan2(dy, dx)) + 90.0) % 360.0  # 0° at up, still CCW-ish
    theta = (-theta) % 360.0  # make clockwise-positive
    return theta


def fit_2d_polynomial(data, roi_bounds=None, exclude_nan=False, degree=2):
    """
    Fits a 2D polynomial to the data, either excluding a region of interest (ROI)
    or ignoring NaN values.

    Parameters:
    data (np.ndarray): 2D array of shape (N1, M1).
    roi_bounds (tuple, optional): Bounds of the ROI in the format 
                                  (roi_x_start, roi_x_end, roi_y_start, roi_y_end).
                                  If None, the entire data is used.
    exclude_nan (bool): If True, the function ignores NaN values. Defaults to False.
    degree (int): Degree of the polynomial to fit.

    Returns:
    np.ndarray: 2D array with the 2D polynomial values.
    """
    N1, M1 = data.shape

    if exclude_nan:
        # Create a mask for NaN values
        mask = ~np.isnan(data)
    elif roi_bounds is not None:
        # Create a mask for the ROI
        roi_x_start, roi_x_end, roi_y_start, roi_y_end = roi_bounds
        mask = np.ones((N1, M1), dtype=bool)
        mask[roi_x_start:roi_x_end, roi_y_start:roi_y_end] = False
    else:
        # If no mask condition is applied, fit the entire dataset
        mask = np.ones((N1, M1), dtype=bool)

    # Extract the points based on the mask
    masked_data = data[mask]

    # Generate the coordinates for all points
    x_coords, y_coords = np.meshgrid(np.arange(N1), np.arange(M1), indexing='ij')
    x_coords_flat = x_coords.flatten()
    y_coords_flat = y_coords.flatten()

    # Generate the coordinates for the masked points
    x_coords_masked = x_coords[mask]
    y_coords_masked = y_coords[mask]

    # Fit a 2D polynomial
    input_pts = np.vstack((x_coords_masked, y_coords_masked)).T

    poly = PolynomialFeatures(degree)
    in_features = poly.fit_transform(input_pts)

    model = LinearRegression()
    model.fit(in_features, masked_data.flatten())

    # Predict values on the full grid
    full_grid_coords = np.vstack((x_coords_flat, y_coords_flat)).T
    full_in_features = poly.fit_transform(full_grid_coords)
    predicted_full_grid = model.predict(full_in_features)

    # Reshape the predicted values to the original grid shape
    predicted_data_full = predicted_full_grid.reshape(N1, M1)

    return predicted_data_full


def gaussian_2d(xydata, amplitude, center_x, center_y, sigma_x, sigma_y, offset):
    x, y = xydata
    if sigma_x <= 0 or sigma_y <= 0:
        return np.full_like(x, np.inf)
    exponent = -0.5 * ( ((x - center_x)/sigma_x)**2 + ((y - center_y)/sigma_y)**2 )
    return amplitude * np.exp(exponent) + offset

def fit_gaussian_2d(xdata, ydata, zdata, p0=None, bounds=None,
                    loss='linear', max_nfev=5000):
    xdata = np.array(xdata, dtype=float)
    ydata = np.array(ydata, dtype=float)
    zdata = np.array(zdata, dtype=float)
    if xdata.shape != ydata.shape or xdata.shape != zdata.shape:
        return {'success': False,
                'message': 'xdata, ydata, and zdata must have the same shape.'}

    valid_mask = np.isfinite(xdata) & np.isfinite(ydata) & np.isfinite(zdata)
    if not np.any(valid_mask):
        return {'success': False, 'message': 'No valid data'}

    x_valid = xdata[valid_mask]
    y_valid = ydata[valid_mask]
    z_valid = zdata[valid_mask]

    xy_valid = (x_valid, y_valid)

    def estimate_initial_params(x_arr, y_arr, z_arr):
        z_min, z_max = np.nanmin(z_arr), np.nanmax(z_arr)
        amp_guess = z_max - z_min
        offset_guess = z_min
        # center guess at the peak
        idx_max = np.argmax(z_arr)
        cx_guess = x_arr[idx_max]
        cy_guess = y_arr[idx_max]
        # rough sigma guesses
        half_level = z_min + 0.5 * amp_guess
        above_half = z_arr > half_level
        if not np.any(above_half):
            sigma_x_guess = (np.max(x_arr) - np.min(x_arr)) / 6.0
            sigma_y_guess = (np.max(y_arr) - np.min(y_arr)) / 6.0
        else:
            x_half = x_arr[above_half]
            y_half = y_arr[above_half]
            sigma_x_guess = np.sqrt(np.mean((x_half - cx_guess)**2)) or 1e-3
            sigma_y_guess = np.sqrt(np.mean((y_half - cy_guess)**2)) or 1e-3
        return [amp_guess, cx_guess, cy_guess, sigma_x_guess, sigma_y_guess, offset_guess]

    if p0 is None:
        p0 = estimate_initial_params(x_valid, y_valid, z_valid)

    if bounds is None:
        lower_bounds = [0.0, -np.inf, -np.inf, 1e-12, 1e-12, -np.inf]
        upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        bounds = (lower_bounds, upper_bounds)

    def residual(params, xy, z):
        model = gaussian_2d(xy, *params)
        return z - model

    res = least_squares(residual, x0=p0, args=(xy_valid, z_valid),
                        bounds=bounds, loss=loss, max_nfev=max_nfev)

    amp, cx, cy, sx, sy, offset = res.x
    return {
        'success':    res.success,
        'message':    res.message,
        'cost':       res.cost,
        'params':     res.x,
        'amplitude':  amp,
        'center_x':   cx,
        'center_y':   cy,
        'sigma_x':    sx,
        'sigma_y':    sy,
        'offset':     offset
    }
