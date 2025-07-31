# Base class ImageAnalyzer for PyGEECSPlotter
# Author: Alex Picksley
# Version 0.4
# Created: 2024-02-26
# Last Modified: 2025-02-19


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
from scipy.ndimage import affine_transform
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import imageio as imio

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

import PyGEECSPlotter.ni_imread as ni_imread
from PyGEECSPlotter.utils import super_gaussian, get_lineout_width

class ImageAnalyzer:
    """
    Base class for image analysis. 
    Derive from this class to customize analysis steps for specific image types.
    """

    def __init__(self, 
                 diagnostic=None, 
                 file_ext=None, 
                 analyzer_dict={}, 
                 display_dict={},
                 output_diagnostic=None,
                 output_file_ext=None
                ):
        
        self.diagnostic = diagnostic
        self.file_ext = file_ext
        self.analyzer_dict = analyzer_dict 
        self.display_dict = display_dict 
        self.output_diagnostic = None
        self.output_file_ext = None

    # -------------------------------------------------------------------
    # Public pipeline method
    # -------------------------------------------------------------------
    def load_data(self, filename):
        return ni_imread.read_imaq_image(filename).astype('float')
    
    def analyze_data(self, data, analyzer_dict=None, bg=None):
        if analyzer_dict is None:
            analyzer_dict = self.analyzer_dict

        data_out = np.copy(data)
        results = {}

        # 1) Saturation
        if analyzer_dict.get('return_n_saturated', True):
            results['n_saturated'] = self.compute_n_saturated(
                data_out,
                bit_depth=analyzer_dict.get('bit_depth', 12),
                saturation_threshold=analyzer_dict.get('saturation_threshold', None)
            )

        # 2) Background
        if analyzer_dict.get('bg_file', False) and bg is not None:
            data_out -= bg
            
        data_out -= analyzer_dict.get('bg_const', 0)

        # 3) ROI
        if analyzer_dict.get('roi', None) is not None:
            data_out = self.roi_data(data_out, analyzer_dict['roi'])
            
        if analyzer_dict.get('auto_roi', False):
            x0_roi, y0_roi = ImageAnalyzer._get_centroid(data_out, 
                                    thresh_min=analyzer_dict.get('auto_roi_thresh', 0.85)
                                    )
            data_out = ImageAnalyzer.extract_subarray(data_out, x0_roi, y0_roi, analyzer_dict.get('auto_roi_size', 500))

        # 3.5) (Optional) remove outliers
        if analyzer_dict.get('remove_outliers', False):
            data_out = self.remove_outliers(
                data_out,
                threshold=analyzer_dict.get('outlier_threshold', 3),
                mask=analyzer_dict.get('outlier_mask', None)
            )

        # 4) Data stats
        if analyzer_dict.get('return_data_counts', True):
            results.update(self.compute_data_counts(data_out))

        # 5) Median filter
        if analyzer_dict.get('median_filter', False):
            N_filt = analyzer_dict.get('N_filt', 3)
            data_out = medfilt2d(data_out.astype(np.float64), [N_filt, N_filt])

        # 6) Spatial coords + lineouts
        x, y, x0, y0 = self.get_spatial_coords(
            data_out,
            method=analyzer_dict.get('centroid_method', 'center'),
            x0=analyzer_dict.get('x0', None),
            y0=analyzer_dict.get('y0', None),
            dx=analyzer_dict.get('dx', 1),
            dy=analyzer_dict.get('dy', 1),
            centroid_thresh=analyzer_dict.get('centroid_threshold_low', 0.85)
        )
        x_lo, y_lo = self.get_lineouts(
            data_out,
            x0,
            y0,
            method=analyzer_dict.get('lineout_method', 'center'),
            Nlo=analyzer_dict.get('Nlo', 2)
        )
        results['x']    = x
        results['y']    = y
        results['x_lo'] = x_lo
        results['y_lo'] = y_lo
        results['imshow_extent'] = ImageAnalyzer.get_imshow_extent(x,y)

        # 7) Fit super-Gaussian
        if analyzer_dict.get('fit_super_gaussian', False):
            include_exponent = analyzer_dict.get('include_exponent', True)
            fit_x = self.fit_super_gaussian(x, x_lo, fit_exponent=include_exponent)
            fit_y = self.fit_super_gaussian(y, y_lo, fit_exponent=include_exponent)
            results['fit_x'] = fit_x
            results['fit_y'] = fit_y
            
            if include_exponent:
                keys_to_add = ['amplitude', 'center', 'sigma', 'offset', 'N']
            else:
                keys_to_add = ['amplitude', 'center', 'sigma', 'offset']
            
            filtered_data = {f'gauss fit x {key}': fit_x[key] for key in keys_to_add}
            results.update(filtered_data)
            filtered_data = {f'gauss fit y {key}': fit_y[key] for key in keys_to_add}
            results.update(filtered_data)

        # 8) Threshold final data
        threshold_low  = analyzer_dict.get('threshold_low', -1)
        threshold_high = analyzer_dict.get('threshold_high', -1)
        if threshold_low >= 0 or threshold_high > 0:
            data_out = self.apply_threshold(data_out, threshold_low, threshold_high)
            
        # 9) Second moment
        # Doesn't work yet
        
        # 9.5) FWHM
        fwhm, hdx, ldx = get_lineout_width(x_lo)
        results['fwhm x'] = fwhm*analyzer_dict.get('dx', 1)
        fwhm, hdx, ldx = get_lineout_width(y_lo)
        results['fwhm y'] = fwhm*analyzer_dict.get('dy', 1)
        
        # 10) Misalignment from target
        if analyzer_dict.get('measured_misalignment', False):
            results.update(self.calculate_misalignment(data_out, x0, y0, analyzer_dict=analyzer_dict))
        if analyzer_dict.get('fitted_misalignment', False) and analyzer_dict.get('fit_super_gaussian', False):
            fitted_x0 = fit_x['center']
            fitted_y0 = fit_y['center']
            results.update(self.calculate_misalignment(data_out, 0, 0, 
                                           measured_x0=fitted_x0, 
                                           measured_y0=fitted_y0, 
                                           analyzer_dict={},
                                           offset_type='fitted'
                                          ))


        return data_out, results

    def display_data(self, data, display_dict=None, title=None):
        if display_dict is None:
            display_dict = self.display_dict
            
        fig, ax = plt.subplots(figsize=display_dict.get('figsize', (6,5)))
        im = ax.imshow(data,
                        aspect=display_dict.get('aspect', 'equal'),
                        norm=display_dict.get('norm', None),
                        cmap=display_dict.get('cmap', 'RdBu'),
                        interpolation=display_dict.get('interpolation', None),
                        origin='lower',
                        extent=display_dict.get('extent', None),
                        vmin=display_dict.get('vmin', None),
                        vmax=display_dict.get('vmax', None),
                      )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(display_dict.get('cbar_label', 'Counts'))

        spatial_units = display_dict.get('spatial_units', 'pixels')
        xtitle = display_dict.get('xtitle', 'x')
        ytitle = display_dict.get('ytitle', 'y')

        ax.set_xlabel(r'$%s \ (\mathrm{%s})$' %(xtitle, spatial_units))
        ax.set_ylabel(r'$%s \ (\mathrm{%s})$' %(ytitle, spatial_units))

        axlims = display_dict.get('axlims', None)
        tick_dx = display_dict.get('tick_dx', None)
        if tick_dx is not None and axlims is not None:
            ax.set_xticks(np.arange(-axlims, axlims+tick_dx, tick_dx))
            ax.set_yticks(np.arange(-axlims, axlims+tick_dx, tick_dx))

        if axlims is not None:
            ax.set_xlim([-axlims, axlims])
            ax.set_ylim([-axlims, axlims])


        for v_line in display_dict.get('v_lines', []):
            ax.axvline(x=v_line, color='k', linestyle='--')

        for h_line in display_dict.get('h_lines', []):
            ax.axhline(y=h_line, color='k', linestyle='--')

        if title is not None:
            ax.set_title(title)

        return fig, ax

    def display_lineout(self, x, x_lo, display_dict=None, title=None):
        if display_dict is None:
            display_dict = self.display_dict
            
        fig, ax = plt.subplots(figsize=display_dict.get('figsize', (6,5)))
        ax.plot(x, x_lo)

        spatial_units = display_dict.get('spatial_units', 'pixels')
        xtitle = display_dict.get('xtitle', 'x')
        ytitle = display_dict.get('ytitle', 'y')

        ax.set_xlabel(r'$%s \ (\mathrm{%s})$' %(xtitle, spatial_units))
        ax.set_ylabel(r'$%s \ (\mathrm{%s})$' %(ytitle, spatial_units))

        axlims = display_dict.get('axlims', None)
        tick_dx = display_dict.get('tick_dx', None)
        if tick_dx is not None and axlims is not None:
            ax.set_xticks(np.arange(0, axlims+tick_dx, tick_dx))

        if axlims is not None:
            ax.set_xlim([0, axlims])

        for v_line in display_dict.get('v_lines', []):
            ax.axvline(x=v_line, color='k', linestyle='--')

        for h_line in display_dict.get('h_lines', []):
            ax.axhline(y=h_line, color='k', linestyle='--')

        title_append = display_dict.get('title_append', '')
        if title is not None:
            ax.set_title('%s - %s' %(title, title_append))

        return fig, ax


    # -------------------------------------------------------------------
    # Utility methods (can be overridden or extended in subclasses)
    # -------------------------------------------------------------------

    @staticmethod
    def roi_data(data, roi):
        """
        Crop data to region of interest.
        ROI should be [row_start, row_end, col_start, col_end].
        """
        return data[roi[0]:roi[1], roi[2]:roi[3]]

    @staticmethod
    def compute_data_counts(data):
        """
        Return max, mean, and sum of data ignoring NaNs.
        """
        return {
            'max_counts':  np.nanmax(data),
            'mean_counts': np.nanmean(data),
            'sum_counts':  np.nansum(data)
        }

    @staticmethod
    def calculate_misalignment(data, 
                               target_x, target_y, 
                               measured_x0=None, measured_y0=None, 
                               analyzer_dict={}, 
                               offset_type='measured'):
        
        if measured_x0 is None and measured_y0 is None:
            measured_x0, measured_y0 = ImageAnalyzer._get_centroid(data, 
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

    @staticmethod
    def compute_n_saturated(data, bit_depth=12, saturation_threshold=None):
        """
        Count the number of saturated pixels in 'data'.
        """
        if saturation_threshold is None:
            saturation_threshold = (2**bit_depth) - 2
        return np.count_nonzero(data >= saturation_threshold)

    @staticmethod
    def apply_threshold(data, threshold_low, threshold_high):
        """
        Zero‐out pixels outside [threshold_low, threshold_high].
        Thresholds can be absolute or fractional (0–1 of max).
        """
        data_copy = np.copy(data)

        if threshold_low >= 0:
            if threshold_low > 1:
                data_copy *= (data > threshold_low).astype(np.uint8)
            else:
                data_copy *= (data > threshold_low * np.nanmax(data)).astype(np.uint8)

        if threshold_high > 0:
            if threshold_high < 1:
                data_copy *= (data < threshold_high * np.nanmax(data)).astype(np.uint8)
            else:
                data_copy *= (data < threshold_high).astype(np.uint8)

        return data_copy

    @staticmethod
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

    @staticmethod
    def get_spatial_coords(data, method='center', x0=None, y0=None,
                           dx=1, dy=1, centroid_thresh=0.85):
        """
        Compute x, y arrays for data, using one of several 'method's to place origin:
          - 'center', 'pixel', 'manual', 'centroid'
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
            x0, y0 = ImageAnalyzer._get_centroid(data, thresh_min=centroid_thresh)
        else:
            x0, y0 = 0.5*data.shape[1], 0.5*data.shape[0]

        x = (np.arange(data.shape[1]) - x0) * dx
        y = (np.arange(data.shape[0]) - y0) * dy
        return x, y, x0, y0

    @staticmethod
    def _get_centroid(data, thresh_min=0.85, thresh_max=4095):
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

    @staticmethod
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

    @staticmethod
    def get_imshow_extent(x, y):
        return np.array([x[0], x[-1], y[0], y[-1]])

    @staticmethod
    def fit_super_gaussian(xdata, ydata, fit_exponent=False, p0=None, bounds=None,
                           loss='linear', max_nfev=5000):
        """
        Fit a super‐Gaussian function to (xdata,ydata).
        """
        xdata = np.array(xdata, dtype=float)
        ydata = np.array(ydata, dtype=float)

        valid_mask = np.isfinite(xdata) & np.isfinite(ydata)
        xdata = xdata[valid_mask]
        ydata = ydata[valid_mask]
        if xdata.size == 0:
            return {'success': False, 'message': 'No valid data'}

        def estimate_initial_params(xx, yy):
            y_min, y_max = np.nanmin(yy), np.nanmax(yy)
            amp_guess    = y_max - y_min
            idx_max      = np.nanargmax(yy)
            cen_guess    = xx[idx_max]
            off_guess    = y_min

            half_level = y_min + 0.5*amp_guess
            crosses = np.where(np.diff((yy > half_level).astype(int)) != 0)[0]
            if len(crosses) >= 2:
                x_left, x_right = xx[crosses[0]], xx[crosses[-1]]
                sigma_guess = (x_right - x_left)/2.355
                sigma_guess = max(1e-6, np.abs(sigma_guess))
            else:
                sigma_guess = max(1e-6, (xx.max() - xx.min())/10)
            return [amp_guess, cen_guess, sigma_guess, off_guess]

        if p0 is None:
            p0_est = estimate_initial_params(xdata, ydata)
            if fit_exponent:
                p0_est.append(2.0)
            p0 = p0_est

        if bounds is None:
            if fit_exponent:
                bounds = ([0, -np.inf, 1e-12, -np.inf, 1],
                          [np.inf, np.inf, np.inf, np.inf, 20])
            else:
                bounds = ([0, -np.inf, 1e-12, -np.inf],
                          [np.inf, np.inf, np.inf, np.inf])

        def residual(params, xx, yy):
            if fit_exponent:
                amp, cen, sig, off, N = params
            else:
                amp, cen, sig, off = params
                N = 2
            return yy - super_gaussian(xx, amp, cen, sig, off, N)

        res = least_squares(residual, p0, args=(xdata, ydata),
                            bounds=bounds, loss=loss, max_nfev=max_nfev)

        result = {
            'success':  res.success,
            'message':  res.message,
            'cost':     res.cost,
            'params':   res.x
        }
        if fit_exponent:
            amp, cen, sig, off, exponent = res.x
            result.update({'amplitude': amp, 'center': cen, 'sigma': sig,
                           'offset': off, 'N': exponent})
        else:
            amp, cen, sig, off = res.x
            result.update({'amplitude': amp, 'center': cen, 'sigma': sig,
                           'offset': off})
        return result

    # -------------------------------------------------------------------
    # Optional / Misc: plotting, rotating, or special calculations
    # -------------------------------------------------------------------

    @staticmethod
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

    @staticmethod
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
            avg = ImageAnalyzer.interpolate_nan_values(avg)
        return avg

    @staticmethod
    def interpolate_nan_values(data):
        """
        Interpolate over 1D array's NaN regions using nearest valid points.
        """
        nans = np.isnan(data)
        notnans = ~nans
        if not np.any(notnans):
            return data  # all NaNs
        indices = np.arange(len(data))
        data[nans] = np.interp(indices[nans], indices[notnans], data[notnans])
        return data

    @staticmethod
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
    
    @staticmethod
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
        masked_array = np.copy(array)
        # Apply the mask
        masked_array[mask] = fill_value
        return masked_array
    
    @staticmethod    
    def extract_subarray(array, x0, y0, n):
        """
        Extracts an n x n subarray centered around (x0, y0) from a 2D array.
        
        Parameters:
            array (ndarray): Input 2D array of shape (N, M).
            x0 (int): Center x-coordinate.
            y0 (int): Center y-coordinate.
            n (int): Size of the desired subarray (n x n).
        
        Returns:
            ndarray: Extracted subarray of shape (n, n), zero-padded if necessary.
        """
        x0, y0 = int(x0), int(y0)
        N, M = array.shape
        half_n = n // 2

        # Compute slice indices ensuring they remain within bounds
        x_start = max(0, y0 - half_n)
        x_end = min(N, y0 + half_n + 1)
        y_start = max(0, x0 - half_n)
        y_end = min(M, x0 + half_n + 1)

        subarray = array[x_start:x_end, y_start:y_end]

        # Determine the required padding
        pad_x_before = max(0, half_n - y0)
        pad_x_after = max(0, (y0 + half_n + 1) - N)
        pad_y_before = max(0, half_n - x0)
        pad_y_after = max(0, (x0 + half_n + 1) - M)

        # Apply padding only if necessary
        subarray = np.pad(subarray, 
                          ((pad_x_before, pad_x_after), 
                           (pad_y_before, pad_y_after)), 
                          mode='constant', constant_values=0)

        return subarray

    @staticmethod
    def add_lineout_to_imshow(ax, x, y, x_lo, y_lo, axlims, normalize=True, style_dict=None):
        """
        Plots a scaled line on the given Axes, using x for the horizontal coordinate
        and a rescaled version of x_lo for the vertical coordinate.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes on which to draw the plot.
        x : array-like
            The x-coordinates to plot.
        y : array-like
            The x-coordinates to plot.
        x_lo : array-like
            Data used to determine the line's vertical extent. It will be rescaled
            based on its max value and the provided axlims.
        y_lo : array-like
            Data used to determine the line's vertical extent. It will be rescaled
            based on its max value and the provided axlims.
        axlims : float
            A scaling factor applied after normalizing x_lo.
        style_dict : dict, optional
            Dictionary of style parameters. Recognized keys include:
            - 'color' (default: 'k')
            - 'linestyle' (default: '-')
            - 'alpha' (default: 0.75)

        Returns
        -------
        lines : list of matplotlib.lines.Line2D
            List of Line2D objects that were plotted (the return value of ax.plot).
        """
        if style_dict is None:
            style_dict = {}
        color = style_dict.get('color', 'k')
        linestyle = style_dict.get('linestyle', '-')
        alpha = style_dict.get('alpha', 0.75)
        
        if normalize:
            max_x = np.nanmax(x_lo)
            max_y = np.nanmax(y_lo)
        else:
            max_x = 1.0
            max_y = 1.0
            
        
        # Plot the lines
        line_x = ax.plot(x, 0.25*axlims*x_lo/max_x-axlims, 
                        color=color, linestyle=linestyle, alpha=alpha)
        line_y = ax.plot(0.25*axlims*y_lo/max_y-axlims, y, 
                        color=color, linestyle=linestyle, alpha=alpha)
        
        return line_x, line_y

    @staticmethod
    def overlay_fitted_curve(ax, return_dict, curve_function, axlims, style_dict=None):
        fitted_x = curve_function(return_dict['x'], *return_dict['fit_x']['params'])
        fitted_y = curve_function(return_dict['y'], *return_dict['fit_y']['params'])

        x = return_dict['x']
        y = return_dict['y']
        x_lo = return_dict['x_lo']
        y_lo = return_dict['y_lo']

        max_x = np.nanmax([x_lo, fitted_x])
        max_y = np.nanmax([y_lo, fitted_y])
        line_x, line_y = ImageAnalyzer.add_lineout_to_imshow(ax, x, y, x_lo/max_x, y_lo/max_y, axlims, normalize=False, style_dict=style_dict)
        line_fitx, line_fity = ImageAnalyzer.add_lineout_to_imshow(ax, x, y, fitted_x/max_x, fitted_y/max_y, axlims, normalize=False, style_dict={'color' : 'r'})
        return line_x, line_y


    
