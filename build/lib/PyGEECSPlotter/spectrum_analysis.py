import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.constants import physical_constants
r_e = physical_constants['classical electron radius'][0]
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import PyGEECSPlotter.py_geecs_v2 as pygc
import PyGEECSPlotter.image_analysis as gpi
from PyGEECSPlotter.utils import calculate_moving_average_and_std, get_lineout_width


class OpticalSpectrumAnalyzer:
    """
    Base class for analyzing 1D optical spectra with columns for Wavelength (nm) and Counts.
    It provides a pipeline for:
      1) Loading data from file
      2) Subtracting background
      3) Optional filtering
      4) Applying diagnostic response arrays
      5) Computing stats (peak, FWHM, etc.)
      6) Returning processed data and metrics
    """

    # -------------------------------------------------------------------
    # Public pipeline methods
    # -------------------------------------------------------------------
    @staticmethod
    def load_data(filename):
        """
        Load data from a text/CSV file into a pandas DataFrame.
        Expected columns: [Wavelength (nm), Counts]

        Parameters
        ----------
        filename : str
            Path to the spectrum data file.

        Returns
        -------
        data_df : pandas.DataFrame
            DataFrame with columns 'Wavelength (nm)' and 'Counts'.
        """
        data_df = pd.read_csv(filename, sep=r'\s+', header=None, names=["Wavelength (nm)", "Counts"])
        return data_df

    def analyze_data(self, data_df, analyzer_dict={}, bg=None):
        """
        Main pipeline for analyzing 1D spectrum data with optional background subtraction,
        filtering, diagnostic-response application, spectral shifts, etc.

        Steps:
          1) Background subtraction (file-based or constant)
          2) Filtering (median, moving-average)
          3) Diagnostic response correction
          4) Compute stats (max, mean, sum, peak wavelength)
          5) Compute spectral shifts
          6) Prepare output DataFrame

        Parameters
        ----------
        data_df : pandas.DataFrame
            Input data with 'Wavelength (nm)' and 'Counts'.
        analyzer_dict : dict, optional
            Dictionary controlling analysis steps and settings.
        bg : array-like or None, optional
            Background data (1D array of same shape as 'Counts'), if applicable.

        Returns
        -------
        final_df : pandas.DataFrame
            Processed data after applying all pipeline steps.
        results : dict
            Dictionary of computed metrics (stats, shifts, etc.).
        """
        if analyzer_dict is None:
            analyzer_dict = {}

        # Copy for local use
        wl = data_df['Wavelength (nm)'].values
        counts = data_df['Counts'].values.astype(float)

        # 1) Background subtraction
        counts = self._subtract_background(counts, wl, analyzer_dict, bg)

        # 2) Filtering
        counts = self._apply_filters(counts, analyzer_dict)

        # 3) Diagnostic response correction
        wl_out, counts_out = self._apply_diagnostic_response(wl, counts, analyzer_dict)

        # 4) Compute stats (max, mean, sum, peak wl)
        max_counts = np.nanmax(counts_out)
        mean_counts = np.nanmean(counts_out)
        sum_counts = np.nansum(counts_out)
        x0 = np.argmax(np.nan_to_num(counts_out, nan=-np.inf))
        peak_wl = wl_out[x0] if len(wl_out) > 0 else np.nan

        # 5) Spectral shifts
        shifts_dict, cumsum_dict = {}, {}
        if analyzer_dict.get('calculate_red_blue_shifts', False):
            threshold_for_shifts = analyzer_dict.get('threshold_for_shifts', 600)
            shifts_dict = self.compute_spectrum_shifts(counts_out, wl_out, threshold=threshold_for_shifts)
            cumsum_dict = self.compute_cumulative_spectrum_shifts(wl_out, counts_out, threshold=threshold_for_shifts)

        # 6) Build output
        final_df = pd.DataFrame({'Wavelength (nm)': wl_out, 
                                 'Counts': np.nan_to_num(counts_out, nan=0)})
        results = {
            'peak wl (nm)': peak_wl,
            'max counts': max_counts,
            'mean counts': mean_counts,
            'sum counts': sum_counts,
        }
        # merge in shift info
        results = pygc.merge_dicts_overwrite(results, shifts_dict, cumsum_dict)

        return final_df, results

    def display_data(self, data_df, display_dict={}, title=None):
        """
        Simple plotting of 1D spectrum data.

        Parameters
        ----------
        data_df : pandas.DataFrame
            DataFrame containing 'Wavelength (nm)' and 'Counts'.
        display_dict : dict, optional
            Plotting style and options (x-limits, y-limits, lines, legend, etc.).
        title : str, optional
            Plot title.

        Returns
        -------
        (fig, ax) : tuple
            The matplotlib Figure and Axes objects.
        """
        fig, ax = plt.subplots(constrained_layout=True, figsize=display_dict.get('fig_size', (6,4)))
        ax.plot(data_df['Wavelength (nm)'], data_df['Counts'], label=display_dict.get('legend_label', None))

        # X/Y limits
        ax.set_xlabel(display_dict.get('wl_label', 'Wavelength (nm)'))
        ax.set_xlim([
            display_dict.get('wl_low', np.nanmin(data_df['Wavelength (nm)'])),
            display_dict.get('wl_high', np.nanmax(data_df['Wavelength (nm)']))
        ])
        ax.set_ylim([
            display_dict.get('counts_low', np.nanmin(data_df['Counts'])),
            display_dict.get('counts_high', 1.05 * np.nanmax(data_df['Counts']))
        ])

        # Legend
        if display_dict.get('add_legend', False):
            ax.legend(loc=display_dict.get('legend_location', 'best'),
                      ncol=display_dict.get('legend_ncol', 1),
                      title=display_dict.get('legend_title', None))

        # Additional lines
        for v_line in display_dict.get('v_lines', []):
            ax.axvline(x=v_line, color='k', linestyle='--')
        for h_line in display_dict.get('h_lines', []):
            ax.axhline(y=h_line, color='k', linestyle='--')

        # Title
        title_append = display_dict.get('title_append', '')
        if title:
            ax.set_title(f"{title} - {title_append}")

        return fig, ax

    def write_analyzed_data(self, filename, data_df):
        """
        Write analyzed data to disk.

        Parameters
        ----------
        filename : str
            Output file path.
        data_df : pandas.DataFrame
            DataFrame with 'Wavelength (nm)' and 'Counts'.
        """
        data_df.to_csv(filename, sep='\t', index=False, header=False, float_format='%.3f')

    def average_data_list(self, list_of_data):
        """
        Given a list of DataFrames (with 'Wavelength (nm)' and 'Counts'),
        compute the mean and std across all of them, grouped by Wavelength.

        Parameters
        ----------
        list_of_data : list of pandas.DataFrame
            Each DataFrame must have the same 'Wavelength (nm)' axis or
            data that can be concatenated meaningfully.

        Returns
        -------
        mean_data : pandas.DataFrame
            Mean of all input DataFrames (grouped by Wavelength).
        std_data : pandas.DataFrame
            Standard deviation of all input DataFrames (grouped by Wavelength).
        """
        all_data = pd.concat(list_of_data, keys=range(len(list_of_data)), names=['DataFrame', 'Row'])
        mean_data = all_data.groupby("Wavelength (nm)")["Counts"].mean().reset_index()
        std_data = all_data.groupby("Wavelength (nm)")["Counts"].std().reset_index()
        return mean_data, std_data

    # -------------------------------------------------------------------
    # Utility or specialized methods (spectral shifts, background, etc.)
    # -------------------------------------------------------------------
    @staticmethod
    def clip_spectrum(data_df, wl_low=700, wl_high=900):
        return data_df[(data_df['Wavelength (nm)'] >= wl_low) & (data_df['Wavelength (nm)'] <= wl_high)].reset_index(drop=True)

    
    def compute_spectrum_shifts(self, data, wl, threshold=0):
        """
        Compute lineout widths (FWHM, 1/e, etc.) and corresponding wavelengths.
        If the maximum value of the data is below 'threshold', returns zeros.

        Parameters
        ----------
        data : np.ndarray
            1D array for spectrum counts.
        wl : np.ndarray
            Wavelength array of the same shape as 'data'.
        threshold : float, optional
            If np.nanmax(data) < threshold, the method returns zeros for shift metrics.

        Returns
        -------
        dict
            Dictionary containing widths and corresponding wavelength edges for
            multiple threshold levels (half-max, 1/e, 5%, 1%).
        """
        def safe_index(arr, idx):
            try:
                return arr[int(idx)]
            except (IndexError, ValueError):
                return np.nan

        x0 = np.argmax(np.nan_to_num(data, nan=-np.inf))
        if np.nanmax(data) < threshold:
            # Return zeroed dictionary
            return {
                'fwhm (nm)': 0,
                'lambda_b half max (nm)': 0,
                'lambda_r half max (nm)': 0,
                'width at 1/e (nm)': 0,
                'lambda_b 1/e (nm)': 0,
                'lambda_r 1/e (nm)': 0,
                'width at 5pct (nm)': 0,
                'lambda_b 5pct (nm)': 0,
                'lambda_r 5pct (nm)': 0,
                'width at 1pct (nm)': 0,
                'lambda_b 1pct (nm)': 0,
                'lambda_r 1pct (nm)': 0
            }

        # FWHM
        fwhm_x, ldx, hdx = get_lineout_width(data, x0, from_center=False, width_at=0.5)
        fwhm_nm = safe_index(wl, hdx) - safe_index(wl, ldx)
        lambda_b_half = safe_index(wl, ldx)
        lambda_r_half = safe_index(wl, hdx)

        # 1/e
        one_e_x, ldx, hdx = get_lineout_width(data, x0, from_center=False, width_at=1.0 / np.e)
        one_e_nm = safe_index(wl, hdx) - safe_index(wl, ldx)
        lambda_b_one_e = safe_index(wl, ldx)
        lambda_r_one_e = safe_index(wl, hdx)

        # 5%
        p5_x, ldx, hdx = get_lineout_width(data, x0, from_center=False, width_at=0.05)
        p5_nm = safe_index(wl, hdx) - safe_index(wl, ldx)
        lambda_b_5pct = safe_index(wl, ldx)
        lambda_r_5pct = safe_index(wl, hdx)

        # 1%
        p1_x, ldx, hdx = get_lineout_width(data, x0, from_center=False, width_at=0.01)
        p1_nm = safe_index(wl, hdx) - safe_index(wl, ldx)
        lambda_b_1pct = safe_index(wl, ldx)
        lambda_r_1pct = safe_index(wl, hdx)

        return {
            'fwhm (nm)': fwhm_nm,
            'lambda_b half max (nm)': lambda_b_half,
            'lambda_r half max (nm)': lambda_r_half,
            'width at 1/e (nm)': one_e_nm,
            'lambda_b 1/e (nm)': lambda_b_one_e,
            'lambda_r 1/e (nm)': lambda_r_one_e,
            'width at 5pct (nm)': p5_nm,
            'lambda_b 5pct (nm)': lambda_b_5pct,
            'lambda_r 5pct (nm)': lambda_r_5pct,
            'width at 1pct (nm)': p1_nm,
            'lambda_b 1pct (nm)': lambda_b_1pct,
            'lambda_r 1pct (nm)': lambda_r_1pct
        }

    def compute_cumulative_spectrum_shifts(self, wl, data, threshold=0):
        """
        Compute cumulative spectrum sums and various lambda points (1%, 5%, 1/e).
        If the maximum value of the data is below 'threshold', returns zeros.

        Parameters
        ----------
        wl : np.ndarray
            Wavelength array.
        data : np.ndarray
            Spectrum counts.
        threshold : float
            If np.nanmax(data) < threshold, the method returns zeros.

        Returns
        -------
        dict
            Dictionary with bounding wavelengths at fractional cumulative thresholds.
        """
        if np.nanmax(data) < threshold:
            return {
                'lambda_b cumulative 1pct (nm)': 0,
                'lambda_b cumulative 5pct (nm)': 0,
                'lambda_b cumulative 1/e (nm)': 0,
                'lambda_r cumulative 1pct (nm)': 0,
                'lambda_r cumulative 5pct (nm)': 0,
                'lambda_r cumulative 1/e (nm)': 0
            }

        # Compute cumulative sum ignoring NaNs
        cumsum = np.nancumsum(data)
        total = np.nansum(data)

        def find_wl_for_fraction(frac):
            idx = np.where(cumsum > frac * total)[0]
            return wl[idx[0]] if len(idx) > 0 else np.nan

        # Lower edges (b) at 1%, 5%, 1/e
        lambda_b_1pct = find_wl_for_fraction(0.01)
        lambda_b_5pct = find_wl_for_fraction(0.05)
        lambda_b_one_e = find_wl_for_fraction(1.0 / np.e)

        # Upper edges (r) at 99%, 95%, (1 - 1/e)
        lambda_r_1pct = find_wl_for_fraction(0.99)
        lambda_r_5pct = find_wl_for_fraction(0.95)
        lambda_r_one_e = find_wl_for_fraction(1.0 - 1.0 / np.e)

        return {
            'lambda_b cumulative 1pct (nm)': lambda_b_1pct,
            'lambda_b cumulative 5pct (nm)': lambda_b_5pct,
            'lambda_b cumulative 1/e (nm)': lambda_b_one_e,
            'lambda_r cumulative 1pct (nm)': lambda_r_1pct,
            'lambda_r cumulative 5pct (nm)': lambda_r_5pct,
            'lambda_r cumulative 1/e (nm)': lambda_r_one_e
        }

    # -------------------------------------------------------------------
    # Internal / Private sub-methods (mirroring the pipeline steps)
    # -------------------------------------------------------------------
    def _subtract_background(self, counts, wl, analyzer_dict, bg):
        """
        Handle multiple background subtraction modes:
          - File-based background (bg array passed in)
          - Constant background from a spectral region
          - No background
        """
        # 1) If an external background file is provided
        if analyzer_dict.get('bg_file', False) and bg is not None:
            counts = counts - bg

        # 2) Constant background from a range of wavelengths
        if analyzer_dict.get('bg_constant', False):
            bg_low = analyzer_dict.get('bg_constant_low_wl', 340)
            bg_high = analyzer_dict.get('bg_constant_high_wl', 380)

            bg_ldx = np.argmin(np.abs(wl - bg_low))
            bg_hdx = np.argmin(np.abs(wl - bg_high))
            local_bg = np.nanmean(counts[bg_ldx:bg_hdx])
            counts -= local_bg

        return counts

    def _apply_filters(self, counts, analyzer_dict):
        """
        Apply filtering steps:
          - Median filter
          - Moving-average filter
        """
        # Median filter
        if analyzer_dict.get('apply_median_filter', False):
            N_filt = analyzer_dict.get('N_filt', 5)
            counts = medfilt(counts, N_filt)

        # Moving-mean filter
        if analyzer_dict.get('apply_mov_mean_filter', False):
            N_filt = analyzer_dict.get('N_filt', 5)
            counts, _ = calculate_moving_average_and_std(counts, N_filt)

        return counts

    def _apply_diagnostic_response(self, wl, counts, analyzer_dict):
        """
        If included, apply the diagnostic response arrays (e.g., QE or intensity scaling).
        Typically requires interpolating to a common wavelength grid.
        """
        wl_lin = analyzer_dict.get('wl_lin', None)
        if analyzer_dict.get('include_diagnostic_repsonse', False) and wl_lin is not None:
            # Interpolate data to new wavelength axis
            new_counts = np.interp(wl_lin, wl, counts)
            # Apply all response arrays
            for response_array in analyzer_dict.get('diagnotic_response_arrays', []):
                new_counts = new_counts / response_array
            new_counts[np.isinf(new_counts)] = np.nan
            return wl_lin, new_counts
        else:
            return wl, counts
