import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt

from PyGEECSPlotter.diagnostic_analyzer import DiagnosticAnalyzer
from PyGEECSPlotter.navigation_utils import get_analysed_shot_save_path
from PyGEECSPlotter.utils import (
    calculate_moving_average_and_std,
    get_lineout_width,
    merge_dicts_overwrite,
)


class OpticalSpectrumAnalyzer(DiagnosticAnalyzer):
    """
    Per-shot analyzer for 1-D optical spectra.

    Expects each shot's file to load into a 2-column DataFrame with
    'Wavelength (nm)' and 'Counts'. Pipeline:

        1) Background subtraction (file-based or constant region)
        2) Optional median / moving-average filtering
        3) Optional diagnostic-response correction onto wl_lin
        4) Stats: peak wavelength, max / mean / sum counts
        5) Optional red/blue spectral shifts (lineout-width + cumulative)
    """

    def __init__(self,
                 diagnostic=None,
                 file_ext=None,
                 analyzer_dict=None,
                 display_dict=None,
                 output_diagnostic=None,
                 output_file_ext=None,
                 ):
        super().__init__(
            diagnostic=diagnostic,
            file_ext=file_ext,
            analyzer_dict=analyzer_dict,
            display_dict=display_dict,
            output_diagnostic=output_diagnostic,
            output_file_ext=output_file_ext,
        )

    # ------------------------------------------------------------------
    # Pipeline contract
    # ------------------------------------------------------------------
    def load_data(self, filename):
        if filename is None or not os.path.exists(filename):
            return None
        return pd.read_csv(
            filename,
            sep=r'\s+',
            header=None,
            names=['Wavelength (nm)', 'Counts'],
        )

    def analyze_data(self, data, bg=None, context=None, analyzer_dict=None):
        if analyzer_dict is None:
            analyzer_dict = self.analyzer_dict

        if data is None:
            return None, {}, {}

        wl = np.asarray(data['Wavelength (nm)'].values, dtype=float)
        counts = np.asarray(data['Counts'].values, dtype=float)

        # 1) Background subtraction
        counts = self._subtract_background(counts, wl, analyzer_dict, bg)

        # 2) Filters
        counts = self._apply_filters(counts, analyzer_dict)

        # 3) Diagnostic-response correction
        wl_out, counts_out = self._apply_diagnostic_response(wl, counts, analyzer_dict)

        # 4) Stats
        max_counts = float(np.nanmax(counts_out)) if counts_out.size else np.nan
        mean_counts = float(np.nanmean(counts_out)) if counts_out.size else np.nan
        sum_counts = float(np.nansum(counts_out)) if counts_out.size else np.nan
        if counts_out.size:
            x0 = int(np.argmax(np.nan_to_num(counts_out, nan=-np.inf)))
            peak_wl = float(wl_out[x0])
        else:
            peak_wl = np.nan

        results = {
            'peak wl (nm)': peak_wl,
            'max counts': max_counts,
            'mean counts': mean_counts,
            'sum counts': sum_counts,
        }

        # 5) Spectral shifts
        if analyzer_dict.get('calculate_red_blue_shifts', False):
            threshold = analyzer_dict.get('threshold_for_shifts', 600)
            shifts = self.compute_spectrum_shifts(counts_out, wl_out, threshold=threshold)
            cumsum = self.compute_cumulative_spectrum_shifts(wl_out, counts_out, threshold=threshold)
            results = merge_dicts_overwrite(results, shifts, cumsum)

        final_df = pd.DataFrame({
            'Wavelength (nm)': wl_out,
            'Counts': np.nan_to_num(counts_out, nan=0.0),
        })

        return final_df, results, {}

    def display_data(self, data, display_dict=None, return_dict=None, title=None, fig=None, ax=None):
        if display_dict is None:
            display_dict = self.display_dict

        if fig is None or ax is None:
            fig, ax = plt.subplots(
                constrained_layout=True,
                figsize=display_dict.get('figsize', display_dict.get('fig_size', (6, 4))),
            )

        ax.plot(
            data['Wavelength (nm)'],
            data['Counts'],
            label=display_dict.get('legend_label', None),
        )

        ax.set_xlabel(display_dict.get('wl_label', 'Wavelength (nm)'))
        ax.set_xlim([
            display_dict.get('wl_low', np.nanmin(data['Wavelength (nm)'])),
            display_dict.get('wl_high', np.nanmax(data['Wavelength (nm)'])),
        ])
        ax.set_ylim([
            display_dict.get('counts_low', np.nanmin(data['Counts'])),
            display_dict.get('counts_high', 1.05 * np.nanmax(data['Counts'])),
        ])

        if display_dict.get('add_legend', False):
            ax.legend(
                loc=display_dict.get('legend_location', 'best'),
                ncol=display_dict.get('legend_ncol', 1),
                title=display_dict.get('legend_title', None),
            )

        for v_line in display_dict.get('v_lines', []):
            ax.axvline(x=v_line, color='k', linestyle='--')
        for h_line in display_dict.get('h_lines', []):
            ax.axhline(y=h_line, color='k', linestyle='--')

        title_append = display_dict.get('title_append', '')
        if title is not None:
            ax.set_title(f"{title} - {title_append}" if title_append else title)

        return fig, ax

    def write_analyzed_data(self, data, analysis_dir, scan, shot_num, context=None):
        save_path = get_analysed_shot_save_path(
            analysis_dir,
            self.output_diagnostic or self.diagnostic,
            scan,
            shot_num,
            self.output_file_ext or '.txt',
        )
        data.to_csv(save_path, sep='\t', index=False, header=False, float_format='%.3f')

    # ------------------------------------------------------------------
    # Pipeline sub-steps
    # ------------------------------------------------------------------
    def _subtract_background(self, counts, wl, analyzer_dict, bg):
        if analyzer_dict.get('bg_file', False) and bg is not None:
            if isinstance(bg, pd.DataFrame):
                counts = counts - np.asarray(bg['Counts'].values, dtype=float)
            else:
                counts = counts - np.asarray(bg, dtype=float)

        if analyzer_dict.get('bg_constant', False):
            bg_low = analyzer_dict.get('bg_constant_low_wl', 340)
            bg_high = analyzer_dict.get('bg_constant_high_wl', 380)
            bg_ldx = int(np.argmin(np.abs(wl - bg_low)))
            bg_hdx = int(np.argmin(np.abs(wl - bg_high)))
            if bg_hdx > bg_ldx:
                counts = counts - np.nanmean(counts[bg_ldx:bg_hdx])

        return counts

    def _apply_filters(self, counts, analyzer_dict):
        if analyzer_dict.get('apply_median_filter', False):
            n_filt = int(analyzer_dict.get('N_filt', 5))
            counts = medfilt(counts.astype(np.float64), n_filt)
        if analyzer_dict.get('apply_mov_mean_filter', False):
            n_filt = int(analyzer_dict.get('N_filt', 5))
            counts, _ = calculate_moving_average_and_std(counts, n_filt)
        return counts

    def _apply_diagnostic_response(self, wl, counts, analyzer_dict):
        wl_lin = analyzer_dict.get('wl_lin', None)
        if analyzer_dict.get('include_diagnostic_response', False) and wl_lin is not None:
            new_counts = np.interp(wl_lin, wl, counts)
            for response_array in analyzer_dict.get('diagnostic_response_arrays', []):
                new_counts = new_counts / response_array
            new_counts[np.isinf(new_counts)] = np.nan
            return np.asarray(wl_lin, dtype=float), new_counts
        return wl, counts

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def clip_spectrum(data, wl_low=700, wl_high=900):
        mask = (data['Wavelength (nm)'] >= wl_low) & (data['Wavelength (nm)'] <= wl_high)
        return data[mask].reset_index(drop=True)

    def average_data_list(self, list_of_data):
        all_data = pd.concat(list_of_data, keys=range(len(list_of_data)), names=['DataFrame', 'Row'])
        mean_data = all_data.groupby('Wavelength (nm)')['Counts'].mean().reset_index()
        std_data = all_data.groupby('Wavelength (nm)')['Counts'].std().reset_index()
        return mean_data, std_data

    @staticmethod
    def _safe_index(arr, idx):
        try:
            return arr[int(idx)]
        except (IndexError, ValueError, TypeError):
            return np.nan

    def compute_spectrum_shifts(self, data, wl, threshold=0):
        """
        Compute FWHM / 1-e / 5% / 1% widths and the bracketing (blue, red)
        wavelengths. Returns a dict of zeros if ``max(data) < threshold``.
        """
        zeros = {
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
            'lambda_r 1pct (nm)': 0,
        }
        if not np.any(np.isfinite(data)) or np.nanmax(data) < threshold:
            return zeros

        safe = self._safe_index
        x0 = int(np.argmax(np.nan_to_num(data, nan=-np.inf)))

        result = {}
        for label, frac in [('half max', 0.5), ('1/e', 1.0 / np.e),
                            ('5pct', 0.05), ('1pct', 0.01)]:
            _, ldx, hdx = get_lineout_width(data, x0, from_center=False, width_at=frac)
            lam_b = safe(wl, ldx)
            lam_r = safe(wl, hdx)
            width_key = 'fwhm (nm)' if label == 'half max' else f'width at {label} (nm)'
            result[width_key] = lam_r - lam_b
            result[f'lambda_b {label} (nm)'] = lam_b
            result[f'lambda_r {label} (nm)'] = lam_r
        return result

    def compute_cumulative_spectrum_shifts(self, wl, data, threshold=0):
        """
        Compute the wavelengths at which the cumulative sum first crosses
        1%, 5%, 1/e of the total (blue side) and the corresponding right-side
        percentages. Returns zeros if ``max(data) < threshold``.
        """
        zeros = {
            'lambda_b cumulative 1pct (nm)': 0,
            'lambda_b cumulative 5pct (nm)': 0,
            'lambda_b cumulative 1/e (nm)': 0,
            'lambda_r cumulative 1pct (nm)': 0,
            'lambda_r cumulative 5pct (nm)': 0,
            'lambda_r cumulative 1/e (nm)': 0,
        }
        if not np.any(np.isfinite(data)) or np.nanmax(data) < threshold:
            return zeros

        cumsum = np.nancumsum(data)
        total = np.nansum(data)
        if total <= 0:
            return zeros

        def first_wl_above(frac):
            idx = np.where(cumsum > frac * total)[0]
            return float(wl[idx[0]]) if idx.size else np.nan

        return {
            'lambda_b cumulative 1pct (nm)': first_wl_above(0.01),
            'lambda_b cumulative 5pct (nm)': first_wl_above(0.05),
            'lambda_b cumulative 1/e (nm)': first_wl_above(1.0 / np.e),
            'lambda_r cumulative 1pct (nm)': first_wl_above(0.99),
            'lambda_r cumulative 5pct (nm)': first_wl_above(0.95),
            'lambda_r cumulative 1/e (nm)': first_wl_above(1.0 - 1.0 / np.e),
        }
