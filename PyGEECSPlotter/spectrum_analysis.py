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

# Misspelt keys used by the pre-port notebooks, still accepted so old
# analyzer_dicts run unchanged.
_LEGACY_KEYS = {
    'include_diagnostic_response': 'include_diagnostic_repsonse',
    'diagnostic_response_arrays': 'diagnotic_response_arrays',
}


def get_config(analyzer_dict, key, default=None):
    """``analyzer_dict[key]``, falling back to its legacy misspelling."""
    if key in analyzer_dict:
        return analyzer_dict[key]
    return analyzer_dict.get(_LEGACY_KEYS.get(key, key), default)


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

    ``aux`` is ``{'wl': wavelength, 'wl_lo': counts}``, so the lineout
    displayers (``OpticalSpectrumWaterfall``, ``OpticalSpectrumMeanPerBin``,
    ``OpticalSpectrumMeanWaterfall``) work with ``axis='wl'``. The
    waterfalls need every shot on the same wavelength axis: true for raw
    spectra from one spectrometer, and guaranteed by ``wl_lin``.

    analyzer_dict keys
    ------------------
    bg_file : bool
        Subtract the ``bg`` passed to ``analyze_data`` (a DataFrame, a
        2-column array or a counts array; see ``average_background``).
    bg_constant, bg_constant_low_wl, bg_constant_high_wl :
        Subtract the mean counts in a signal-free wavelength window.
    apply_median_filter, apply_mov_mean_filter, N_filt :
        Smoothing.
    include_diagnostic_response, wl_lin, diagnostic_response_arrays :
        Interpolate onto ``wl_lin`` and divide by each response array
        (see ``load_response``).
    calculate_red_blue_shifts, threshold_for_shifts :
        Width / edge wavelengths; shots whose peak is below the threshold
        get ``below_threshold_value`` (default 0; ``np.nan`` keeps dim shots
        out of per-bin means).
    camera_responses : dict
        ``{name: (wl, response)}`` (see ``load_camera_response``): adds a
        ``'<name> response ratio'`` result per camera, the fraction of this
        spectrum the camera registers (``camera_response_ratio``). Divide a
        camera's summed counts by it to compare beam energy across shots
        with different spectra (``CameraEnergyAnalyzer``).
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

        # 4-5) Stats + optional shifts
        results = self.spectrum_results(wl_out, counts_out, analyzer_dict)

        counts_out = np.nan_to_num(counts_out, nan=0.0)
        final_df = pd.DataFrame({'Wavelength (nm)': wl_out, 'Counts': counts_out})

        return final_df, results, {'wl': wl_out, 'wl_lo': counts_out}

    def spectrum_results(self, wl, counts, analyzer_dict):
        """Peak wavelength, max / mean / sum counts, plus the shifts if
        ``calculate_red_blue_shifts``."""
        if counts.size and np.any(np.isfinite(counts)):
            x0 = int(np.argmax(np.nan_to_num(counts, nan=-np.inf)))
            results = {
                'peak wl (nm)': float(wl[x0]),
                'max counts': float(np.nanmax(counts)),
                'mean counts': float(np.nanmean(counts)),
                'sum counts': float(np.nansum(counts)),
            }
        else:
            results = dict.fromkeys(['peak wl (nm)', 'max counts', 'mean counts', 'sum counts'], np.nan)
        if analyzer_dict.get('calculate_red_blue_shifts', False):
            results = merge_dicts_overwrite(results, self.shift_results(wl, counts, analyzer_dict))
        for name, (cam_wl, cam_response) in analyzer_dict.get('camera_responses', {}).items():
            results[f'{name} response ratio'] = self.camera_response_ratio(wl, counts, cam_wl, cam_response)
        return results

    @staticmethod
    def camera_response_ratio(wl, counts, cam_wl, cam_response):
        """
        Fraction of the spectrum a camera registers:
        ``sum(S * r) / sum(S)``, with ``r`` the camera's spectral response
        interpolated onto ``wl`` (0 outside the curve's range) and ``S`` the
        spectrum clipped at 0 (residual negative counts after background
        subtraction would otherwise make the ratio noisy on dim shots).

        Plain sums, so ``r`` is taken as a response per unit energy and
        ``wl`` as a uniform grid (true for ``wl_lin`` and the combined
        spectrum). NaN when the spectrum has no positive signal.
        """
        s = np.clip(np.nan_to_num(np.asarray(counts, dtype=float), nan=0.0), 0, None)
        total = s.sum()
        if total <= 0:
            return np.nan
        r = np.interp(wl, cam_wl, cam_response, left=0.0, right=0.0)
        return float(np.sum(s * r) / total)

    def load_camera_response(self, path):
        """``(wl, response)`` arrays from a 2-column (wavelength, response)
        file, for ``analyzer_dict['camera_responses']``."""
        curve = self.load_data(path)
        if curve is None:
            raise FileNotFoundError(path)
        return curve['Wavelength (nm)'].to_numpy(dtype=float), curve['Counts'].to_numpy(dtype=float)

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
            counts = counts - self._bg_counts_on(bg, wl)

        if analyzer_dict.get('bg_constant', False):
            bg_low = analyzer_dict.get('bg_constant_low_wl', 340)
            bg_high = analyzer_dict.get('bg_constant_high_wl', 380)
            bg_ldx = int(np.argmin(np.abs(wl - bg_low)))
            bg_hdx = int(np.argmin(np.abs(wl - bg_high)))
            if bg_hdx > bg_ldx:
                counts = counts - np.nanmean(counts[bg_ldx:bg_hdx])

        return counts

    @staticmethod
    def _bg_counts_on(bg, wl):
        """
        Background counts on the shot's wavelength axis ``wl``. ``bg`` is a
        'Wavelength (nm)' / 'Counts' DataFrame, a (n, 2) array of the same
        columns (as ``scan.mean_std_diagnostic`` returns), or a bare counts
        array already on ``wl``. With a wavelength column, a background on
        a different axis is interpolated rather than mis-subtracted.
        """
        if isinstance(bg, pd.DataFrame):
            bg_wl = np.asarray(bg['Wavelength (nm)'].values, dtype=float)
            bg_counts = np.asarray(bg['Counts'].values, dtype=float)
        else:
            arr = np.asarray(bg, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 2:
                bg_wl, bg_counts = arr[:, 0], arr[:, 1]
            else:
                if arr.shape != wl.shape:
                    raise ValueError(f"Background has {arr.shape} counts but the spectrum has "
                                     f"{wl.shape}; pass a DataFrame with a wavelength column.")
                return arr
        if bg_wl.shape == wl.shape and np.allclose(bg_wl, wl):
            return bg_counts
        return np.interp(wl, bg_wl, bg_counts)

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
        if get_config(analyzer_dict, 'include_diagnostic_response', False) and wl_lin is not None:
            new_counts = np.interp(wl_lin, wl, counts)
            for response_array in get_config(analyzer_dict, 'diagnostic_response_arrays', []):
                new_counts = new_counts / response_array
            new_counts[np.isinf(new_counts)] = np.nan
            return np.asarray(wl_lin, dtype=float), new_counts
        return wl, counts

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def average_background(self, scan, show_progress=False):
        """
        Mean raw spectrum over ``scan``'s active shots, as a 'Wavelength (nm)'
        / 'Counts' DataFrame to pass as ``bg`` (replaces the old
        ``generate_averaged_background``). ``scan`` is a ``ScanDataAnalyzer``
        loaded on a background (e.g. laser-blocked) sfile. Shots are loaded
        unprocessed, without this analyzer's bg / filters / response.
        """
        raw = _RawSpectrum(diagnostic=self.diagnostic, file_ext=self.file_ext)
        raw.register_with_scan(scan)
        mean, _ = scan.mean_std_diagnostic(raw, show_progress=show_progress)
        if mean is None:
            raise RuntimeError(f"No {self.diagnostic} spectra found for the background.")
        return pd.DataFrame(mean, columns=['Wavelength (nm)', 'Counts'])

    def load_response(self, path, wl_lin, zero_to=np.inf, nan_below_index=None):
        """
        Spectral response curve from a 2-column (wavelength, response) file,
        interpolated onto ``wl_lin``, for ``diagnostic_response_arrays``.
        Zero response becomes ``zero_to`` (inf divides the counts to 0; NaN
        blanks them), and samples before ``nan_below_index`` become NaN,
        which ``CombinedVisNIRSpectrum`` also uses to find the overlap.
        """
        curve = self.load_data(path)
        if curve is None:
            raise FileNotFoundError(path)
        response = np.interp(wl_lin, curve['Wavelength (nm)'].values, curve['Counts'].values)
        response[response == 0.0] = zero_to
        if nan_below_index:
            response[:nan_below_index] = np.nan
        return response

    def shift_results(self, wl, counts, analyzer_dict):
        """Lineout-width and cumulative shifts, with the dict's threshold settings."""
        threshold = analyzer_dict.get('threshold_for_shifts', 600)
        fill = analyzer_dict.get('below_threshold_value', 0)
        shifts = self.compute_spectrum_shifts(counts, wl, threshold=threshold, fill=fill)
        cumsum = self.compute_cumulative_spectrum_shifts(wl, counts, threshold=threshold, fill=fill)
        return merge_dicts_overwrite(shifts, cumsum)

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

    _WIDTH_LEVELS = [('half max', 0.5), ('1/e', 1.0 / np.e), ('5pct', 0.05), ('1pct', 0.01)]

    def compute_spectrum_shifts(self, data, wl, threshold=0, fill=0):
        """
        Compute FWHM / 1-e / 5% / 1% widths and the bracketing (blue, red)
        wavelengths. Every value is ``fill`` if ``max(data) < threshold``.
        """
        if not np.any(np.isfinite(data)) or np.nanmax(data) < threshold:
            keys = []
            for label, _ in self._WIDTH_LEVELS:
                keys += ['fwhm (nm)' if label == 'half max' else f'width at {label} (nm)',
                         f'lambda_b {label} (nm)', f'lambda_r {label} (nm)']
            return dict.fromkeys(keys, fill)

        safe = self._safe_index
        x0 = int(np.argmax(np.nan_to_num(data, nan=-np.inf)))

        result = {}
        for label, frac in self._WIDTH_LEVELS:
            _, ldx, hdx = get_lineout_width(data, x0, from_center=False, width_at=frac)
            lam_b = safe(wl, ldx)
            lam_r = safe(wl, hdx)
            width_key = 'fwhm (nm)' if label == 'half max' else f'width at {label} (nm)'
            result[width_key] = lam_r - lam_b
            result[f'lambda_b {label} (nm)'] = lam_b
            result[f'lambda_r {label} (nm)'] = lam_r
        return result

    # (label, fraction of the total) for the cumulative-sum edges; the red
    # side uses 1 - fraction
    _CUMULATIVE_LEVELS = [('1pct', 0.01), ('5pct', 0.05), ('10pct', 0.10),
                          ('20pct', 0.20), ('1/e', 1.0 / np.e)]

    def compute_cumulative_spectrum_shifts(self, wl, data, threshold=0, fill=0):
        """
        Wavelengths at which the cumulative sum first crosses 1%, 5%, 10%,
        20%, 1/e of the total (blue side, ``lambda_b``) and 1 minus those
        (red side, ``lambda_r``). Every value is ``fill`` if
        ``max(data) < threshold``.
        """
        keys = [f'lambda_{side} cumulative {label} (nm)'
                for side in 'br' for label, _ in self._CUMULATIVE_LEVELS]
        if not np.any(np.isfinite(data)) or np.nanmax(data) < threshold:
            return dict.fromkeys(keys, fill)

        cumsum = np.nancumsum(data)
        total = np.nansum(data)
        if total <= 0:
            return dict.fromkeys(keys, fill)

        def first_wl_above(frac):
            idx = np.where(cumsum > frac * total)[0]
            return float(wl[idx[0]]) if idx.size else np.nan

        result = {}
        for label, frac in self._CUMULATIVE_LEVELS:
            result[f'lambda_b cumulative {label} (nm)'] = first_wl_above(frac)
        for label, frac in self._CUMULATIVE_LEVELS:
            result[f'lambda_r cumulative {label} (nm)'] = first_wl_above(1.0 - frac)
        return result


class _RawSpectrum(OpticalSpectrumAnalyzer):
    """Loads spectra and passes them through unprocessed, as an (n, 2) array
    (for ``OpticalSpectrumAnalyzer.average_background``)."""

    def analyze_data(self, data, bg=None, context=None, analyzer_dict=None):
        if data is None:
            return None, {}, {}
        return data[['Wavelength (nm)', 'Counts']].to_numpy(dtype=float), {}, {}
