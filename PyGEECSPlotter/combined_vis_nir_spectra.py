# Combined visible + NIR optical spectrum analyzer.
# Worked example of MultiDiagnosticAnalyzer: takes one frame from each of two
# spectrometers per shot, pre-processes each via OpticalSpectrumAnalyzer,
# then stitches them into a single combined spectrum with overlap blending.

import numpy as np
import pandas as pd

from PyGEECSPlotter.multi_diagnostic_analyzer import MultiDiagnosticAnalyzer
from PyGEECSPlotter.spectrum_analysis import OpticalSpectrumAnalyzer
from PyGEECSPlotter.utils import merge_dicts_overwrite


class CombinedVisNIRSpectrum(MultiDiagnosticAnalyzer):
    """
    Combine a VIS spectrometer and a NIR spectrometer into one spectrum.

    Each shot's pipeline:

      1) Run the VIS sub-analyzer on its spectrum (bg subtraction,
         filtering, response correction).
      2) Run the NIR sub-analyzer the same way.
      3) Optionally zero out NIR above a watchdog wavelength if a
         2nd-order 800 nm contamination check is below threshold.
      4) Find the overlap region using the NIR response array's
         non-NaN support.
      5) Compute a single scalar to scale NIR onto VIS in the overlap.
      6) Linear blend across the overlap, then concatenate
         pre-VIS / blend / post-NIR onto a uniform VIS-spaced axis.
      7) Optional cumulative spectral shifts on the combined spectrum.

    Parameters
    ----------
    vis_analyzer : OpticalSpectrumAnalyzer
        Analyzer for the VIS spectrometer (must have ``diagnostic`` and
        ``file_ext`` set).
    nir_analyzer : OpticalSpectrumAnalyzer
        Analyzer for the NIR spectrometer.
    analyzer_dict : dict, optional
        Multi-level analyzer config. Recognised keys:
          - ``'vis_analyzer_dict'``: forwarded to ``vis_analyzer.analyze_data``
            (defaults to ``vis_analyzer.analyzer_dict``).
          - ``'nir_analyzer_dict'``: forwarded to ``nir_analyzer.analyze_data``.
          - ``'remove_2nd_order_800nm'`` (bool): enable the watchdog.
          - ``'check_2nd_order_800_low_wl'`` / ``'..._high_wl'`` /
            ``'..._thresh'``: watchdog window + threshold.
          - ``'calculate_red_blue_shifts'`` (bool) +
            ``'threshold_for_shifts'``: forwarded to
            ``OpticalSpectrumAnalyzer.compute_spectrum_shifts`` /
            ``compute_cumulative_spectrum_shifts``.
    output_diagnostic, output_file_ext, display_dict : as usual.
    """

    def __init__(
        self,
        vis_analyzer: OpticalSpectrumAnalyzer,
        nir_analyzer: OpticalSpectrumAnalyzer,
        analyzer_dict=None,
        display_dict=None,
        output_diagnostic=None,
        output_file_ext=None,
    ):
        if vis_analyzer.diagnostic is None or nir_analyzer.diagnostic is None:
            raise ValueError("Both sub-analyzers must have `diagnostic` set.")

        super().__init__(
            inputs=[
                (vis_analyzer.diagnostic, vis_analyzer.file_ext),
                (nir_analyzer.diagnostic, nir_analyzer.file_ext),
            ],
            sub_analyzers={
                vis_analyzer.diagnostic: vis_analyzer,
                nir_analyzer.diagnostic: nir_analyzer,
            },
            analyzer_dict=analyzer_dict,
            display_dict=display_dict,
            output_diagnostic=output_diagnostic or 'VIS_NIR_combined',
            output_file_ext=output_file_ext or '.txt',
        )
        self.vis_diagnostic = vis_analyzer.diagnostic
        self.nir_diagnostic = nir_analyzer.diagnostic
        # Used to compute spectral-shift stats on the combined spectrum,
        # without re-deriving the OpticalSpectrumAnalyzer helper functions.
        self._shift_helper = OpticalSpectrumAnalyzer()

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def analyze_data(self, data, bg=None, context=None):
        if data is None:
            return None, {}, {}
        vis_in = data.get(self.vis_diagnostic)
        nir_in = data.get(self.nir_diagnostic)
        if vis_in is None or nir_in is None:
            return None, {}, {}

        analyzer_dict = self.analyzer_dict
        vis_dict = analyzer_dict.get('vis_analyzer_dict',
                                     self.sub_analyzers[self.vis_diagnostic].analyzer_dict)
        nir_dict = analyzer_dict.get('nir_analyzer_dict',
                                     self.sub_analyzers[self.nir_diagnostic].analyzer_dict)

        vis_bg = self._bg_for(bg, self.vis_diagnostic)
        nir_bg = self._bg_for(bg, self.nir_diagnostic)

        # 1-2) Sub-analyzers process each spectrometer independently.
        vis_data, vis_results, _ = self.sub_analyzers[self.vis_diagnostic].analyze_data(
            vis_in, bg=vis_bg, context=context, analyzer_dict=vis_dict,
        )
        nir_data, nir_results, _ = self.sub_analyzers[self.nir_diagnostic].analyze_data(
            nir_in, bg=nir_bg, context=context, analyzer_dict=nir_dict,
        )

        # 3) Optional 2nd-order 800 nm watchdog: zero out NIR above the
        # watchdog wavelength if the check region is too dim.
        if analyzer_dict.get('remove_2nd_order_800nm', False):
            nir_data = self._maybe_zero_nir_above_watchdog(nir_data, analyzer_dict)

        # 4) Overlap support from the NIR response array(s).
        nir_xover_include = self._nir_overlap_support(nir_dict, nir_data)

        # 5-6) Scale NIR onto VIS, blend across the overlap, build axis.
        combined_wl, combined_counts = self._stitch(
            vis_data, nir_data, nir_xover_include,
        )

        # 7) Combined stats + optional shifts.
        results = self._combined_results(combined_wl, combined_counts, analyzer_dict)

        # Surface per-sub scalars too so they end up alongside the
        # combined ones in the sfile.
        for key, value in vis_results.items():
            results[f'vis {key}'] = value
        for key, value in nir_results.items():
            results[f'nir {key}'] = value

        combined_df = pd.DataFrame({
            'Wavelength (nm)': combined_wl,
            'Counts': np.nan_to_num(combined_counts, nan=0.0),
        })
        return combined_df, results, {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _maybe_zero_nir_above_watchdog(self, nir_data, analyzer_dict):
        """
        If the mean NIR counts in the watchdog window are below threshold,
        treat the spectrum above the watchdog as 2nd-order 800 nm leakage
        and zero it out.

        Uses ``.iloc[hdx:]`` (positional) rather than ``.loc[hdx:]`` so the
        behaviour is independent of whatever index the DataFrame happens
        to carry.
        """
        wl = nir_data['Wavelength (nm)'].values
        counts = nir_data['Counts'].values
        low = analyzer_dict.get('check_2nd_order_800_low_wl', 1400)
        high = analyzer_dict.get('check_2nd_order_800_high_wl', 1500)
        thresh = analyzer_dict.get('check_2nd_order_800_thresh', 750)

        ldx = int(np.argmin(np.abs(wl - low)))
        hdx = int(np.argmin(np.abs(wl - high)))
        if hdx <= ldx:
            return nir_data

        check_mean = np.nanmean(counts[ldx:hdx])
        if check_mean < thresh:
            nir_data = nir_data.copy()
            nir_data.iloc[hdx:, nir_data.columns.get_loc('Counts')] = 0.0
        return nir_data

    @staticmethod
    def _nir_overlap_support(nir_dict, nir_data):
        """
        Indices (into the NIR wavelength axis) where every NIR response
        array is finite — i.e. where the NIR diagnostic actually carries
        signal that's eligible for the overlap blend.

        Falls back to all NIR indices if no response arrays are configured.
        """
        response_arrays = nir_dict.get('diagnostic_response_arrays', [])
        if not response_arrays:
            return np.arange(len(nir_data))
        valid = np.ones(len(nir_data), dtype=bool)
        for arr in response_arrays:
            valid &= ~np.isnan(np.asarray(arr))
        idx = np.where(valid)[0]
        if idx.size == 0:
            return np.arange(len(nir_data))
        return idx

    def _stitch(self, vis_data, nir_data, nir_xover_include):
        """
        Scale NIR onto VIS in the overlap region, linearly blend across
        the overlap, then resample onto a uniform VIS-spaced wavelength
        axis spanning min(vis) to max(nir).
        """
        vis_wl = vis_data['Wavelength (nm)'].values
        vis_counts = vis_data['Counts'].values
        nir_wl = nir_data['Wavelength (nm)'].values
        nir_counts = nir_data['Counts'].values

        nir_overlap_min_wl = float(np.min(nir_wl[nir_xover_include]))
        vis_xover_idcs = np.where(vis_wl > nir_overlap_min_wl)[0]
        nir_xover_idcs = np.where(nir_wl < float(np.max(vis_wl)))[0]

        if vis_xover_idcs.size == 0 or nir_xover_idcs.size == 0:
            # No overlap → just concatenate
            wl_combined = np.concatenate([vis_wl, nir_wl])
            counts_combined = np.concatenate([vis_counts, nir_counts])
        else:
            vis_tmp = vis_counts[vis_xover_idcs]
            nir_tmp = nir_counts[nir_xover_idcs]

            # Weighted means tip toward the side the weight emphasises.
            w_vis = 1.0 - np.arange(len(vis_xover_idcs)) / len(vis_xover_idcs)
            w_nir = np.arange(len(nir_xover_idcs)) / len(nir_xover_idcs)
            mean_vis = np.nanmean(w_vis * vis_tmp)
            mean_nir = np.nanmean(w_nir * nir_tmp)

            scale = mean_vis / mean_nir if mean_nir != 0 else 1.0
            nir_counts = nir_counts * scale
            nir_tmp_scaled = nir_tmp * scale

            # Interpolate scaled NIR overlap onto VIS wavelengths,
            # then blend.
            nir_on_vis = np.interp(
                vis_wl[vis_xover_idcs],
                nir_wl[nir_xover_idcs],
                nir_tmp_scaled,
            )
            w_nir_on_vis = np.arange(len(vis_xover_idcs)) / len(vis_xover_idcs)
            blended = vis_tmp * w_vis + nir_on_vis * w_nir_on_vis

            pre_vis_counts = vis_counts[:vis_xover_idcs[0]]
            post_nir_counts = nir_counts[nir_xover_idcs[-1]:]
            post_nir_wl = nir_wl[nir_xover_idcs[-1]:]

            counts_combined = np.concatenate([pre_vis_counts, blended, post_nir_counts])
            # Wavelengths: pre-overlap VIS + overlap VIS + post-overlap NIR
            wl_combined = np.concatenate([vis_wl, post_nir_wl])

        # Resample onto a uniform VIS-spaced axis. Note: this oversamples
        # NIR; a future option could keep the heterogeneous axis instead.
        dwl = float(np.mean(np.diff(vis_wl)))
        wl_uniform = np.arange(np.min(vis_wl), np.max(nir_wl) + dwl, dwl)
        counts_uniform = np.interp(wl_uniform, wl_combined, counts_combined)
        return wl_uniform, counts_uniform

    def _combined_results(self, wl, counts, analyzer_dict):
        max_counts = float(np.nanmax(counts)) if counts.size else np.nan
        mean_counts = float(np.nanmean(counts)) if counts.size else np.nan
        sum_counts = float(np.nansum(counts)) if counts.size else np.nan
        if counts.size:
            x0 = int(np.argmax(np.nan_to_num(counts, nan=-np.inf)))
            peak_wl = float(wl[x0])
        else:
            peak_wl = np.nan

        results = {
            'peak wl (nm)': peak_wl,
            'max counts': max_counts,
            'mean counts': mean_counts,
            'sum counts': sum_counts,
        }
        if analyzer_dict.get('calculate_red_blue_shifts', False):
            threshold = analyzer_dict.get('threshold_for_shifts', 600)
            shifts = self._shift_helper.compute_spectrum_shifts(counts, wl, threshold=threshold)
            cumsum = self._shift_helper.compute_cumulative_spectrum_shifts(wl, counts, threshold=threshold)
            results = merge_dicts_overwrite(results, shifts, cumsum)
        return results

    # ------------------------------------------------------------------
    # display + write (delegate to a fresh OpticalSpectrumAnalyzer view)
    # ------------------------------------------------------------------
    def display_data(self, data, display_dict=None, return_dict=None,
                     title=None, fig=None, ax=None):
        return self._shift_helper.display_data(
            data, display_dict=display_dict or self.display_dict,
            return_dict=return_dict, title=title, fig=fig, ax=ax,
        )

    def write_analyzed_data(self, data, analysis_dir, scan, shot_num, context=None):
        helper = OpticalSpectrumAnalyzer(
            diagnostic=self.output_diagnostic,
            output_diagnostic=self.output_diagnostic,
            output_file_ext=self.output_file_ext,
        )
        helper.write_analyzed_data(data, analysis_dir, scan, shot_num, context=context)
