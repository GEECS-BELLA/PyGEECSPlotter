# Relative beam energy on a camera, corrected for the beam's spectrum.
# Column math only: combines a camera's summed counts (ImageAnalyzer
# 'sum_counts') with the spectrum analyzer's '<camera> response ratio'.

import numpy as np

from PyGEECSPlotter.column_math_analysis import ColumnMathAnalyzer


class CameraEnergyAnalyzer(ColumnMathAnalyzer):
    """
    Beam energy on a camera relative to a reference, corrected for the
    camera's spectral response.

    A camera's background-subtracted summed counts ``C`` scale with the beam
    energy times ``R``, the fraction of the beam's spectrum the camera
    registers (``OpticalSpectrumAnalyzer.camera_response_ratio``, written as
    the ``'<camera> response ratio'`` column by a spectrum analyzer given
    ``analyzer_dict['camera_responses']``). So ``C / R`` is proportional to
    the energy even as the spectrum changes (e.g. redshifts along a scan),
    and::

        relative energy = (C / R) / reference

    with ``reference`` the mean ``C / R`` over a reference scan
    (``reference_from_scan``). It assumes the spectrometer samples the same
    beam the camera images.

    Parameters
    ----------
    counts_column : str
        Sfile column with the camera's summed counts, e.g.
        ``'CAM-HPD-M3Near sum_counts'``.
    ratio_column : str
        Sfile column with the response ratio, e.g.
        ``'SPEC-Combined CAM-HPD-M3Near response ratio'``.
    reference : float, optional
        Mean ``C / R`` of the reference. Without it only
        ``'corrected counts'`` (``C / R``) is returned.
    output_diagnostic : str, optional
        Prefix for the result columns.

    Results
    -------
    ``'corrected counts'`` and, with a reference, ``'relative energy'``;
    NaN when either column is missing / NaN or ``R <= 0``.
    """

    def __init__(self, counts_column, ratio_column, reference=None, output_diagnostic=None):
        super().__init__()
        self.counts_column = counts_column
        self.ratio_column = ratio_column
        self.reference = reference
        self.output_diagnostic = output_diagnostic

    def corrected_counts(self, context):
        """``C / R`` for one row (dict or Series), NaN if not computable."""
        counts = context.get(self.counts_column, np.nan)
        ratio = context.get(self.ratio_column, np.nan)
        try:
            counts, ratio = float(counts), float(ratio)
        except (TypeError, ValueError):
            return np.nan
        if not (np.isfinite(counts) and np.isfinite(ratio)) or ratio <= 0:
            return np.nan
        return counts / ratio

    def analyze_data(self, data, bg=None, context=None):
        corrected = self.corrected_counts(context or {})
        results = {'corrected counts': corrected}
        if self.reference is not None:
            results['relative energy'] = corrected / self.reference
        return None, results, {}

    def reference_from_scan(self, scan):
        """
        Mean ``C / R`` over ``scan.active_data``: a scan (or a filtered
        subset) that already has both columns. Pass the result as
        ``reference``.
        """
        missing = [c for c in (self.counts_column, self.ratio_column) if c not in scan.active_data]
        if missing:
            raise KeyError(f"Reference scan is missing column(s) {missing}.")
        values = [self.corrected_counts(row) for _, row in scan.active_data.iterrows()]
        if not np.any(np.isfinite(values)):
            raise ValueError("No reference shot has finite counts and response ratio.")
        return float(np.nanmean(values))
