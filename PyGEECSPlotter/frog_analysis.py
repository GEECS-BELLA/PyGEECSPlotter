import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyGEECSPlotter.diagnostic_analyzer import DiagnosticAnalyzer
from PyGEECSPlotter.navigation_utils import get_analysed_shot_save_path
from PyGEECSPlotter.utils import get_lineout_width


class FrogAnalyzer(DiagnosticAnalyzer):
    """
    Per-shot analyzer for FROG-retrieved pulses.

    Each shot's file is a tab-separated table holding two independent traces
    side by side:

        Wavelength [nm]  Spectral Amp  Spectral Phase
        Time [fs]        Temporal Amp  Temporal Phase
        ER  EI                                  (complex field, unused here)

    Pipeline:

        1) Trim the zero padding off each trace (they pad independently)
        2) Optionally normalise each amplitude to peak 1
        3) Optionally resample both onto a common grid, so that shots with
           different retrieval grids can be averaged or stacked
        4) Stats: pulse duration, bandwidth, peak power, centroids

    Notes
    -----
    ``Temporal Amp`` is an *intensity* (it matches ER^2 + EI^2 normalised to
    1), so its FWHM is the conventional intensity pulse duration and
    ``max(I) / integral(I dt)`` is a peak power per unit pulse energy.

    The files are zero padded, and the spectral and temporal halves are padded
    independently: a 1024-row file commonly carries only 512 valid time points,
    and the wavelength column is sometimes short too. The padding is detected
    by finding where the axis stops advancing monotonically (and stripping any
    trailing zeros) rather than assumed to be half the file, since the row
    count varies from shot to shot.

    Because ``analyze_data`` returns a dict of two DataFrames rather than one
    array, this analyzer does not work with ``scan.mean_std_diagnostic`` /
    ``scan.aggregate_per_bin`` (which expect array-like per-shot data). Use
    ``TraceWaterfall`` and ``TraceMeanPerBin`` for scan-level views; they
    aggregate the traces themselves.
    """

    # Columns as written by the FROG retrieval.
    SPECTRAL_COLS = ('Wavelength [nm]', 'Spectral Amp', 'Spectral Phase')
    TEMPORAL_COLS = ('Time [fs]', 'Temporal Amp', 'Temporal Phase')

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
        return pd.read_csv(filename, sep='\t')

    def analyze_data(self, data, bg=None, context=None, analyzer_dict=None):
        if analyzer_dict is None:
            analyzer_dict = self.analyzer_dict or {}

        if data is None:
            return None, {}, {}

        wl, s_amp, s_phase = self._extract(data, self.SPECTRAL_COLS, ascending=True)
        t, t_amp, t_phase = self._extract(data, self.TEMPORAL_COLS, ascending=True)

        if analyzer_dict.get('normalise', True):
            s_amp = self._normalise(s_amp)
            t_amp = self._normalise(t_amp)

        # Stats are computed on the retrieved grid, before any resampling, so
        # that interpolation cannot shift a width or a centroid.
        results = {}
        results.update(self._temporal_stats(t, t_amp, analyzer_dict))
        results.update(self._spectral_stats(wl, s_amp, s_phase))

        if analyzer_dict.get('resample', True):
            t_grid = self._resolve_grid(analyzer_dict.get('t_grid', None), t)
            wl_grid = self._resolve_grid(analyzer_dict.get('wl_grid', None), wl)
            t, t_amp, t_phase = self._resample(t, t_grid, t_amp, t_phase)
            wl, s_amp, s_phase = self._resample(wl, wl_grid, s_amp, s_phase)

        data_out = {
            'temporal': pd.DataFrame({
                'Time [fs]': t,
                'Temporal Amp': t_amp,
                'Temporal Phase': t_phase,
            }),
            'spectral': pd.DataFrame({
                'Wavelength [nm]': wl,
                'Spectral Amp': s_amp,
                'Spectral Phase': s_phase,
            }),
        }

        return data_out, results, {}

    def display_data(self, data, display_dict=None, return_dict=None, title=None, fig=None, ax=None):
        """
        Plot the retrieved pulse: amplitude on the main axis, phase twinned on
        the right.

        ``display_dict['panel']`` selects ``'both'`` (default, spectral and
        temporal side by side), ``'temporal'`` or ``'spectral'``. A supplied
        ``fig`` / ``ax`` is only honoured for the single-panel modes, which is
        what lets the grid displayers drive this.
        """
        if display_dict is None:
            display_dict = self.display_dict

        panel = display_dict.get('panel', 'both')

        if panel == 'both':
            if fig is None or ax is None:
                fig, ax = plt.subplots(
                    1, 2,
                    constrained_layout=True,
                    figsize=display_dict.get('figsize', (11, 4)),
                )
            axes = np.atleast_1d(ax)
            self._plot_spectral(fig, axes[0], data['spectral'], display_dict)
            self._plot_temporal(fig, axes[1], data['temporal'], display_dict)
            if title is not None:
                fig.suptitle(title)
            return fig, ax

        if fig is None or ax is None:
            fig, ax = plt.subplots(
                constrained_layout=True,
                figsize=display_dict.get('figsize', (6, 4)),
            )

        if panel == 'temporal':
            self._plot_temporal(fig, ax, data['temporal'], display_dict)
        elif panel == 'spectral':
            self._plot_spectral(fig, ax, data['spectral'], display_dict)
        else:
            raise ValueError(
                f"display_dict['panel'] must be 'both', 'temporal' or "
                f"'spectral', got {panel!r}."
            )

        if title is not None:
            ax.set_title(title)
        return fig, ax

    def write_analyzed_data(self, data, analysis_dir, scan, shot_num, context=None):
        """Write the two traces as ``*_temporal`` / ``*_spectral`` files."""
        diagnostic = self.output_diagnostic or self.diagnostic
        file_ext = self.output_file_ext or '.txt'
        for key, suffix in (('temporal', '_temporal'), ('spectral', '_spectral')):
            save_path = get_analysed_shot_save_path(
                analysis_dir, diagnostic, scan, shot_num, file_ext,
                append_info=suffix,
            )
            data[key].to_csv(save_path, sep='\t', index=False, float_format='%.6g')

    # ------------------------------------------------------------------
    # Trimming / extraction
    # ------------------------------------------------------------------
    @staticmethod
    def valid_length(axis):
        """
        Number of leading rows before the zero padding starts.

        The retrieval writes a monotonic axis and pads the remainder with
        zeros, so the padding begins where the axis first stops moving in its
        initial direction. Returns the full length if it never does.

        A descending axis needs the extra zero-strip below: its step down into
        the padding is still a decrease, so monotonicity alone does not spot
        the first padded row (only the flat 0 -> 0 after it).
        """
        axis = np.asarray(axis, dtype=float)
        if axis.size < 2:
            return axis.size

        steps = np.diff(axis)
        direction = steps[0]
        if direction == 0:
            return 1

        stalled = np.where(steps * np.sign(direction) <= 0)[0]
        n = int(stalled[0]) + 1 if stalled.size else axis.size

        # Trailing exact zeros are padding. A real axis may pass through 0
        # (time does), but only in its interior, never at the end of the run.
        while n > 1 and axis[n - 1] == 0:
            n -= 1

        return n

    def _extract(self, data, cols, ascending=True):
        """Pull one trace out of the frame, trimmed and optionally sorted."""
        axis_col, amp_col, phase_col = cols
        axis = np.asarray(data[axis_col].values, dtype=float)
        n = self.valid_length(axis)

        axis = axis[:n]
        amp = np.asarray(data[amp_col].values, dtype=float)[:n]
        phase = np.asarray(data[phase_col].values, dtype=float)[:n]

        # Wavelength is written descending; np.interp and the plots both want
        # an ascending axis.
        if ascending and axis.size > 1 and axis[0] > axis[-1]:
            axis, amp, phase = axis[::-1], amp[::-1], phase[::-1]

        return axis, amp, phase

    @staticmethod
    def _normalise(amp):
        peak = np.nanmax(amp) if amp.size else np.nan
        return amp / peak if np.isfinite(peak) and peak > 0 else amp

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_grid(grid, axis):
        """
        Turn a grid spec into an array.

        Accepts an explicit array, a ``(min, max, n)`` triple, or ``None`` to
        keep the shot's own axis (which leaves shots un-alignable, so the
        scan-level displayers ask for an explicit grid).
        """
        if grid is None:
            return np.asarray(axis, dtype=float)
        grid = np.asarray(grid, dtype=float)
        if grid.size == 3 and grid.ndim == 1:
            lo, hi, n = grid
            return np.linspace(lo, hi, int(n))
        return grid

    @staticmethod
    def _resample(axis, grid, amp, phase):
        """
        Interpolate a trace onto ``grid``.

        Points outside the shot's own support become NaN rather than being
        clamped to the edge value, so a short retrieval does not masquerade as
        a wide flat pulse.
        """
        grid = np.asarray(grid, dtype=float)
        if axis.size < 2:
            nan = np.full(grid.shape, np.nan)
            return grid, nan, nan.copy()

        outside = (grid < axis[0]) | (grid > axis[-1])
        amp_out = np.interp(grid, axis, amp)
        phase_out = np.interp(grid, axis, phase)
        amp_out[outside] = np.nan
        phase_out[outside] = np.nan
        return grid, amp_out, phase_out

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _temporal_stats(self, t, amp, analyzer_dict):
        nan_result = {
            'Temporal FWHM (fs)': np.nan,
            'Peak Power (TW/J)': np.nan,
            'effective duration (fs)': np.nan,
            'centroid time (fs)': np.nan,
        }
        if t.size < 2 or not np.any(np.isfinite(amp)) or np.nanmax(amp) <= 0:
            return nan_result

        width_at = analyzer_dict.get('temporal_width_at', 0.5)
        fwhm = self._width_on_axis(t, amp, width_at)
        peak_power = self.get_TW_per_J(t, amp)
        integral = np.trapezoid(amp, t)
        peak = np.nanmax(amp)

        return {
            'Temporal FWHM (fs)': fwhm,
            'Peak Power (TW/J)': peak_power,
            'effective duration (fs)': integral / peak if peak > 0 else np.nan,
            'centroid time (fs)': self._centroid(t, amp),
        }

    def _spectral_stats(self, wl, amp, phase):
        nan_result = {
            'Spectral FWHM (nm)': np.nan,
            'centroid wl (nm)': np.nan,
            'phase support low wl (nm)': np.nan,
            'phase support high wl (nm)': np.nan,
        }
        if wl.size < 2 or not np.any(np.isfinite(amp)) or np.nanmax(amp) <= 0:
            return nan_result

        # The retrieval only writes a non-zero phase where it had signal, so
        # the extent of that region is a useful plotting / sanity bracket.
        has_phase = np.where(phase > 0)[0]
        if has_phase.size:
            low_wl, high_wl = float(wl[has_phase[0]]), float(wl[has_phase[-1]])
        else:
            low_wl = high_wl = np.nan

        return {
            'Spectral FWHM (nm)': self._width_on_axis(wl, amp, 0.5),
            'centroid wl (nm)': self._centroid(wl, amp),
            'phase support low wl (nm)': low_wl,
            'phase support high wl (nm)': high_wl,
        }

    @staticmethod
    def _width_on_axis(axis, amp, width_at=0.5):
        """
        Width of ``amp`` at ``width_at`` of its peak, in axis units.

        ``get_lineout_width`` returns ``(width, high_idx, low_idx)`` — note the
        high index comes first.
        """
        _, hdx, ldx = get_lineout_width(amp, from_center=False, width_at=width_at)
        if not np.isfinite(hdx) or not np.isfinite(ldx):
            return np.nan
        return float(axis[int(hdx)] - axis[int(ldx)])

    @staticmethod
    def _centroid(axis, amp):
        weights = np.nan_to_num(amp, nan=0.0)
        total = np.sum(weights)
        return float(np.sum(axis * weights) / total) if total > 0 else np.nan

    @staticmethod
    def get_TW_per_J(t_fs, intensity, t_min=None, t_max=None):
        """
        Peak laser power per joule of pulse energy, in TW/J.

        Parameters
        ----------
        t_fs : array
            Time axis in femtoseconds.
        intensity : array
            FROG-retrieved temporal intensity in arbitrary units (the
            normalisation cancels).
        t_min, t_max : float or None
            Integration bounds in fs. If None, use the full range.

        Returns
        -------
        float
            ``max(I) / integral(I dt)``. With time in fs the 1e15 s/fs and the
            1e-12 W/TW combine into the factor of 1000.
        """
        t_fs = np.asarray(t_fs, dtype=float)
        intensity = np.asarray(intensity, dtype=float)

        mask = np.ones_like(t_fs, dtype=bool)
        if t_min is not None:
            mask &= t_fs >= t_min
        if t_max is not None:
            mask &= t_fs <= t_max
        if mask.sum() < 2:
            return np.nan

        t_sel = t_fs[mask]
        i_sel = intensity[mask]
        integral = np.trapezoid(i_sel, t_sel)
        if not np.isfinite(integral) or integral <= 0:
            return np.nan

        return float(1000 * np.nanmax(i_sel) / integral)

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------
    def _plot_temporal(self, fig, ax, df, display_dict):
        t = df['Time [fs]'].values
        amp = df['Temporal Amp'].values
        phase = df['Temporal Phase'].values

        ax.plot(t, amp, color=display_dict.get('amp_color', None))
        ax.set_xlabel(display_dict.get('t_label', 'Time (fs)'))
        ax.set_ylabel(display_dict.get('t_amp_label', 'Temporal Amp (norm.)'))

        xlims = display_dict.get('t_lims', None)
        if xlims is None:
            xlims = self._support_lims(t, amp, display_dict)
        if xlims is not None:
            ax.set_xlim(xlims)

        if display_dict.get('show_phase', True):
            ax2 = ax.twinx()
            ax2.plot(t, phase, color=display_dict.get('phase_color', 'C1'))
            ax2.set_ylabel(display_dict.get('t_phase_label', 'Temporal Phase (rad)'))

    def _plot_spectral(self, fig, ax, df, display_dict):
        wl = df['Wavelength [nm]'].values
        amp = df['Spectral Amp'].values
        phase = df['Spectral Phase'].values

        ax.plot(wl, amp, color=display_dict.get('amp_color', None))
        ax.set_xlabel(display_dict.get('wl_label', 'Wavelength (nm)'))
        ax.set_ylabel(display_dict.get('s_amp_label', 'Spectral Amp (norm.)'))

        xlims = display_dict.get('wl_lims', None)
        if xlims is None:
            xlims = self._support_lims(wl, amp, display_dict)
        if xlims is not None:
            ax.set_xlim(xlims)

        if display_dict.get('show_phase', True):
            ax2 = ax.twinx()
            ax2.plot(wl, phase, color=display_dict.get('phase_color', 'C1'))
            ax2.set_ylabel(display_dict.get('s_phase_label', 'Spectral Phase (rad)'))

    @staticmethod
    def _support_lims(axis, amp, display_dict):
        """
        Zoom to where the trace actually has signal.

        The retrieved grid is much wider than the pulse, so plotting the full
        axis buries the pulse in a few pixels.
        """
        frac = display_dict.get('support_threshold', 0.01)
        finite = np.isfinite(amp)
        if not np.any(finite):
            return None
        peak = np.nanmax(amp)
        if not np.isfinite(peak) or peak <= 0:
            return None

        idx = np.where(finite & (amp > frac * peak))[0]
        if idx.size < 2:
            return None

        lo, hi = axis[idx[0]], axis[idx[-1]]
        pad = display_dict.get('support_pad', 0.25) * (hi - lo)
        return [lo - pad, hi + pad]
