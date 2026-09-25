from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer


# Per-component plumbing: which analyzed DataFrame to read, and the names of
# its axis / amplitude columns.
COMPONENTS = {
    'temporal': ('temporal', 'Time [fs]', 'Temporal Amp', 'Time (fs)'),
    'spectral': ('spectral', 'Wavelength [nm]', 'Spectral Amp', 'Wavelength (nm)'),
}


class TraceWaterfall(ScanDisplayer):
    """
    Every shot's 1-D trace stacked into an image: one row per shot.

    The scan-level view for a 1-D diagnostic — x is time or wavelength, y is
    shot (or bin), colour is amplitude. Drift over a long scan shows up as
    curvature or broadening down the image in a way a grid of line plots
    cannot convey.

    Requires every shot to land on the same axis, so the analyzer must be
    resampling onto a common grid (for ``FrogAnalyzer``, pass ``t_grid`` /
    ``wl_grid`` in ``analyzer_dict``). Raises with that advice if the shots
    disagree.

    Parameters
    ----------
    analyzer : DiagnosticAnalyzer
        Per-shot analyzer whose ``analyze_data`` returns a dict of trace
        DataFrames (e.g. ``FrogAnalyzer``).
    bg : optional
        Background spec forwarded to the per-shot pipeline.
    component : {'temporal', 'spectral'}, optional
        Which trace to stack.
    y_column : str, optional
        Column of ``active_data`` to order the rows by, and to label the
        y-axis with. Defaults to ``scan.scan_parameter``, so shots are shown
        in order of the scanned control parameter rather than acquisition
        order. Pass ``False`` to keep the raw acquisition order (labelled
        ``'Shot (in scan order)'``).
    overlay_fwhm : bool, optional
        Overlay the per-shot width, centred on the centroid, as a line.
    display_dict : dict, optional
        Style overrides: ``figsize``, ``cmap``, ``vmin``, ``vmax``,
        ``xlims``, ``normalise_rows``, ``n_yticks``.
    """

    def __init__(
        self,
        analyzer,
        bg=None,
        component: str = 'temporal',
        y_column=None,
        overlay_fwhm: bool = False,
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        if component not in COMPONENTS:
            raise ValueError(
                f"component must be one of {sorted(COMPONENTS)}, got {component!r}."
            )
        diag = analyzer.output_diagnostic or analyzer.diagnostic
        super().__init__(name=f'{diag}_{component}_waterfall', display_dict=display_dict)
        self.analyzer = analyzer
        self.bg = bg
        self.component = component
        self.y_column = y_column
        self.overlay_fwhm = overlay_fwhm

    # ------------------------------------------------------------------
    def _collect(self, scan, y_col):
        """Stack every active shot's trace into a (n_shots, n_points) array."""
        key, axis_col, amp_col, _ = COMPONENTS[self.component]

        axis = None
        rows, shots, widths, centres, y_vals = [], [], [], [], []

        width_key, centre_key = self._stat_keys()

        for context, data, results, _ in scan._iter_shots(self.analyzer, bg=self.bg):
            if data is None or key not in data:
                continue
            df = data[key]
            this_axis = np.asarray(df[axis_col].values, dtype=float)

            if axis is None:
                axis = this_axis
            elif this_axis.shape != axis.shape or not np.allclose(this_axis, axis):
                raise ValueError(
                    f"{type(self).__name__} needs every shot on the same "
                    f"{axis_col} axis, but shot "
                    f"{context.get('Shotnumber', '?')} differs. Give the "
                    f"analyzer a common grid (e.g. analyzer_dict "
                    f"{{'t_grid': (lo, hi, n)}}) so shots can be stacked."
                )

            rows.append(np.asarray(df[amp_col].values, dtype=float))
            shots.append(context.get('Shotnumber', len(rows)))
            widths.append(results.get(width_key, np.nan))
            centres.append(results.get(centre_key, np.nan))
            y_vals.append(context.get(y_col, np.nan) if y_col else np.nan)

        if not rows:
            raise RuntimeError(f"{type(self).__name__} found no shots with data.")

        return (axis, np.vstack(rows), np.asarray(shots, dtype=float),
                np.asarray(widths, dtype=float), np.asarray(centres, dtype=float),
                np.asarray(y_vals, dtype=float))

    def _stat_keys(self):
        """Result keys holding the width and centroid for this component."""
        if self.component == 'temporal':
            return 'Temporal FWHM (fs)', 'centroid time (fs)'
        return 'Spectral FWHM (nm)', 'centroid wl (nm)'

    def display(self, scan, fig=None, ax=None):
        _, _, _, axis_label = COMPONENTS[self.component]

        y_col = self.y_column
        if y_col is None:
            y_col = scan.scan_parameter
        y_col = y_col if y_col else None

        axis, stack, shots, widths, centres, y_vals = self._collect(scan, y_col)

        if y_col is not None and np.any(np.isfinite(y_vals)):
            order = np.argsort(y_vals, kind='stable')
            stack, shots, widths, centres, y_vals = (
                stack[order], shots[order], widths[order], centres[order], y_vals[order]
            )
        else:
            y_col = None

        if self.display_dict.get('normalise_rows', False):
            peaks = np.nanmax(stack, axis=1, keepdims=True)
            with np.errstate(invalid='ignore', divide='ignore'):
                stack = np.where(peaks > 0, stack / peaks, stack)

        fig, ax = self._new_fig(fig, ax, figsize=(7, 6))

        y = np.arange(len(shots))
        im = ax.imshow(
            stack,
            aspect='auto',
            origin='lower',
            cmap=self.display_dict.get('cmap', 'viridis'),
            vmin=self.display_dict.get('vmin', None),
            vmax=self.display_dict.get('vmax', None),
            extent=[axis[0], axis[-1], y[0] - 0.5, y[-1] + 0.5],
            interpolation='nearest',
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(self.display_dict.get('cbar_label', 'Amplitude (norm.)'))

        if self.overlay_fwhm and np.any(np.isfinite(widths)):
            centre = np.where(np.isfinite(centres), centres, 0.0)
            for sign in (-0.5, 0.5):
                ax.plot(centre + sign * widths, y,
                        color=self.display_dict.get('overlay_color', 'w'),
                        lw=1.0, ls='--')

        ax.set_xlabel(axis_label)
        if y_col is not None:
            ax.set_ylabel(self.display_dict.get('ylabel', y_col))
            n_ticks = min(self.display_dict.get('n_yticks', 10), len(y))
            tick_idx = np.linspace(0, len(y) - 1, n_ticks).round().astype(int)
            ax.set_yticks(y[tick_idx])
            ax.set_yticklabels([f'{v:.4g}' for v in y_vals[tick_idx]])
        else:
            ax.set_ylabel(self.display_dict.get('ylabel', 'Shot (in scan order)'))

        xlims = self.display_dict.get('xlims', None)
        if xlims is not None:
            ax.set_xlim(xlims)

        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        ax.set_title(scan.scan_data_title(f'{diag} {self.component} waterfall'))

        self.last_export = {
            'stack': stack,
            'axis': axis,
            'shots': shots,
            'widths': widths,
            'centres': centres,
            'y_column': y_col if y_col is not None else '',
            'y_values': y_vals,
        }

        return fig, ax
