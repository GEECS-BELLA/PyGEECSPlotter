from typing import Optional, Dict, Any

import numpy as np

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer
from PyGEECSPlotter.displayers._lineout_binning import on_grid


class LineoutWaterfall(ScanDisplayer):
    """
    Every shot's image lineout stacked into one image: one row per shot.

    ``TraceWaterfall``'s counterpart for analyzers whose ``aux`` is a flat
    dict of coordinate/lineout pairs (e.g. ``ImageAnalyzer`` with
    ``generate_lineouts=True``, which returns ``{'x': x, 'y': y,
    'x_lo': x_lo, 'y_lo': y_lo}``) rather than a dict of trace DataFrames.
    x is the lineout coordinate, y is shot (or bin), colour is amplitude.

    Every shot's lineout is put on the first shot's coordinate axis (see
    ``align``) when a shot's axis is shifted (e.g. ``centroid_method=
    'centroid'`` sets x = 0 at each shot's own centroid); points off a
    shot's axis are NaN.

    Parameters
    ----------
    analyzer : DiagnosticAnalyzer
        Per-shot analyzer whose ``analyze_data`` returns a lineout-style
        ``aux`` dict.
    axis : str, optional
        Which coordinate to stack — the analyzer's aux dict must contain
        both ``axis`` and ``f'{axis}_lo'``. Default ``'x'``.
    bg : optional
        Background spec forwarded to the per-shot pipeline.
    y_column : str, optional
        Column of ``active_data`` to order the rows by, and to label the
        y-axis with. Defaults to ``scan.scan_parameter``, so shots are shown
        in order of the scanned control parameter rather than acquisition
        order. Pass ``False`` to keep the raw acquisition order (labelled
        ``'Shot (in scan order)'``).
    align : {'snap', 'interp'}, optional
        How shots whose coordinate axes are shifted (e.g. ImageAnalyzer
        ``centroid_method='centroid'``) are put on one axis: ``'snap'``
        (default) shifts by whole pixels, exact, needing ImageAnalyzer's
        ``round_centroid=True`` (its default); ``'interp'`` interpolates,
        for sub-pixel origins.
    overlay_y_values : bool, optional
        Mark each row's ``y_column`` value (the scan parameter in
        acquisition order) as a short tick against a second x-axis across
        the top, so non-uniform steps show. Colour:
        ``display_dict['y_overlay_color']`` (default black).
    display_dict : dict, optional
        Style overrides: ``figsize``, ``cmap``, ``vmin``, ``vmax``,
        ``xlims``, ``normalise_rows``, ``n_yticks``.
    """

    def __init__(
        self,
        analyzer,
        axis: str = 'x',
        bg=None,
        y_column=None,
        align: str = 'snap',
        overlay_y_values: bool = False,
        display_dict: Optional[Dict[str, Any]] = None,
        output_subdir: Optional[str] = None,
        timestamp_files: bool = True,
    ):
        diag = analyzer.output_diagnostic or analyzer.diagnostic
        super().__init__(name=f'{diag}_{axis}_lineout_waterfall', display_dict=display_dict,
                          output_subdir=output_subdir, timestamp_files=timestamp_files)
        self.analyzer = analyzer
        self.axis = axis
        self.bg = bg
        self.y_column = y_column
        self.align = align
        self.overlay_y_values = overlay_y_values

    # ------------------------------------------------------------------
    def _collect(self, scan, y_col):
        """Stack every active shot's lineout into a (n_shots, n_points) array."""
        coord_key, lo_key = self.axis, f'{self.axis}_lo'

        coord = None
        rows, y_vals = [], []

        for context, _, _, aux in scan._iter_shots(self.analyzer, bg=self.bg):
            if not aux or coord_key not in aux or lo_key not in aux:
                continue
            this_coord = np.asarray(aux[coord_key], dtype=float)

            if coord is None:
                coord = this_coord
            # first shot's axis is the reference; shots on a shifted axis
            # (e.g. per-shot centroid origin) are aligned onto it
            rows.append(on_grid(this_coord, np.asarray(aux[lo_key], dtype=float), coord, self.align))
            y_vals.append(context.get(y_col, np.nan) if y_col else np.nan)

        if not rows:
            raise RuntimeError(
                f"{type(self).__name__} found no shots with "
                f"{coord_key!r}/{lo_key!r} in aux."
            )

        return coord, np.vstack(rows), np.asarray(y_vals, dtype=float)

    def display(self, scan, fig=None, ax=None):
        y_col = self.y_column
        if y_col is None:
            y_col = scan.scan_parameter
        y_col = y_col if y_col else None
        # column behind y_vals: the y_column, else (acquisition order) the
        # scan parameter, so overlay_y_values still has values to show
        val_col = y_col or scan.scan_parameter

        coord, stack, y_vals = self._collect(scan, val_col)

        if y_col is not None and np.any(np.isfinite(y_vals)):
            order = np.argsort(y_vals, kind='stable')
            stack, y_vals = stack[order], y_vals[order]
        else:
            y_col = None

        if self.display_dict.get('normalise_rows', False):
            peaks = np.nanmax(stack, axis=1, keepdims=True)
            with np.errstate(invalid='ignore', divide='ignore'):
                stack = np.where(peaks > 0, stack / peaks, stack)

        fig, ax = self._new_fig(fig, ax, figsize=(7, 6))

        y = np.arange(stack.shape[0])
        im = ax.imshow(
            stack,
            aspect='auto',
            origin='lower',
            cmap=self.display_dict.get('cmap', 'viridis'),
            vmin=self.display_dict.get('vmin', None),
            vmax=self.display_dict.get('vmax', None),
            extent=[coord[0], coord[-1], y[0] - 0.5, y[-1] + 0.5],
            interpolation='nearest',
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(self.display_dict.get('cbar_label', 'Amplitude'))

        ax.set_xlabel(self.display_dict.get('xlabel', self.axis))
        if y_col is not None:
            ax.set_ylabel(self.display_dict.get('ylabel', y_col))
            n_ticks = min(self.display_dict.get('n_yticks', 10), len(y))
            tick_idx = np.linspace(0, len(y) - 1, n_ticks).round().astype(int)
            ax.set_yticks(y[tick_idx])
            ax.set_yticklabels([f'{v:.4g}' for v in y_vals[tick_idx]])
        else:
            ax.set_ylabel(self.display_dict.get('ylabel', 'Shot (in scan order)'))
        if self.overlay_y_values:
            self._overlay_row_values(ax, y_vals, label=val_col)

        xlims = self.display_dict.get('xlims', None)
        if xlims is not None:
            ax.set_xlim(xlims)

        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        ax.set_title(scan.scan_data_title(f'{diag} {self.axis} lineout waterfall'))

        self.last_export = {
            'stack': stack,
            'coord': coord,
            'y_column': y_col if y_col is not None else '',
            'y_values': y_vals,
        }

        return fig, ax
