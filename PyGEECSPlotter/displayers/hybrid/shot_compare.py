from typing import Optional, Dict, Any, Union, Callable, List

import numpy as np
import matplotlib.pyplot as plt

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer
from PyGEECSPlotter.displayers.scalars.scalar_vs_parameter import ScalarVsParameter
from PyGEECSPlotter.displayers.scalars.scan_stats_table import present_columns, format_value, draw_table
from PyGEECSPlotter.displayers._shot_selection import select_positions


class ShotCompare(ScanDisplayer):
    """
    Several diagnostics' views of the same shot side by side, one figure
    per selected shot: the scan-level view of a ``ShotCompareReader``.

    Each diagnostic panel is that diagnostic's own ``display_data`` for the
    shot, read through the reader's per-shot pipeline (``scan._iter_shots``),
    so it matches what ``analyze_scan(display_data=True)`` would show.
    ``extra_panels`` adds scan-level context after the diagnostics.

    Parameters
    ----------
    analyzer : ShotCompareReader
        Declares the diagnostics (one sub-analyzer each) in panel order.
    shots : optional
        Which active shots get a figure:
          * ``'first'`` (default) / ``'last'`` — the first / last active shot
            with the most diagnostics present
          * ``int`` — the row with that ``Shotnumber``
          * list of Shotnumbers (those not active are skipped)
          * ``'all'`` — every active shot
          * ``('sample', n)`` — ``n`` shots evenly spaced through the scan
          * callable(active_df) -> int row position
    extra_panels : list, optional
        Panels after the diagnostic ones, in the order given:
          * a ``ScalarVsParameter`` — drawn as usual, with this shot
            highlighted by an open ring
          * ``('stats', [columns])`` — a table of this shot's value of each
            column next to the scan's mean and std (over the active shots)
    nrows, ncols : int, optional
        Grid for all panels (diagnostics + extras). Defaults to the reader's
        ``display_dict`` ``'nrows'`` / ``'ncols'``, else near-square.
    bg : optional
        ``{diagnostic: bg}``, forwarded to the per-shot pipeline.
    display_dict : dict, optional
        Whole-figure overrides: ``'panel_aspect'``, ``'panel_width'``,
        ``'max_figsize'``, ``'figsize'`` (see ``ScanDisplayer._grid_figsize``),
        ``'titles'`` (``{diag: panel title}``), ``'highlight_color'``
        (default ``'r'``), ``'highlight_size'`` (marker area, default 200),
        ``'stats_fmt'`` (default ``'{:.4g}'``).

    Notes
    -----
    Creates its own figures; ``fig`` / ``ax`` arguments are ignored.
    ``display`` returns the last shot's figure, and keeps every
    ``(shot, fig)`` in ``figures``; ``save`` writes one PNG per shot
    (``..._shot{n:03d}``). ``last_export`` holds the shot numbers and each
    shot's scalar results (``'<scalar>'`` arrays, one entry per shot).
    """

    def __init__(
        self,
        analyzer,
        shots: Union[str, int, List[int], tuple, Callable] = 'first',
        extra_panels: Optional[List[Any]] = None,
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        bg=None,
        display_dict: Optional[Dict[str, Any]] = None,
        output_subdir: Optional[str] = None,
        timestamp_files: bool = True,
    ):
        super().__init__(name=analyzer.output_diagnostic, display_dict=display_dict,
                         output_subdir=output_subdir, timestamp_files=timestamp_files)
        self.analyzer = analyzer
        self.shots = shots
        self.extra_panels = list(extra_panels or [])
        for p in self.extra_panels:
            if not (isinstance(p, ScalarVsParameter)
                    or (isinstance(p, tuple) and len(p) == 2 and p[0] == 'stats')):
                raise ValueError(f"extra_panels entries must be a ScalarVsParameter or "
                                 f"('stats', [columns]), got {p!r}.")
        self.nrows = nrows
        self.ncols = ncols
        self.bg = bg
        self.figures = []

    def display(self, scan, fig=None, ax=None):
        if any(f'{name} file_list' not in scan.data.columns for name, _ in self.analyzer.inputs):
            self.analyzer.register_with_scan(scan)

        positions = select_positions(scan.active_data, self.analyzer.inputs, self.shots)
        if not positions:
            raise RuntimeError(f"{type(self).__name__}: no active shot matches shots={self.shots!r}.")

        panel_dict = {**self.analyzer.display_dict, **self.display_dict}
        if self.nrows is not None:
            panel_dict['nrows'] = self.nrows
        if self.ncols is not None:
            panel_dict['ncols'] = self.ncols
        n_diag = len(self.analyzer.sub_analyzers)
        n_panels = n_diag + len(self.extra_panels)
        nrows, ncols = self.analyzer.grid_shape(panel_dict, n=n_panels)

        self.figures = []
        shot_numbers, results_per_shot = [], []
        rows = scan.active_data.iloc[positions]
        for context, data, results, aux in scan._iter_shots(
                self.analyzer, bg=self.bg, show_progress=False, rows=rows):
            shot = int(context['Shotnumber'])
            fig, axes = plt.subplots(nrows, ncols, figsize=self._grid_figsize(ncols, nrows),
                                     constrained_layout=True, squeeze=False)
            flat = list(axes.ravel())
            self.analyzer.display_data(data, display_dict=panel_dict, fig=fig,
                                       ax=np.asarray(flat[:n_diag]), aux=aux)
            for a in flat[n_panels:]:
                a.set_visible(False)
            tables = []
            for k, panel in enumerate(self.extra_panels):
                a = flat[n_diag + k]
                if not isinstance(panel, ScalarVsParameter):
                    a = flat[n_diag + k] = self._widen(fig, axes, n_diag + k, n_panels, ncols)
                    tables.append(a)
                self._draw_extra(scan, context, panel, fig, a)
            self._apply_panel_aspect([a for a in flat[:n_panels] if a not in tables])
            fig.suptitle(scan.scan_data_title(f'shot {shot}'))
            self.figures.append((shot, fig))
            shot_numbers.append(shot)
            results_per_shot.append(results or {})

        keys = sorted({k for r in results_per_shot for k in r})
        self.last_export = {'shots': np.asarray(shot_numbers, dtype=float)}
        for k in keys:
            self.last_export[k] = np.asarray([r.get(k, np.nan) for r in results_per_shot],
                                             dtype=float)
        return self.figures[-1][1], axes

    # ------------------------------------------------------------------
    @staticmethod
    def _widen(fig, axes, k, n_panels, ncols):
        """Panel ``k`` as an axis spanning it and the unused cells to its
        right on the same row (tables need the width more than plots)."""
        row, col = divmod(k, ncols)
        end = col + 1
        while end < ncols and row * ncols + end >= n_panels:
            end += 1
        if end == col + 1:
            return axes[row, col]
        for c in range(col, end):
            axes[row, c].remove()
        return fig.add_subplot(axes[row, col].get_gridspec()[row, col:end])

    def _draw_extra(self, scan, context, panel, fig, a):
        if isinstance(panel, ScalarVsParameter):
            self._draw_scalar(scan, context, panel, fig, a)
        else:
            self._draw_stats(scan, context, panel[1], a)

    def _draw_scalar(self, scan, context, svp, fig, a):
        """The scan's ``svp`` plot with this shot ringed."""
        svp.display(scan, fig=fig, ax=a)
        x_col = svp.x_col or scan.scan_parameter
        x, y = context.get(x_col, np.nan), context.get(svp.y_col, np.nan)
        if np.isfinite(x) and np.isfinite(y):
            a.scatter([x], [y], s=self.display_dict.get('highlight_size', 200),
                      facecolors='none', lw=2, zorder=5,
                      edgecolors=self.display_dict.get('highlight_color', 'r'))
        a.set_title('')

    def _draw_stats(self, scan, context, columns, a):
        """This shot's value of each column, next to the scan's mean and std."""
        cols = present_columns(scan, list(columns), type(self).__name__)
        fmt = self.display_dict.get('stats_fmt', '{:.4g}')
        active = scan.active_data[cols]
        mean, std = active.mean(), active.std()
        cells = [[format_value(float(context.get(c, np.nan)), fmt),
                  format_value(mean[c], fmt), format_value(std[c], fmt)] for c in cols]
        # a word prefix every column shares (usually the diagnostic) goes in the title
        words = [c.split(' ') for c in cols]
        n_common = 0
        while (len(cols) > 1 and all(len(w) > n_common + 1 for w in words)
               and len({w[n_common] for w in words}) == 1):
            n_common += 1
        prefix = ' '.join(words[0][:n_common])
        labels = [' '.join(w[n_common:]) for w in words]
        draw_table(a, cells, labels, ['this shot', 'scan mean', 'scan std'], self.display_dict,
                   fit_axes=True)
        if prefix:
            a.set_title(prefix, fontsize='small')

    def save(self, fig, scan, suffix: str = "", dpi: int = 200):
        """One PNG per shot figure from the last ``display``; returns the paths."""
        return [super(ShotCompare, self).save(f, scan, suffix=f'{suffix}_shot{shot:03d}', dpi=dpi)
                for shot, f in self.figures]
