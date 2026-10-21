import textwrap
import warnings
from typing import Optional, Dict, Any, Iterable

import numpy as np

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer

_DESCRIBE_STATS = ('count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max')


def present_columns(scan, columns, owner='ScanStatsTable'):
    """``columns`` found in ``scan.active_data``, warning about the rest."""
    cols = [c for c in columns if c in scan.active_data.columns]
    missing = [c for c in columns if c not in cols]
    if missing:
        warnings.warn(f"{owner}: columns not in active_data: {missing}")
    if not cols:
        raise ValueError("None of the requested columns are present in scan.active_data.")
    return cols


def format_value(v, fmt='{:.4g}'):
    return fmt.format(v) if np.isfinite(v) else '—'


def draw_table(ax, cells, row_labels, col_labels, display_dict=None, fit_axes=False):
    """
    ``cells`` (list of rows of strings) as a matplotlib table in ``ax``,
    which is turned off. Header and row-label cells are shaded; headers
    wrap at ``display_dict['wrap']`` characters (default 18). Also
    ``display_dict['fontsize']`` (default 8) and ``'header_color'``
    (default light grey).

    ``fit_axes=False`` sizes columns to their text (for a figure sized to
    the table). ``fit_axes=True`` keeps the whole table, row labels
    included, inside ``ax`` (for a table sharing a figure with other
    panels), wrapping long row labels. Returns the table.
    """
    display_dict = display_dict or {}
    wrap = display_dict.get('wrap', 18)
    ax.axis('off')
    wrapped_cols = ['\n'.join(textwrap.wrap(str(c), wrap)) or str(c) for c in col_labels]
    if fit_axes:
        # row labels as an ordinary first column, so bbox holds them too
        labels = ['\n'.join(textwrap.wrap(str(r), 2 * wrap)) for r in row_labels]
        tbl = ax.table(cellText=[[l] + list(row) for l, row in zip(labels, cells)],
                       colLabels=[''] + wrapped_cols, bbox=[0, 0, 1, 1], cellLoc='center')
        label_col = 0
        n_value = len(col_labels)
        widths = [0.45] + [0.55 / n_value] * n_value
        for (r, c), cell in tbl.get_celld().items():
            cell.set_width(widths[c])
    else:
        tbl = ax.table(cellText=cells, rowLabels=[str(r) for r in row_labels],
                       colLabels=wrapped_cols, loc='center', cellLoc='center')
        label_col = -1
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(display_dict.get('fontsize', 8))
    if not fit_axes:
        tbl.auto_set_column_width(list(range(-1, len(col_labels))))
    header_lines = max(c.count('\n') + 1 for c in wrapped_cols)
    header_color = display_dict.get('header_color', '0.9')
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            if not fit_axes:
                cell.set_height(cell.get_height() * header_lines)
            cell.set_facecolor(header_color)
            cell.set_text_props(weight='bold')
        elif c == label_col:
            cell.set_facecolor(header_color)
    return tbl


class ScanStatsTable(ScanDisplayer):
    """
    Summary statistics of scalar columns over the active shots, drawn as a
    matplotlib table: ``scan.active_data[columns].describe()``, one row per
    column and one column per statistic.

    Parameters
    ----------
    columns : iterable of str
        Scalar columns to summarise. Missing columns are dropped with a
        warning.
    stats : iterable of str, optional
        ``describe()`` rows to show, in order (default: count, mean, std,
        min, 25%, 50%, 75%, max).
    fmt : str, optional
        Number format for the cells (default ``'{:.4g}'``).
    display_dict : dict, optional
        ``figsize``, ``fontsize`` (default 8), ``header_color`` (default
        light grey), ``wrap`` (characters per header line, default 18).

    Notes
    -----
    Draws into ``ax`` if one is given (e.g. next to another plot), else
    makes a figure sized to the table. ``last_export`` holds ``values``,
    ``row_labels`` and ``col_labels``; the table as a DataFrame is kept as
    ``last_table``.
    """

    def __init__(
        self,
        columns: Iterable[str],
        stats: Iterable[str] = _DESCRIBE_STATS,
        fmt: str = '{:.4g}',
        display_dict: Optional[Dict[str, Any]] = None,
        output_subdir: Optional[str] = None,
        timestamp_files: bool = True,
    ):
        super().__init__(name='stats', display_dict=display_dict, output_subdir=output_subdir,
                         timestamp_files=timestamp_files)
        self.columns = list(columns)
        self.stats = list(stats)
        self.fmt = fmt
        self.last_table = None

    def display(self, scan, fig=None, ax=None):
        cols = present_columns(scan, self.columns, type(self).__name__)
        desc = scan.active_data[cols].describe()
        stats = [s for s in self.stats if s in desc.index]
        table = desc.loc[stats].T
        values = table.to_numpy(dtype=float)
        cells = [[format_value(v, self.fmt) for v in row] for row in values]
        self.last_table = table
        self.last_export = {'values': values,
                            'row_labels': np.asarray(cols, dtype=object),
                            'col_labels': np.asarray(stats, dtype=object)}

        if fig is None or ax is None:
            label_w = 0.08 * min(max(len(c) for c in cols), 50) + 0.5
            fig, ax = self._new_fig(fig, ax, figsize=(label_w + 0.9 * len(stats),
                                                      0.9 + 0.22 * (len(cols) + 2)))
        draw_table(ax, cells, cols, stats, self.display_dict)
        ax.set_title(scan.scan_data_title(f'{len(scan.active_data)} active shots'))
        return fig, ax
