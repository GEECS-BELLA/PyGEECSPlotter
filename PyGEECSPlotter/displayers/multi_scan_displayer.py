# Base class for displays that compare several scans on one figure.
# Like ScanDisplayer, but display() takes a list of scans, and output goes
# to an explicit directory (default: the first scan's analysis dir), since
# no single scan owns the figure.

import os
import time
from typing import Optional, Dict, Any, Sequence

import numpy as np

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer, _UNSAFE_FILENAME_CHARS

_LABEL_STATS = ('mean', 'median', 'max', 'min')


def scan_label(scan, label_by='scan', label_fmt='{:.3g}', all_scans=None, index=0):
    """
    Legend label for one scan of a comparison.

    ``label_by``:
      * ``'scan'`` — ``'Scan 023'``, or ``'24_0131 Scan 023'`` if
        ``all_scans`` span more than one day
      * ``(stat, column)`` with ``stat`` in mean / median / max / min —
        ``'<column> = <value>'`` over that scan's active shots
      * a list of strings — ``label_by[index]``, used as given (e.g. to tell
        apart two differently filtered copies of the same scan)
      * callable(scan) -> str
    """
    if callable(label_by):
        return label_by(scan)
    if isinstance(label_by, (list, np.ndarray)):
        return str(label_by[index])
    if isinstance(label_by, tuple):
        stat, column = label_by
        if stat not in _LABEL_STATS:
            raise ValueError(f"label_by stat must be one of {_LABEL_STATS}, got {stat!r}.")
        if column not in scan.active_data.columns:
            raise KeyError(f"label_by column '{column}' not in scan.active_data.")
        value = getattr(scan.active_data[column], stat)()
        return f'{column} = {label_fmt.format(value)}'
    if label_by != 'scan':
        raise ValueError(f"label_by must be 'scan', (stat, column), a list or a callable, "
                         f"got {label_by!r}.")
    days = {os.path.basename(os.path.normpath(s.top_dir)) for s in (all_scans or [scan])}
    label = f'Scan {int(scan.scan):03d}'
    if len(days) > 1:
        label = f'{os.path.basename(os.path.normpath(scan.top_dir))} {label}'
    return label


class MultiScanDisplayer(ScanDisplayer):
    """
    Base class for figures comparing two or more scans.

    Subclasses implement ``display(scans, fig=None, ax=None)``, reading each
    scan's ``active_data`` (so each keeps its own filters and binning), and
    set ``self.last_export``. Run with ``display_scans``:

        display_scans(MyMultiScanDisplayer(...), [scan_a, scan_b], save=True)

    Files are named ``Scans023-024_{name}[_{timestamp}]`` and go to
    ``output_dir`` if given, else the first scan's analysis directory; either
    way nested under ``output_subdir`` if set.

    Parameters
    ----------
    label_by, label_fmt :
        How each scan is named in the legend; see ``scan_label``.
    output_dir : str, optional
        Where saved / exported files go. No default data path: falls back to
        the first scan's analysis directory.
    """

    def __init__(
        self,
        name: str = 'multiscan',
        label_by='scan',
        label_fmt: str = '{:.3g}',
        display_dict: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        output_subdir: Optional[str] = None,
        timestamp_files: bool = True,
    ):
        super().__init__(name=name, display_dict=display_dict, output_subdir=output_subdir,
                         timestamp_files=timestamp_files)
        self.label_by = label_by
        self.label_fmt = label_fmt
        self.output_dir = output_dir

    def display(self, scans, fig=None, ax=None):
        raise NotImplementedError(
            f"{type(self).__name__} must implement display(scans, fig, ax)."
        )

    # ------------------------------------------------------------------
    def _check_scans(self, scans):
        scans = list(scans)
        if not scans:
            raise ValueError(f"{type(self).__name__}: no scans given.")
        if isinstance(self.label_by, (list, np.ndarray)) and len(self.label_by) != len(scans):
            raise ValueError(f"label_by has {len(self.label_by)} labels for {len(scans)} scans.")
        return scans

    def _labels(self, scans):
        return [scan_label(s, self.label_by, self.label_fmt, all_scans=scans, index=k)
                for k, s in enumerate(scans)]

    def _colors(self, n):
        cmap = self.display_dict.get('cmap')
        if cmap is None:
            return [f'C{k % 10}' for k in range(n)]
        import matplotlib.pyplot as plt
        cm = plt.get_cmap(cmap)
        return [cm(v) for v in np.linspace(0, 1, n)] if n > 1 else [cm(0.5)]

    def _title(self, scans, append_text=None):
        """First scan's title when there is one scan number; else the scan
        numbers listed, plus ``append_text``."""
        nums = list(dict.fromkeys(int(s.scan) for s in scans))
        if len(nums) == 1:
            return scans[0].scan_data_title(append_text)
        title = 'Scans ' + ', '.join(f'{n:03d}' for n in nums)
        return f'{title}: {append_text}' if append_text else title

    # ------------------------------------------------------------------
    # Output: overrides of ScanDisplayer's, taking the list of scans
    # ------------------------------------------------------------------
    def _output_dir(self, scans):
        out = self.output_dir or scans[0].get_scan_data_analysis_dir(make_dir=True)
        if self.output_subdir:
            out = os.path.join(out, self.output_subdir)
        os.makedirs(out, exist_ok=True)
        return out

    def _output_stem(self, scans, suffix: str = ""):
        nums = list(dict.fromkeys(int(s.scan) for s in scans))
        safe_name = _UNSAFE_FILENAME_CHARS.sub('_', f"{self.name}{suffix}")
        stem = 'Scans' + '-'.join(f'{n:03d}' for n in nums) + f'_{safe_name}'
        if self.timestamp_files:
            stem += time.strftime("_%Y%m%d-%H%M%S")
        return stem

    def save(self, fig, scans, suffix: str = "", dpi: int = 200):
        path = os.path.join(self._output_dir(scans), self._output_stem(scans, suffix) + ".png")
        fig.savefig(path, dpi=dpi)
        print(f'Fig saved to {path}')
        return path

    def export(self, scans, suffix: str = ""):
        if self.last_export is None:
            raise RuntimeError(
                f"{type(self).__name__}.export() called before display() "
                "populated last_export."
            )
        path = os.path.join(self._output_dir(scans), self._output_stem(scans, suffix) + ".npz")
        np.savez(path, **self.last_export)
        print(f'Data exported to {path}')
        return path


def display_scans(displayer, scans: Sequence, save=False, export=False, suffix='',
                  fig=None, ax=None, **kwargs):
    """
    ``ScanDataAnalyzer.display_scan`` for a ``MultiScanDisplayer``: render
    ``scans`` on one figure, optionally save it and export its arrays.
    Returns ``(fig, ax)``.
    """
    scans = list(scans)
    fig, ax = displayer.display(scans, fig=fig, ax=ax, **kwargs)
    if save:
        displayer.save(fig, scans, suffix=suffix)
    if export:
        displayer.export(scans, suffix=suffix)
    return fig, ax
