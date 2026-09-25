from typing import Optional, Dict, Any, List, Union, Callable

import numpy as np
import matplotlib.pyplot as plt

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer
from PyGEECSPlotter.image_analysis import ImageAnalyzer


class MultiDiagnosticAlignment(ScanDisplayer):
    """
    Render one frame per diagnostic side-by-side. Designed for alignment /
    beam-quality checks across many cameras in a single scan.

    Each entry in ``diagnostic_dicts`` is a dict with at least:

    - ``'diagnostic'`` : str
    - ``'file_ext'``   : str

    Any additional keys are forwarded as the per-panel ``display_dict``
    (e.g. ``'cmap'``, ``'target_on'``, ``'crosshair'``, ``'axlims'``,
    ``'xtitle'``, ``'ytitle'``, etc.).

    Parameters
    ----------
    diagnostic_dicts : list[dict]
        Per-diagnostic config; see above.
    analyzer : DiagnosticAnalyzer, optional
        Used to load + (lightly) analyze + display each panel.
        Defaults to a shared bare ``ImageAnalyzer()``.
    shot_selector : 'first' | 'last' | int | callable, optional
        Which shot to render per diagnostic.
          * ``'first'`` (default) — the first existing shot
          * ``'last'`` — the last existing shot
          * ``int`` — the row with that ``Shotnumber``
          * callable(present_df) -> int row index
    ncols : int, optional
        Grid columns. Defaults to len(diagnostic_dicts) (single row).
    alignment_name : str, optional
        Used for the figure title and saved filename suffix.
    display_dict : dict, optional
        Whole-figure overrides: ``'figsize'``.

    Notes
    -----
    Creates its own figure; ``fig`` / ``ax`` arguments are ignored.
    Calls ``scan.add_file_list_to_scan_data`` per diagnostic with
    ``remove_missing_files=False`` so missing diagnostics are skipped
    silently rather than masking shots out.
    """

    def __init__(
        self,
        diagnostic_dicts: List[Dict[str, Any]],
        analyzer=None,
        shot_selector: Union[str, int, Callable] = 'first',
        ncols: Optional[int] = None,
        alignment_name: Optional[str] = None,
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        name = f"{alignment_name}_alignment" if alignment_name else "multi_diagnostic_alignment"
        super().__init__(name=name, display_dict=display_dict)
        self.diagnostic_dicts = list(diagnostic_dicts)
        self.analyzer = analyzer if analyzer is not None else ImageAnalyzer()
        self.shot_selector = shot_selector
        self.ncols = ncols
        self.alignment_name = alignment_name

    def _pick_filename(self, scan, diagnostic):
        col_exists = f'{diagnostic} file_exists'
        col_files = f'{diagnostic} file_list'
        if col_exists not in scan.data.columns:
            return None
        present = scan.data[scan.data[col_exists] != 0].reset_index(drop=True)
        if len(present) == 0:
            return None

        sel = self.shot_selector
        if sel == 'first':
            return present[col_files].iloc[0]
        if sel == 'last':
            return present[col_files].iloc[-1]
        if isinstance(sel, (int, np.integer)):
            row = present[present['Shotnumber'] == int(sel)]
            return row[col_files].iloc[0] if len(row) else None
        if callable(sel):
            idx = sel(present)
            return present[col_files].iloc[idx]
        return present[col_files].iloc[0]

    def display(self, scan, fig=None, ax=None):
        n = len(self.diagnostic_dicts)
        ncols = self.ncols if self.ncols is not None else n
        nrows = int(np.ceil(n / ncols))

        figsize = self.display_dict.get('figsize', (3 * ncols, 3 * nrows))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize,
            constrained_layout=True,
            squeeze=False,
        )

        for k, d in enumerate(self.diagnostic_dicts):
            a = axes.flat[k]
            diagnostic = d['diagnostic']
            file_ext = d['file_ext']

            scan.add_file_list_to_scan_data(diagnostic, file_ext, remove_missing_files=False)
            fname = self._pick_filename(scan, diagnostic)

            if fname is None:
                a.set_visible(False)
                continue

            data = self.analyzer.load_data(fname)
            if data is None:
                a.set_visible(False)
                continue
            data, _, _ = self.analyzer.analyze_data(data, analyzer_dict={})

            panel_dict = {**d, 'cbar_off': True}
            self.analyzer.display_data(data, display_dict=panel_dict, fig=fig, ax=a)

        for k in range(n, nrows * ncols):
            axes.flat[k].set_visible(False)

        title = scan.scan_data_title(self.alignment_name or 'alignment')
        fig.suptitle(title)
        return fig, axes
