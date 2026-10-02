from typing import Optional, Dict, Any, Iterable, List, Tuple

import numpy as np

from PyGEECSPlotter.displayers.shot_selection_grid import ShotSelectionGrid
from PyGEECSPlotter.displayers._trace_binning import bin_labels


_MODES = ('first', 'last', 'max', 'min')


class RepresentativeImagePerBin(ShotSelectionGrid):
    """
    Grid showing one representative shot's image per bin.

    For each bin (``temp Bin number``), pick a single representative shot
    and display its actual analyzed image:

      - ``mode='first'`` — the first shot in the bin (scan order).
      - ``mode='last'``  — the last shot in the bin.
      - ``mode='max'``   — the shot with the largest ``parameter`` value.
      - ``mode='min'``   — the shot with the smallest ``parameter`` value.

    Unlike ``MeanImagePerBin`` (which averages a bin's shots pixel-wise),
    this shows one real shot per bin.

    Parameters
    ----------
    analyzer : DiagnosticAnalyzer
    mode : {'first', 'last', 'max', 'min'}, optional
        Selection rule within each bin. Default ``'first'``.
    parameter : str, optional
        Scalar column in ``active_data`` used by ``mode='max'`` / ``'min'``.
        Required for those modes; ignored for ``'first'`` / ``'last'``.
    bg : optional
        Background spec forwarded to the per-shot pipeline.
    bins : iterable of int, optional
        Bin numbers to render. Defaults to all unique bins in ``active_data``.
    label_column : str, optional
        Scan-data column whose per-bin mean titles each panel (e.g. the scan
        parameter). Default None titles panels ``'Bin {n}'``.
    label_fmt : str, optional
        Format string for the label column's value, e.g. ``'{:.3g} mm'``.
    ncols, use_analyzer_display, suppress_labels, display_dict :
        See ``ImageGridDisplayer``.
    """

    def __init__(
        self,
        analyzer,
        mode: str = 'first',
        parameter: Optional[str] = None,
        bg=None,
        bins: Optional[Iterable[int]] = None,
        label_column: Optional[str] = None,
        label_fmt: str = '{:.4g}',
        ncols: int = 4,
        use_analyzer_display: bool = True,
        suppress_labels: bool = True,
        display_dict: Optional[Dict[str, Any]] = None,
        output_subdir: Optional[str] = None,
        timestamp_files: bool = True,
    ):
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}.")
        if mode in ('max', 'min') and parameter is None:
            raise ValueError(f"mode={mode!r} requires a `parameter` column.")

        name = f"{analyzer.output_diagnostic or analyzer.diagnostic}_{mode}_per_bin"
        super().__init__(
            analyzer,
            bg=bg,
            ncols=ncols,
            use_analyzer_display=use_analyzer_display,
            suppress_labels=suppress_labels,
            display_dict=display_dict,
            name=name,
            output_subdir=output_subdir,
            timestamp_files=timestamp_files,
        )
        self.mode = mode
        self.parameter = parameter
        self.bins = bins
        self.label_column = label_column
        self.label_fmt = label_fmt

    def _select_rows(self, scan) -> List[Tuple[str, int]]:
        active = scan.active_data
        if 'temp Bin number' not in active.columns:
            raise KeyError("'temp Bin number' not in active_data; cannot bin.")
        if self.mode in ('max', 'min') and self.parameter not in active.columns:
            raise KeyError(f"parameter column {self.parameter!r} not in active_data.")

        bin_col = active['temp Bin number']
        bins = list(self.bins) if self.bins is not None else list(np.unique(bin_col))
        labels = bin_labels(scan, bins, label_column=self.label_column or False,
                            label_fmt=self.label_fmt)

        selection: List[Tuple[str, int]] = []
        for b, base in zip(bins, labels):
            in_bin = np.where((bin_col == b).to_numpy())[0]
            if in_bin.size == 0:
                continue
            pos = self._pick_in_bin(active, in_bin)
            selection.append((base, int(pos)))
        return selection

    def _pick_in_bin(self, active, in_bin: np.ndarray) -> int:
        """Positional index (into active_data) of the representative row."""
        if self.mode == 'first':
            return int(in_bin[0])
        if self.mode == 'last':
            return int(in_bin[-1])
        values = active[self.parameter].to_numpy()[in_bin]
        # argmax/argmin ignoring NaNs; fall back to first if all-NaN.
        if np.all(np.isnan(values.astype(float))):
            return int(in_bin[0])
        if self.mode == 'max':
            return int(in_bin[np.nanargmax(values)])
        return int(in_bin[np.nanargmin(values)])

    def _short_param(self) -> str:
        """
        ``self.parameter`` with the diagnostic's own prefix stripped, if
        present — it's usually a column ``analyze_scan`` produced (e.g.
        ``'CAM-PL1-LPMode max_counts'``), and repeating the full diagnostic
        name in the suptitle just doubles it up.
        """
        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        prefix = f'{diag} '
        if self.parameter.startswith(prefix):
            return self.parameter[len(prefix):]
        return self.parameter

    def _suptitle(self, scan) -> str:
        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        detail = self.mode
        if self.mode in ('max', 'min'):
            detail = f'{self.mode} {self._short_param()}'
        title = scan.scan_data_title(f'{diag} {detail} per bin')
        if self.label_column:
            title += f'\n{self.label_column}'
        return title
