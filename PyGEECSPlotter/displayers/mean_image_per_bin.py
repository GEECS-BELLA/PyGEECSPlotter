from typing import Optional, Dict, Any, Iterable, List, Tuple

import numpy as np

from PyGEECSPlotter.displayers.image_grid import ImageGridDisplayer
from PyGEECSPlotter.displayers._trace_binning import bin_labels


class MeanImagePerBin(ImageGridDisplayer):
    """
    Grid of per-bin mean images, computed via the diagnostic analyzer's pipeline.

    Replaces the old ``ScanDataOverview.analyze_scan`` workflow:
    ``scan.display_scan(MeanImagePerBin(analyzer, ...))``.

    Unlike the shot-selecting grids, each panel is a pixel-wise mean over
    all shots in that bin (via ``scan.aggregate_per_bin``).

    Parameters
    ----------
    analyzer : DiagnosticAnalyzer
        Per-shot analyzer used to load + process each shot.
    bg : optional
        Background spec forwarded to ``aggregate_per_bin``.
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
        name = f"{analyzer.output_diagnostic or analyzer.diagnostic}_mean_per_bin"
        super().__init__(
            analyzer,
            ncols=ncols,
            use_analyzer_display=use_analyzer_display,
            suppress_labels=suppress_labels,
            display_dict=display_dict,
            name=name,
            output_subdir=output_subdir,
            timestamp_files=timestamp_files,
        )
        self.bg = bg
        self.bins = bins
        self.label_column = label_column
        self.label_fmt = label_fmt

    def _collect_panels(self, scan) -> List[Tuple[str, Any, Optional[Dict[str, Any]]]]:
        bins, mean_per_bin, _ = scan.aggregate_per_bin(
            self.analyzer, bg=self.bg, bins=self.bins
        )
        if mean_per_bin is None:
            raise RuntimeError("No bins produced data.")

        labels = bin_labels(scan, bins, label_column=self.label_column or False,
                            label_fmt=self.label_fmt)
        panels: List[Tuple[str, Any, Optional[Dict[str, Any]]]] = []
        for label, data in zip(labels, mean_per_bin):
            if data is None or np.all(np.isnan(data)):
                data = None
            panels.append((label, data, None))
        return panels

    def _suptitle(self, scan) -> str:
        title = scan.scan_data_title(f'{self.analyzer.diagnostic} mean per bin')
        if self.label_column:
            title += f'\n{self.label_column}'
        return title
