from typing import Optional, Dict, Any, Iterable, List, Tuple

import numpy as np

from PyGEECSPlotter.displayers.image_grid import ImageGridDisplayer


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
    ncols, use_analyzer_display, suppress_labels, display_dict :
        See ``ImageGridDisplayer``.
    """

    def __init__(
        self,
        analyzer,
        bg=None,
        bins: Optional[Iterable[int]] = None,
        ncols: int = 4,
        use_analyzer_display: bool = True,
        suppress_labels: bool = True,
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        name = f"{analyzer.output_diagnostic or analyzer.diagnostic}_mean_per_bin"
        super().__init__(
            analyzer,
            ncols=ncols,
            use_analyzer_display=use_analyzer_display,
            suppress_labels=suppress_labels,
            display_dict=display_dict,
            name=name,
        )
        self.bg = bg
        self.bins = bins

    def _collect_panels(self, scan) -> List[Tuple[str, Any, Optional[Dict[str, Any]]]]:
        bins, mean_per_bin, _ = scan.aggregate_per_bin(
            self.analyzer, bg=self.bg, bins=self.bins
        )
        if mean_per_bin is None:
            raise RuntimeError("No bins produced data.")

        panels: List[Tuple[str, Any, Optional[Dict[str, Any]]]] = []
        for b, data in zip(bins, mean_per_bin):
            if data is None or np.all(np.isnan(data)):
                data = None
            panels.append((f'Bin {int(b)}', data, None))
        return panels

    def _suptitle(self, scan) -> str:
        return scan.scan_data_title(f'{self.analyzer.diagnostic} mean per bin')
