from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from PyGEECSPlotter.displayers.shot_selection_grid import ShotSelectionGrid


class SampledImages(ShotSelectionGrid):
    """
    Grid of individual shot images sampled evenly across ``active_data``.

    Picks ``n_samples`` shots equally spaced across ``scan.active_data``
    (the first and last active shots are always included) and renders each
    through the analyzer's pipeline.

    Parameters
    ----------
    analyzer : DiagnosticAnalyzer
        Per-shot analyzer used to load + process each sampled shot.
    bg : optional
        Background spec forwarded to the per-shot pipeline.
    n_samples : int, optional
        Number of shots to display. Defaults to ``len(scan.active_data)``
        (every active shot). Clamped to the number of active shots. When
        rounding to integer row positions would collide (``n_samples``
        close to the shot count), fewer unique panels may result.
    ncols, use_analyzer_display, suppress_labels, display_dict :
        See ``ImageGridDisplayer``.
    """

    def __init__(
        self,
        analyzer,
        bg=None,
        n_samples: Optional[int] = None,
        ncols: int = 4,
        use_analyzer_display: bool = True,
        suppress_labels: bool = True,
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        name = f"{analyzer.output_diagnostic or analyzer.diagnostic}_sampled"
        super().__init__(
            analyzer,
            bg=bg,
            ncols=ncols,
            use_analyzer_display=use_analyzer_display,
            suppress_labels=suppress_labels,
            display_dict=display_dict,
            name=name,
        )
        self.n_samples = n_samples

    def _select_rows(self, scan) -> List[Tuple[str, int]]:
        active = scan.active_data
        n_total = len(active)
        n = self.n_samples if self.n_samples is not None else n_total
        n = max(1, min(int(n), n_total))
        idxs = np.unique(np.linspace(0, n_total - 1, n).round().astype(int))

        shot_col = active['Shotnumber'] if 'Shotnumber' in active.columns else None
        selection: List[Tuple[str, int]] = []
        for pos in idxs:
            shot = int(shot_col.iloc[pos]) if shot_col is not None else int(pos)
            selection.append((f'shot {shot}', int(pos)))
        return selection

    def _suptitle(self, scan) -> str:
        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        return scan.scan_data_title(f'{diag} sampled shots')
