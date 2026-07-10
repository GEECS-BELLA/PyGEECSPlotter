from typing import Optional, Dict, Any, List, Tuple

from PyGEECSPlotter.displayers.image_grid import ImageGridDisplayer


class ShotSelectionGrid(ImageGridDisplayer):
    """
    Base for grids that pick one representative shot per panel and display
    that shot's actual image (as opposed to an aggregate like a mean).

    Subclasses implement ``_select_rows(scan)`` returning, in panel order,
    a list of ``(label, positional_index)`` where ``positional_index`` is a
    row position into ``scan.active_data``. This base then loads + analyzes
    only those rows (via ``scan._iter_shots(rows=...)``) and renders each.

    Only the selected rows are processed, so this stays cheap even on
    thousand-shot scans. A selected shot whose file is missing (analyzed
    ``data`` is ``None``) leaves a blank panel in place, preserving layout.
    """

    def __init__(
        self,
        analyzer,
        bg=None,
        ncols: int = 4,
        use_analyzer_display: bool = True,
        suppress_labels: bool = True,
        display_dict: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            analyzer,
            ncols=ncols,
            use_analyzer_display=use_analyzer_display,
            suppress_labels=suppress_labels,
            display_dict=display_dict,
            name=name,
        )
        self.bg = bg

    # ------------------------------------------------------------------
    # Subclasses override this.
    # ------------------------------------------------------------------
    def _select_rows(self, scan) -> List[Tuple[str, int]]:
        """
        Return ``[(label, positional_index), ...]`` in panel order, where
        ``positional_index`` indexes into ``scan.active_data`` (0-based).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _select_rows(scan)."
        )

    def _collect_panels(self, scan) -> List[Tuple[str, Any, Optional[Dict[str, Any]]]]:
        active = scan.active_data
        if len(active) == 0:
            raise RuntimeError("No active shots to display.")

        selection = self._select_rows(scan)
        if not selection:
            raise RuntimeError(f"{type(self).__name__} selected no rows.")

        labels = [lab for lab, _ in selection]
        positions = [pos for _, pos in selection]
        subset = active.iloc[positions]

        # Iterate the selected rows once. _iter_shots preserves subset order,
        # so the k-th yielded result corresponds to the k-th selected panel.
        panels: List[Tuple[str, Any, Optional[Dict[str, Any]]]] = []
        for label, (_, data, results, _) in zip(
            labels,
            scan._iter_shots(self.analyzer, bg=self.bg, show_progress=False, rows=subset),
        ):
            panels.append((label, data, results))
        return panels
