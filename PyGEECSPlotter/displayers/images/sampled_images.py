from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from PyGEECSPlotter.displayers.images.shot_selection_grid import ShotSelectionGrid


class SampledImages(ShotSelectionGrid):
    """
    Grid of individual shot images: evenly spaced shots, or the top N by a column.

    By default picks ``n_samples`` shots equally spaced across
    ``scan.active_data`` (the first and last selected shots are always
    included). With ``order_by``, shots are taken in that column's order
    instead:

    - ``top=True``: the ``n_samples`` shots with the largest values
      (``ascending=True``: the smallest), e.g. the brightest shots
    - ``top=False``: ``n_samples`` shots evenly spaced through the sorted
      order, from the smallest to the largest value

    ``per_bin=True`` (with ``order_by`` and ``top=True``) takes the top
    ``n_samples`` of every bin. Each shot is rendered through the analyzer's
    pipeline and labelled with its shot number and ``order_by`` value.

    Parameters
    ----------
    analyzer : DiagnosticAnalyzer
        Per-shot analyzer used to load + process each sampled shot.
    bg : optional
        Background spec forwarded to the per-shot pipeline.
    n_samples : int, optional
        Number of shots to display (per bin with ``per_bin``). Defaults to
        every selected shot. When rounding to integer row positions would
        collide (``n_samples`` close to the shot count), fewer unique
        panels may result.
    order_by : str, optional
        Column to order the shots by.
    top : bool
        With ``order_by``: show the top ``n_samples`` (default False: spread
        through the sorted order).
    ascending : bool
        With ``top``: False (default) picks the largest values, True the smallest.
    per_bin : bool
        With ``order_by`` and ``top``: the top ``n_samples`` of each bin.
    label_fmt : str
        Format of the ``order_by`` value in panel labels.
    ncols, use_analyzer_display, suppress_labels, display_dict :
        See ``ImageGridDisplayer``.
    """

    def __init__(
        self,
        analyzer,
        bg=None,
        n_samples: Optional[int] = None,
        order_by: Optional[str] = None,
        top: bool = False,
        ascending: bool = False,
        per_bin: bool = False,
        label_fmt: str = '{:.4g}',
        ncols: int = 4,
        use_analyzer_display: bool = True,
        suppress_labels: bool = True,
        display_dict: Optional[Dict[str, Any]] = None,
        output_subdir: Optional[str] = None,
        timestamp_files: bool = True,
    ):
        if (top or per_bin) and order_by is None:
            raise ValueError("top / per_bin need order_by (the column to rank by).")
        diag = analyzer.output_diagnostic or analyzer.diagnostic
        if order_by is None:
            name = f"{diag}_sampled"
        else:
            how = ('top' if not ascending else 'bottom') if top else 'sampled'
            name = f"{diag}_{how}{'_per_bin' if per_bin else ''}_by_{order_by}"
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
        self.n_samples = n_samples
        self.order_by = order_by
        self.top = top
        self.ascending = ascending
        self.per_bin = per_bin
        self.label_fmt = label_fmt

    def _ordered_rows(self, scan):
        """The rows to show, in display order."""
        active = scan.active_data
        if self.order_by is None:
            n_total = len(active)
            n = self.n_samples if self.n_samples is not None else n_total
            n = max(1, min(int(n), n_total))
            idxs = np.unique(np.linspace(0, n_total - 1, n).round().astype(int))
            return active.iloc[idxs]
        if self.top:
            n = self.n_samples if self.n_samples is not None else len(active)
            return scan.top(self.order_by, n=n, ascending=self.ascending, per_bin=self.per_bin)
        ranked = scan.sorted_rows(self.order_by, ascending=True)
        ranked = ranked[ranked[self.order_by].notna()]
        n_total = len(ranked)
        if n_total == 0:
            return ranked
        n = self.n_samples if self.n_samples is not None else n_total
        n = max(1, min(int(n), n_total))
        return ranked.iloc[np.unique(np.linspace(0, n_total - 1, n).round().astype(int))]

    def _select_rows(self, scan) -> List[Tuple[str, int]]:
        active = scan.active_data
        rows = self._ordered_rows(scan)
        # ShotSelectionGrid wants positions into active_data
        position = {idx: pos for pos, idx in enumerate(active.index)}
        selection: List[Tuple[str, int]] = []
        for idx, row in rows.iterrows():
            shot = int(row['Shotnumber']) if 'Shotnumber' in row else position[idx]
            label = f'shot {shot}'
            if self.order_by is not None:
                label += f'\n{self.label_fmt.format(row[self.order_by])}'
            selection.append((label, position[idx]))
        return selection

    def _suptitle(self, scan) -> str:
        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        if self.order_by is None:
            return scan.scan_data_title(f'{diag} sampled shots')
        if self.top:
            which = 'smallest' if self.ascending else 'largest'
            per = ' per bin' if self.per_bin else ''
            return scan.scan_data_title(f'{diag}: {which} {self.order_by}{per}')
        return scan.scan_data_title(f'{diag} shots sampled by {self.order_by}')
