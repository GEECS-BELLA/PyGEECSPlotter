from typing import Optional, Dict, Any, Union, Callable

import numpy as np
import matplotlib.pyplot as plt

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer


class MultiDiagnosticAlignment(ScanDisplayer):
    """
    One shot of every alignment camera side by side, each with its target
    crosshair: the scan-level view of an ``AlignmentReader``.

    The chosen shot is read through the reader's per-shot pipeline
    (``scan._iter_shots``), so the panels and targets are the same as
    ``analyze_scan`` gives for that shot.

    Parameters
    ----------
    analyzer : AlignmentReader
        Declares the cameras, their targets and their panel styles.
    shot_selector : 'first' | 'last' | int | callable, optional
        Which active shot to show.
          * ``'first'`` (default) / ``'last'`` — the first / last active shot
            with the most cameras present
          * ``int`` — the row with that ``Shotnumber``
          * callable(active_df) -> int row position
    bg : optional
        Background spec forwarded to the per-shot pipeline (a dict
        ``{diagnostic: bg}``; see ``MultiDiagnosticAnalyzer``).
    ncols : int, optional
        Grid columns. Defaults to the analyzer's (near-square).
    display_dict : dict, optional
        Whole-figure overrides: ``'panel_aspect'`` (plot-area width /
        height, default 1; images drawn with a fixed aspect keep it),
        ``'panel_width'`` (inches, default 2.5), ``'max_figsize'`` (default
        ``MAX_FIG_SIZE``; panels shrink to fit) and ``'figsize'`` (used as
        given, ignoring the cap). Also forwarded to the analyzer's
        ``display_data`` (``'axes_on'``, ``'target_color'``, ...).

    Notes
    -----
    Creates its own figure; ``fig`` / ``ax`` arguments are ignored.
    ``last_export`` holds each camera's image (``'<diag> image'``) and
    target (``'<diag> target'``: x0, y0, x_size, y_size), and the shot
    number.
    """

    def __init__(
        self,
        analyzer,
        shot_selector: Union[str, int, Callable] = 'first',
        bg=None,
        ncols: Optional[int] = None,
        display_dict: Optional[Dict[str, Any]] = None,
        output_subdir: Optional[str] = None,
        timestamp_files: bool = True,
    ):
        super().__init__(name=analyzer.output_diagnostic, display_dict=display_dict,
                         output_subdir=output_subdir, timestamp_files=timestamp_files)
        self.display_dict.setdefault('panel_aspect', 1)
        self.analyzer = analyzer
        self.shot_selector = shot_selector
        self.bg = bg
        self.ncols = ncols

    def _select_position(self, scan):
        """Row position in ``scan.active_data`` of the shot to show, or None."""
        active = scan.active_data
        if len(active) == 0:
            return None
        sel = self.shot_selector
        if isinstance(sel, (int, np.integer)):
            matches = np.flatnonzero(active['Shotnumber'].to_numpy() == int(sel))
            return int(matches[0]) if len(matches) else None
        if callable(sel):
            return int(sel(active))

        exists = [f'{name} file_exists' for name, _ in self.analyzer.inputs
                  if f'{name} file_exists' in active.columns]
        n_present = (active[exists] != 0).sum(axis=1).to_numpy() if exists \
            else np.zeros(len(active))
        best = np.flatnonzero(n_present == n_present.max())
        return int(best[-1] if sel == 'last' else best[0])

    def display(self, scan, fig=None, ax=None):
        if any(f'{name} file_list' not in scan.data.columns for name, _ in self.analyzer.inputs):
            self.analyzer.register_with_scan(scan)

        pos = self._select_position(scan)
        images, aux, shot = {}, {}, None
        if pos is not None:
            rows = scan.active_data.iloc[[pos]]
            context, data, _, aux = next(
                scan._iter_shots(self.analyzer, bg=self.bg, show_progress=False, rows=rows))
            images = data or {}
            shot = int(context['Shotnumber'])

        panel_dict = {**self.analyzer.display_dict, **self.display_dict}
        if self.ncols is not None:
            panel_dict['ncols'] = self.ncols
        nrows, ncols = self.analyzer.grid_shape(panel_dict)
        fig, axes = plt.subplots(nrows, ncols, figsize=self._grid_figsize(ncols, nrows),
                                 constrained_layout=True, squeeze=False)
        self.analyzer.display_data(images, display_dict=panel_dict, fig=fig, ax=axes, aux=aux)
        self._apply_panel_aspect(axes)

        name = self.analyzer.alignment_name or self.analyzer.output_diagnostic
        label = f'{name} alignment' + (f', shot {shot}' if shot is not None else ', no shots')
        fig.suptitle(scan.scan_data_title(label))

        targets = (aux or {}).get('targets', {})
        self.last_export = {'shot': np.array(np.nan if shot is None else shot)}
        for diag, image in images.items():
            self.last_export[f'{diag} image'] = image
        for diag, target in targets.items():
            self.last_export[f'{diag} target'] = np.asarray(target, dtype=float)
        return fig, axes
