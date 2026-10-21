# Laser alignment overview across many cameras.
# MultiDiagnosticAnalyzer that reads one raw frame per alignment camera for
# each shot, plus each camera's ECS target crosshair, and draws all the
# cameras side by side. A reader: it computes nothing.

import os
from typing import Optional, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

from PyGEECSPlotter.analyzers.multi_diagnostic_analyzer import MultiDiagnosticAnalyzer
from PyGEECSPlotter.analyzers.data_reader import DataReader
from PyGEECSPlotter.utils.ecs_live_dumps import read_ecs_live_dump
from PyGEECSPlotter.utils.general import plot_target_ellipse

# ECS Live dump parameters for each crosshair number: centre x, y; size x, y.
_CROSSHAIR_PARAMS = {
    1: ('Target.X', 'Target.Y', 'TargetSize.X', 'TargetSize.Y'),
    2: ('Target2.X', 'Target2.Y', 'Target2Size.X', 'Target2Size.Y'),
}
# diagnostic_dicts keys that configure the reader rather than the panel.
_NON_DISPLAY_KEYS = ('diagnostic', 'file_ext', 'target_on', 'crosshair')


class AlignmentReader(MultiDiagnosticAnalyzer):
    """
    One raw frame per alignment camera per shot, with each camera's target
    crosshair.

    Each entry in ``diagnostic_dicts`` is a dict with:

    - ``'diagnostic'``, ``'file_ext'`` : the camera.
    - ``'target_on'`` (bool, default False) and ``'crosshair'`` (1 or 2,
      default 1): draw that ECS crosshair (``Target.X`` / ``TargetSize.X``
      ..., or ``Target2.X`` ...), read once per scan from
      ``<top_dir>/ECS Live dumps/ScanN.txt``.
    - anything else is that panel's display options: ``'cmap'``, ``'vmin'``
      (default 0), ``'vmax'`` (default the image maximum), ``'norm'``,
      ``'aspect'``.

    Per shot, ``data`` is ``{diag: raw image}`` for each camera present and
    ``aux['targets']`` is ``{diag: (x0, y0, x_size, y_size)}`` (pixels) for
    cameras with a target found. There are no ``results``.

    Cameras missing from a shot (or a whole scan) are left out rather than
    masking the shot: ``register_with_scan`` defaults to
    ``remove_missing_files=False``.

    Parameters
    ----------
    diagnostic_dicts : list[dict]
        Per-camera config; see above.
    alignment_name : str, optional
        Names the default ``output_diagnostic`` (``'<name>-alignment'``) and
        figure titles.
    display_dict : dict, optional
        Whole-figure options: ``'ncols'`` (default: near-square grid),
        ``'axes_on'`` (tick labels, default False), ``'target_color'``,
        ``'target_linewidth'``.
    """

    def __init__(
        self,
        diagnostic_dicts: List[Dict[str, Any]],
        alignment_name: Optional[str] = None,
        display_dict: Optional[Dict[str, Any]] = None,
        output_diagnostic: Optional[str] = None,
    ):
        self.diagnostic_dicts = [dict(d) for d in diagnostic_dicts]
        super().__init__(
            sub_analyzers=[DataReader(diagnostic=d['diagnostic'], file_ext=d['file_ext'])
                           for d in self.diagnostic_dicts],
            output_diagnostic=output_diagnostic or (
                f'{alignment_name}-alignment' if alignment_name else 'alignment'),
            display_dict=display_dict,
        )
        self.alignment_name = alignment_name
        # {scan number: {diag: (x0, y0, x_size, y_size)}}, filled by
        # register_with_scan. Keyed by scan, read-only once filled, so safe to
        # share across threads.
        self.targets: Dict[int, Dict[str, tuple]] = {}

    # ------------------------------------------------------------------
    # Scan-level integration
    # ------------------------------------------------------------------
    def register_with_scan(self, scan, remove_missing_files=False):
        """File lists for every camera, plus the targets of every scan in
        the sfile (from its ECS Live dump)."""
        super().register_with_scan(scan, remove_missing_files=remove_missing_files)
        for scan_num in np.unique(scan.data['scan'].astype(int)):
            path = os.path.join(scan.top_dir, 'ECS Live dumps', f'Scan{scan_num}.txt')
            self.targets[int(scan_num)] = self.read_targets(path)

    def read_targets(self, ecs_dump_path):
        """``{diag: (x0, y0, x_size, y_size)}`` for each camera with
        ``target_on``, from one ECS Live dump; cameras whose target isn't in
        the dump are left out."""
        dump = read_ecs_live_dump(ecs_dump_path)
        if not dump:
            print(f'No ECS Live dump at {ecs_dump_path}: no targets.')
            return {}
        targets = {}
        for d in self.diagnostic_dicts:
            if not d.get('target_on', False):
                continue
            params = dump.get(d['diagnostic'], {})
            values = tuple(params.get(p, np.nan) for p in _CROSSHAIR_PARAMS[d.get('crosshair', 1)])
            if all(isinstance(v, float) and np.isfinite(v) for v in values):
                targets[d['diagnostic']] = values
        return targets

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def analyze_data(self, data, bg=None, context=None):
        images = {diag: image for diag, image in (data or {}).items() if image is not None}
        if not images:
            return None, {}, {}
        scan = (context or {}).get('scan')
        scan_targets = self.targets.get(int(scan), {}) if scan is not None else {}
        targets = {diag: t for diag, t in scan_targets.items() if diag in images}
        return images, {}, {'targets': targets}

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def grid_shape(self, display_dict=None):
        """(nrows, ncols) for one panel per camera."""
        display_dict = self.display_dict if display_dict is None else display_dict
        n = len(self.diagnostic_dicts)
        ncols = display_dict.get('ncols') or int(np.ceil(np.sqrt(n)))
        return int(np.ceil(n / ncols)), ncols

    def display_data(self, data, display_dict=None, return_dict=None, title=None,
                     fig=None, ax=None, aux=None):
        """One panel per camera, in ``diagnostic_dicts`` order, with the
        target ellipse where there is one. ``ax`` may be an array of at
        least one axis per camera; else a new grid is made. Cameras with
        no image get an empty, labelled panel."""
        display_dict = self.display_dict if display_dict is None else display_dict
        images = data or {}
        targets = (aux or {}).get('targets', {})

        if fig is None or ax is None:
            nrows, ncols = self.grid_shape(display_dict)
            fig, ax = plt.subplots(nrows, ncols, squeeze=False, constrained_layout=True,
                                   figsize=display_dict.get('figsize', (2.5 * ncols, 2.5 * nrows)))
        axes = np.atleast_1d(ax).ravel()

        for a, d in zip(axes, self.diagnostic_dicts):
            self.draw_panel(a, d, images.get(d['diagnostic']), targets.get(d['diagnostic']),
                            display_dict)
        for a in axes[len(self.diagnostic_dicts):]:
            a.set_visible(False)

        if title is not None:
            fig.suptitle(title)
        return fig, ax

    @staticmethod
    def draw_panel(ax, panel_dict, image, target, display_dict):
        """One camera: image (``origin='lower'``), target ellipse, title."""
        diag = panel_dict['diagnostic']
        if image is None:
            ax.text(0.5, 0.5, 'no image', ha='center', va='center',
                    transform=ax.transAxes, color='0.5')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            opts = {k: v for k, v in panel_dict.items() if k not in _NON_DISPLAY_KEYS}
            vmax = opts.get('vmax', np.nanmax(image))
            norm = opts.get('norm')
            ax.imshow(image, origin='lower', cmap=opts.get('cmap', 'viridis'),
                      aspect=opts.get('aspect', 'equal'), norm=norm,
                      vmin=None if norm is not None else opts.get('vmin', 0),
                      vmax=None if norm is not None else vmax)
            if target is not None:
                x0, y0, x_size, y_size = target
                plot_target_ellipse(ax, x0=x0, y0=y0, x_size=x_size, y_size=y_size,
                                    line_color=display_dict.get('target_color', 'black'),
                                    linewidth=display_dict.get('target_linewidth', 0.5))
            if not display_dict.get('axes_on', False):
                ax.set_xticklabels([])
                ax.set_yticklabels([])
        ax.set_title(diag, fontsize='small')
