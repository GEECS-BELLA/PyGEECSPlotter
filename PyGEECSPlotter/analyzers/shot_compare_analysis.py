# Single-shot comparison across several diagnostics.
# MultiDiagnosticAnalyzer that runs each diagnostic's own analyzer on its
# file for the same shot, and draws every diagnostic's display_data side by
# side. The per-shot counterpart of the alignment overview, but for any
# analyzers (images, spectra, FROG traces, ...), not only raw camera frames.

import inspect
import os
from typing import Optional, Dict, Any, List, Union

import numpy as np
import matplotlib.pyplot as plt

from PyGEECSPlotter.analyzers.diagnostic_analyzer import DiagnosticAnalyzer
from PyGEECSPlotter.analyzers.multi_diagnostic_analyzer import MultiDiagnosticAnalyzer


class ShotCompareReader(MultiDiagnosticAnalyzer):
    """
    Every listed diagnostic's analysis of the same shot.

    Per shot, each sub-analyzer runs its own ``load_data`` /
    ``analyze_data`` on its diagnostic's file (with its own background,
    from a ``{diagnostic: bg}`` dict). ``data`` is ``{diag: analysed data}``
    (``None`` where the file is missing or the analysis gave nothing),
    ``results`` are every sub-analyzer's scalars prefixed with its
    diagnostic (``'<diag> <scalar>'``), and ``aux`` is
    ``{diag: (return_dict, sub_aux)}`` — what each sub-analyzer's
    ``display_data`` needs to redraw that shot.

    Missing diagnostics leave an empty panel rather than masking the shot:
    ``register_with_scan`` defaults to ``remove_missing_files=False``.

    Parameters
    ----------
    sub_analyzers : list or dict of DiagnosticAnalyzer
        One analyzer per diagnostic, in panel order. Each draws its own
        panel with its own ``display_dict``.
    display_dict : dict, optional
        Whole-figure options: ``'nrows'`` / ``'ncols'`` (default: near-square
        grid), ``'titles'`` (``{diag: panel title}``; default the diagnostic).
    output_diagnostic : str, optional
        Prefix for merged sfile columns (default ``'shot-compare'``).
    """

    def __init__(
        self,
        sub_analyzers: Union[List[DiagnosticAnalyzer], Dict[str, DiagnosticAnalyzer]],
        display_dict: Optional[Dict[str, Any]] = None,
        output_diagnostic: Optional[str] = 'shot-compare',
    ):
        super().__init__(sub_analyzers=sub_analyzers, output_diagnostic=output_diagnostic,
                         display_dict=display_dict)

    # ------------------------------------------------------------------
    # Scan-level integration
    # ------------------------------------------------------------------
    def register_with_scan(self, scan, remove_missing_files=False):
        super().register_with_scan(scan, remove_missing_files=remove_missing_files)

    # ------------------------------------------------------------------
    # Pipeline contract
    # ------------------------------------------------------------------
    def load_data(self, paths):
        """``{diag: data}``; ``None`` for a missing path or file (a shot
        without that diagnostic has a NaN ``file_list`` entry)."""
        out = {}
        for name, path in paths.items():
            sub = self.sub_analyzers.get(name)
            if sub is None or not isinstance(path, str) or not os.path.exists(path):
                out[name] = None
            else:
                out[name] = sub.load_data(path)
        return out

    def analyze_data(self, data, bg=None, context=None):
        data = data or {}
        out_data, results, aux = {}, {}, {}
        for name, sub in self.sub_analyzers.items():
            sub_data = data.get(name)
            if sub_data is None:
                out_data[name] = None
                continue
            sub_out, sub_results, sub_aux = sub.analyze_data(
                sub_data, bg=self._bg_for(bg, name), context=context)
            out_data[name] = sub_out
            sub_results = sub_results or {}
            aux[name] = (sub_results, sub_aux or {})
            prefix = sub.output_diagnostic or sub.diagnostic or name
            for key, value in sub_results.items():
                if np.isscalar(value):
                    results[f'{prefix} {key}'] = value
        return out_data, results, aux

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def grid_shape(self, display_dict=None, n=None):
        """(nrows, ncols) for ``n`` panels (default one per diagnostic)."""
        display_dict = self.display_dict if display_dict is None else display_dict
        n = len(self.sub_analyzers) if n is None else n
        nrows, ncols = display_dict.get('nrows'), display_dict.get('ncols')
        if ncols is None and nrows is not None:
            ncols = int(np.ceil(n / nrows))
        ncols = ncols or int(np.ceil(np.sqrt(n)))
        return int(np.ceil(n / ncols)), int(ncols)

    def display_data(self, data, display_dict=None, return_dict=None, title=None,
                     fig=None, ax=None, aux=None):
        """One panel per diagnostic, each drawn by its own analyzer's
        ``display_data``. ``ax`` may be an array of at least one axis per
        diagnostic; else a new grid is made. A diagnostic with no data, or
        whose ``display_data`` fails, gets an empty, labelled panel."""
        display_dict = self.display_dict if display_dict is None else display_dict
        data = data or {}
        aux = aux or {}
        titles = display_dict.get('titles', {})

        if fig is None or ax is None:
            nrows, ncols = self.grid_shape(display_dict)
            fig, ax = plt.subplots(nrows, ncols, squeeze=False, constrained_layout=True,
                                   figsize=display_dict.get('figsize', (3 * ncols, 3 * nrows)))
        axes = np.atleast_1d(ax).ravel()

        for a, (name, sub) in zip(axes, self.sub_analyzers.items()):
            panel_title = titles.get(name, name)
            sub_data = data.get(name)
            if sub_data is None:
                self._empty_panel(a, panel_title, 'no data')
                continue
            sub_return, sub_aux = aux.get(name, ({}, {}))
            try:
                self._draw_sub(sub, sub_data, sub_return, sub_aux, panel_title, fig, a)
            except Exception as e:
                a.cla()
                self._empty_panel(a, panel_title, f'display failed:\n{type(e).__name__}')
        for a in axes[len(self.sub_analyzers):]:
            a.set_visible(False)

        if title is not None:
            fig.suptitle(title)
        return fig, ax

    @staticmethod
    def _draw_sub(sub, data, return_dict, sub_aux, title, fig, ax):
        """``sub.display_data`` into ``ax``, passing ``aux`` only to analyzers
        whose ``display_data`` accepts it (as ``analyze_scan`` does)."""
        kwargs = dict(return_dict=return_dict, title=title, fig=fig, ax=ax)
        if 'aux' in inspect.signature(sub.display_data).parameters:
            kwargs['aux'] = sub_aux
        sub.display_data(data, **kwargs)
        if not ax.get_title():
            ax.set_title(title, fontsize='small')

    @staticmethod
    def _empty_panel(ax, title, text):
        ax.text(0.5, 0.5, text, ha='center', va='center', transform=ax.transAxes, color='0.5')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize='small')
