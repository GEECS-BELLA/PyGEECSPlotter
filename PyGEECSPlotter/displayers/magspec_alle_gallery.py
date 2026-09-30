from typing import Optional, Dict, Any, List, Tuple, Iterable

import numpy as np

from PyGEECSPlotter.displayers.sampled_images import SampledImages
from PyGEECSPlotter.displayers.representative_image_per_bin import RepresentativeImagePerBin


class _AllEPanels:
    """
    Physical-units rendering for ``ShotSelectionGrid`` subclasses showing
    allE images: each panel is charge density [pC/mrad/(GeV/c)] on momentum
    [GeV/c] x angle [mrad] axes, all panels on one colour scale with a
    single shared colour bar, tick labels on the outer panels only.

    The momentum axis comes from the shot's ``aux['momentum']`` when the
    analyzer provides it (``MagSpecAllEAnalyzer``, or ``MagSpecAllEReader``
    with ``load='all'``), else ``linspace(roi)`` from the analyzer's
    ``analyzer_dict``. The angle axis is ``aux['angle']`` or the fixed
    256-point [-1.3, 1.3] mrad axis.

    display_dict keys: ``figsize``, ``cmap`` (default ``'jet'``), ``vmax``
    (default: 99.9th percentile over all panels), ``xlims``, ``ylims``.
    """

    def _collect_panels(self, scan) -> List[Tuple[str, Any, Optional[Dict[str, Any]]]]:
        # ShotSelectionGrid's selection, but keeping each shot's axes from aux
        active = scan.active_data
        if len(active) == 0:
            raise RuntimeError("No active shots to display.")
        selection = self._select_rows(scan)
        if not selection:
            raise RuntimeError(f"{type(self).__name__} selected no rows.")
        subset = active.iloc[[pos for _, pos in selection]]

        panels = []
        for (label, _), (_, data, _, aux) in zip(
            selection,
            scan._iter_shots(self.analyzer, bg=self.bg, show_progress=False, rows=subset),
        ):
            axes = None
            if data is not None:
                axes = self._panel_axes(np.asarray(data), aux or {})
            panels.append((label, data, axes))

        dens = [self._density(np.asarray(d), ax) for _, d, ax in panels if d is not None]
        vmax = self.display_dict.get('vmax')
        if vmax is None and dens:
            vmax = max(np.nanpercentile(d, 99.9) for d in dens) or None
        self._vmax = vmax
        return panels

    def _panel_axes(self, data, aux):
        n_ang, n_mmt = data.shape
        mmt = aux.get('momentum')
        if mmt is None or len(mmt) != n_mmt:
            roi = self.analyzer.analyzer_dict.get('roi', (0.01, 5.0))
            mmt = np.linspace(roi[0], roi[1], n_mmt)
        ang = aux.get('angle')
        if ang is None or len(ang) != n_ang:
            ang = np.linspace(-1.3, 1.3, n_ang)
        return {'momentum': np.asarray(mmt, float), 'angle': np.asarray(ang, float)}

    @staticmethod
    def _density(data, axes):
        mmt, ang = axes['momentum'], axes['angle']
        return 1e-6 * data / (mmt[1] - mmt[0]) / (ang[1] - ang[0])   # aC -> pC/mrad/(GeV/c)

    def _render_panel(self, fig, a, data, return_dict, label):
        dd = self.display_dict
        mmt, ang = return_dict['momentum'], return_dict['angle']
        self._mappable = a.pcolormesh(
            mmt, ang, self._density(np.asarray(data), return_dict),
            cmap=dd.get('cmap', 'jet'), shading='auto', vmin=0, vmax=self._vmax,
            rasterized=True,
        )
        a.set_title(label, fontsize=9)
        a.set_xlim(dd.get('xlims', (mmt[0], mmt[-1])))
        a.set_ylim(dd.get('ylims', (ang[0], ang[-1])))
        spec = a.get_subplotspec()
        if spec.is_last_row():
            a.set_xlabel('Momentum [GeV/c]')
        else:
            a.set_xticklabels([])
        if spec.is_first_col():
            a.set_ylabel('Angle [mrad]')
        else:
            a.set_yticklabels([])

    def display(self, scan, fig=None, ax=None):
        self._mappable = None
        fig, axes = super().display(scan, fig=fig, ax=ax)
        if self._mappable is not None:
            visible = [a for a in axes.flat if a.get_visible()]
            fig.colorbar(self._mappable, ax=visible, label='pC/mrad/(GeV/c)', shrink=0.9)
        return fig, axes


class MagSpecAllESampledShots(_AllEPanels, SampledImages):
    """
    ``SampledImages`` for allE: ``n_samples`` shots evenly spaced through the
    scan, drawn in physical units on a shared colour scale.

    Works on ``MagSpecAllEAnalyzer`` or, much faster, on
    ``MagSpecAllEReader(load='image')`` / ``load='all'`` (the latter gives
    each panel its exact momentum axis). Only the sampled shots are read.

    Parameters
    ----------
    analyzer : MagSpecAllEAnalyzer or MagSpecAllEReader
    bg : optional
        Background spec forwarded to the per-shot pipeline.
    n_samples : int, optional
        Number of shots to show (default 12).
    ncols : int, optional
        Grid columns (default 4).
    display_dict : dict, optional
        ``figsize``, ``cmap``, ``vmax``, ``xlims``, ``ylims``.
    """

    def __init__(
        self,
        analyzer,
        bg=None,
        n_samples: Optional[int] = 12,
        ncols: int = 4,
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(analyzer, bg=bg, n_samples=n_samples, ncols=ncols,
                         use_analyzer_display=False, suppress_labels=False,
                         display_dict=display_dict)
        diag = analyzer.output_diagnostic or analyzer.diagnostic
        self.name = f'{diag}_sampled_shots'


class MagSpecAllERepresentativePerBin(_AllEPanels, RepresentativeImagePerBin):
    """
    ``RepresentativeImagePerBin`` for allE: one real shot per bin (``mode``
    ``'first'``, ``'last'``, ``'max'`` or ``'min'`` of ``parameter``), drawn
    in physical units on a shared colour scale so bins compare directly.

    With ``MagSpecAllEReader``, ranking uses the scalar columns already in
    the sfile (e.g. ``'MagSpecAllE charge_pC'``, plus the ``analysis_label``
    if one was used) and only the chosen shots' files are read.

    Parameters
    ----------
    analyzer : MagSpecAllEAnalyzer or MagSpecAllEReader
    mode : str
        ``'first'``, ``'last'``, ``'max'`` or ``'min'``.
    parameter : str, optional
        Column to rank on; required for ``'max'`` / ``'min'``.
    bg : optional
        Background spec forwarded to the per-shot pipeline.
    bins : iterable of int, optional
        Bins to show; default all active bins.
    ncols : int, optional
        Grid columns (default 4).
    display_dict : dict, optional
        ``figsize``, ``cmap``, ``vmax``, ``xlims``, ``ylims``.
    """

    def __init__(
        self,
        analyzer,
        mode: str = 'max',
        parameter: Optional[str] = None,
        bg=None,
        bins: Optional[Iterable[int]] = None,
        ncols: int = 4,
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(analyzer, mode=mode, parameter=parameter, bg=bg, bins=bins,
                         ncols=ncols, use_analyzer_display=False, suppress_labels=False,
                         display_dict=display_dict)
        diag = analyzer.output_diagnostic or analyzer.diagnostic
        self.name = f'{diag}_representative_{mode}_per_bin'
