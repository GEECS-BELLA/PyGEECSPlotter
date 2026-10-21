"""
Scan-level displayers.

Every public displayer is importable from this package:

    from PyGEECSPlotter.displayers import ScalarVsParameter, MeanImagePerBin, ...

Modules are grouped by what they show:

    scan_displayer.py        ScanDisplayer base (display / save / export)
    multi_scan_displayer.py  MultiScanDisplayer base + display_scans()
    _lineout_binning.py, _trace_binning.py, _shot_selection.py   shared helpers
    scalars/    sfile columns: ScalarVsParameter, CorrelationHeatmap, ScanStatsTable
    lineouts/   1-D per-shot output: Lineout*, Trace*, OpticalSpectrum*
    images/     2-D per-shot output: ImageGridDisplayer, MeanImagePerBin,
                ShotSelectionGrid, SampledImages, RepresentativeImagePerBin,
                PeakWidthImages
    hybrid/     several diagnostics or several scans in one view: ShotCompare,
                MultiDiagnosticAlignment, FocalScanDisplayer, MultiScan*
    magspec/    MagSpecAllE* (physical-units subclasses of the lineout and
                image families)

Scalars, fits and statistics
----------------------------
``ScalarVsParameter`` plots one scalar against the scan parameter (or any
column), with an optional fit from ``PyGEECSPlotter.utils.fitting``: preset
models (gaussian, super gaussian, lorentzian (+ slope), boltzmann, extreme,
with / without offset), a polynomial, linear regression with a confidence
band and R², or any callable; on the shots or on the per-bin summary.
``CorrelationHeatmap`` shows pairwise correlations; ``ScanStatsTable``
draws ``describe()`` of the active shots as a table.

Single-shot compare
-------------------
``ShotCompare`` shows several diagnostics' ``display_data`` for the same
shot side by side, one figure per selected shot, read through a
``ShotCompareReader`` (``PyGEECSPlotter.analyzers.shot_compare_analysis``) built from
one analyzer per diagnostic. Same pattern as ``MultiDiagnosticAlignment``.
``extra_panels`` adds ``ScalarVsParameter`` plots (with the shot
highlighted) and a table of the shot's scalars next to the scan's.

Multi-scan
----------
``MultiScanDisplayer`` subclasses take a list of scans (each keeping its own
filters) and are run with ``display_scans(displayer, scans, save=True)``.
``MultiScanScalarVsParameter`` overlays one scalar vs parameter per scan
(legend: scan number, or a scan's mean / max of another column);
``MultiScanLineoutMean`` overlays each scan's mean lineouts, per bin or over
the whole scan.

Image-grid family
-----------------
``ImageGridDisplayer`` is the shared base for grids of per-shot images
(layout, render loop, ``suppress_labels``). ``MeanImagePerBin`` averages
each bin; ``ShotSelectionGrid`` subclasses pick one real shot per panel —
``SampledImages`` (evenly spaced across the scan) and
``RepresentativeImagePerBin`` (first / last / max / min per bin).
``PeakWidthImages`` is a three-panel ``RepresentativeImagePerBin``: the
peak bin of a column's per-bin mean, and the bins one half width either
side (``find_peak_bins``).

Focal scan
----------
``FocalScanDisplayer`` is a custom composite for laser focal scans:
``PeakWidthImages`` of the max counts on top, and max counts with the 2nd
moments (x/y, x45/y45) vs stage position below.

Trace family
------------
The 1-D counterparts, for analyzers whose per-shot output is a dict of
trace DataFrames rather than an image (e.g. ``FrogAnalyzer``).
``TraceWaterfall`` stacks every shot's trace into one image to show drift
across a scan; ``TraceMeanPerBin`` is the 1-D analogue of
``MeanImagePerBin``. ``TraceMeanWaterfall`` combines the two — every bin's
mean trace stacked as a waterfall image (intensity only, no phase). All three
need the shots on a common axis, so give the analyzer a resampling grid.

Lineout family
--------------
The same three views, but for analyzers whose per-shot output is a flat
``aux`` dict of coordinate/lineout pairs (e.g. ``ImageAnalyzer`` with
``generate_lineouts=True``, returning ``{'x': x, 'y': y, 'x_lo': x_lo,
'y_lo': y_lo}``) rather than a dict of trace DataFrames. ``LineoutWaterfall``
stacks one axis's lineout across every shot; ``LineoutMeanPerBin`` overlays
several axes' mean lineouts per bin panel; ``LineoutMeanWaterfall`` stacks one
axis's per-bin mean lineout as a waterfall. ``MagSpecAllEWaterfall`` is a
``LineoutWaterfall`` of ``MagSpecAllEAnalyzer``'s spectrum on its fixed
momentum grid (``'p'`` / ``'p_lo'``), with physical labels and a log scale.

Magspec (allE) family
---------------------
Thin, physical-units subclasses of the families above for
``MagSpecAllEAnalyzer`` / ``MagSpecAllEReader``: ``MagSpecAllEWaterfall``
(every shot's spectrum), ``MagSpecAllESpectrumPerBin`` and
``MagSpecAllESpectrumMeanWaterfall`` (mean spectrum per bin, as panels or a
waterfall), and ``MagSpecAllESampledShots`` /
``MagSpecAllERepresentativePerBin`` (allE images in GeV/c x mrad on one
colour scale). Run them on ``MagSpecAllEReader`` to avoid re-analysing.

Optical spectrum family
-----------------------
Lineout-family subclasses on the ``'wl'`` / ``'wl_lo'`` aux of
``OpticalSpectrumAnalyzer`` and ``CombinedVisNIRSpectrum`` (which also
returns ``'vis_wl'`` / ``'nir_wl'``): ``OpticalSpectrumWaterfall`` (every
shot), ``OpticalSpectrumMeanPerBin`` (mean spectrum per bin panel, optionally
overlaying VIS / NIR / combined) and ``OpticalSpectrumMeanWaterfall``.
"""

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer
from PyGEECSPlotter.displayers.scalars.scalar_vs_parameter import ScalarVsParameter
from PyGEECSPlotter.displayers.scalars.correlation_heatmap import CorrelationHeatmap
from PyGEECSPlotter.displayers.scalars.scan_stats_table import ScanStatsTable
from PyGEECSPlotter.displayers.images.image_grid import ImageGridDisplayer
from PyGEECSPlotter.displayers.images.shot_selection_grid import ShotSelectionGrid
from PyGEECSPlotter.displayers.images.mean_image_per_bin import MeanImagePerBin
from PyGEECSPlotter.displayers.images.sampled_images import SampledImages
from PyGEECSPlotter.displayers.images.representative_image_per_bin import RepresentativeImagePerBin
from PyGEECSPlotter.displayers.images.peak_width_images import PeakWidthImages, find_peak_bins
from PyGEECSPlotter.displayers.hybrid.focal_scan import FocalScanDisplayer
from PyGEECSPlotter.displayers.hybrid.multi_diagnostic_alignment import MultiDiagnosticAlignment
from PyGEECSPlotter.displayers.hybrid.shot_compare import ShotCompare
from PyGEECSPlotter.displayers.lineouts.trace_waterfall import TraceWaterfall
from PyGEECSPlotter.displayers.lineouts.trace_mean_per_bin import TraceMeanPerBin
from PyGEECSPlotter.displayers.lineouts.trace_mean_waterfall import TraceMeanWaterfall
from PyGEECSPlotter.displayers.lineouts.lineout_waterfall import LineoutWaterfall
from PyGEECSPlotter.displayers.lineouts.lineout_mean_per_bin import LineoutMeanPerBin
from PyGEECSPlotter.displayers.lineouts.lineout_mean_waterfall import LineoutMeanWaterfall
from PyGEECSPlotter.displayers.magspec.magspec_alle_waterfall import MagSpecAllEWaterfall
from PyGEECSPlotter.displayers.magspec.magspec_alle_gallery import (
    MagSpecAllESampledShots,
    MagSpecAllERepresentativePerBin,
)
from PyGEECSPlotter.displayers.magspec.magspec_alle_spectrum_per_bin import (
    MagSpecAllESpectrumPerBin,
    MagSpecAllESpectrumMeanWaterfall,
)
from PyGEECSPlotter.displayers.lineouts.optical_spectrum import (
    OpticalSpectrumWaterfall,
    OpticalSpectrumMeanPerBin,
    OpticalSpectrumMeanWaterfall,
)
from PyGEECSPlotter.displayers.multi_scan_displayer import MultiScanDisplayer, display_scans
from PyGEECSPlotter.displayers.hybrid.multi_scan_scalar_vs_parameter import MultiScanScalarVsParameter
from PyGEECSPlotter.displayers.hybrid.multi_scan_lineout_mean import MultiScanLineoutMean

__all__ = [
    "ScanDisplayer",
    "ScalarVsParameter",
    "CorrelationHeatmap",
    "ScanStatsTable",
    "ImageGridDisplayer",
    "ShotSelectionGrid",
    "MeanImagePerBin",
    "SampledImages",
    "RepresentativeImagePerBin",
    "PeakWidthImages",
    "find_peak_bins",
    "FocalScanDisplayer",
    "MultiDiagnosticAlignment",
    "ShotCompare",
    "TraceWaterfall",
    "TraceMeanPerBin",
    "TraceMeanWaterfall",
    "LineoutWaterfall",
    "LineoutMeanPerBin",
    "LineoutMeanWaterfall",
    "MagSpecAllEWaterfall",
    "MagSpecAllESampledShots",
    "MagSpecAllERepresentativePerBin",
    "MagSpecAllESpectrumPerBin",
    "MagSpecAllESpectrumMeanWaterfall",
    "OpticalSpectrumWaterfall",
    "OpticalSpectrumMeanPerBin",
    "OpticalSpectrumMeanWaterfall",
    "MultiScanDisplayer",
    "MultiScanScalarVsParameter",
    "MultiScanLineoutMean",
    "display_scans",
]
