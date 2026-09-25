"""
Scan-level displayers.

Each concrete displayer lives in its own module. Import the base or the
concrete class either directly from its module or from this package:

    from PyGEECSPlotter.displayers import (
        ScanDisplayer,
        ScalarVsParameter,
        CorrelationHeatmap,
        ImageGridDisplayer,
        MeanImagePerBin,
        SampledImages,
        RepresentativeImagePerBin,
        MultiDiagnosticAlignment,
        TraceWaterfall,
        TraceMeanPerBin,
        TraceMeanWaterfall,
        LineoutWaterfall,
        LineoutMeanPerBin,
        LineoutMeanWaterfall,
    )

Image-grid family
-----------------
``ImageGridDisplayer`` is the shared base for grids of per-shot images
(layout, render loop, ``suppress_labels``). ``MeanImagePerBin`` averages
each bin; ``ShotSelectionGrid`` subclasses pick one real shot per panel —
``SampledImages`` (evenly spaced across the scan) and
``RepresentativeImagePerBin`` (first / last / max / min per bin).

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
axis's per-bin mean lineout as a waterfall.
"""

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer
from PyGEECSPlotter.displayers.scalar_vs_parameter import ScalarVsParameter
from PyGEECSPlotter.displayers.correlation_heatmap import CorrelationHeatmap
from PyGEECSPlotter.displayers.image_grid import ImageGridDisplayer
from PyGEECSPlotter.displayers.shot_selection_grid import ShotSelectionGrid
from PyGEECSPlotter.displayers.mean_image_per_bin import MeanImagePerBin
from PyGEECSPlotter.displayers.sampled_images import SampledImages
from PyGEECSPlotter.displayers.representative_image_per_bin import RepresentativeImagePerBin
from PyGEECSPlotter.displayers.multi_diagnostic_alignment import MultiDiagnosticAlignment
from PyGEECSPlotter.displayers.trace_waterfall import TraceWaterfall
from PyGEECSPlotter.displayers.trace_mean_per_bin import TraceMeanPerBin
from PyGEECSPlotter.displayers.trace_mean_waterfall import TraceMeanWaterfall
from PyGEECSPlotter.displayers.lineout_waterfall import LineoutWaterfall
from PyGEECSPlotter.displayers.lineout_mean_per_bin import LineoutMeanPerBin
from PyGEECSPlotter.displayers.lineout_mean_waterfall import LineoutMeanWaterfall

__all__ = [
    "ScanDisplayer",
    "ScalarVsParameter",
    "CorrelationHeatmap",
    "ImageGridDisplayer",
    "ShotSelectionGrid",
    "MeanImagePerBin",
    "SampledImages",
    "RepresentativeImagePerBin",
    "MultiDiagnosticAlignment",
    "TraceWaterfall",
    "TraceMeanPerBin",
    "TraceMeanWaterfall",
    "LineoutWaterfall",
    "LineoutMeanPerBin",
    "LineoutMeanWaterfall",
]
