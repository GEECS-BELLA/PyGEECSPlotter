"""PyGEECSPlotter: scan analysis and plotting for GEECS / BELLA data.

Layout
------
scan_data_analyzers   ScanDataAnalyzer (the sfile hub) and its variants
analyzers             per-shot diagnostic analyzers and readers
displayers            scan-level figures (scalars, lineouts, images, hybrid, magspec)
utils                 analysis maths, fitting, navigation, I/O, plotting helpers
"""

__version__ = "0.1.0"

from PyGEECSPlotter import _legacy

_legacy.install()
