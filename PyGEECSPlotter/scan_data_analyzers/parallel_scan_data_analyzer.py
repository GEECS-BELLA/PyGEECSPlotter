# Deprecated: ScanDataAnalyzer.analyze_scan is parallel by default.
# Author: Claude
# Created: 2026-09-30

import warnings

from PyGEECSPlotter.scan_data_analyzers.scan_data_analysis import ScanDataAnalyzer, _aux_kwarg  # noqa: F401


class ParallelScanDataAnalyzer(ScanDataAnalyzer):
    """
    Deprecated alias of ``ScanDataAnalyzer``.

    ``ScanDataAnalyzer.analyze_scan`` now runs shots in a thread pool by
    default (``parallel=True``, ``max_workers=``); pass ``parallel=False``
    for one shot at a time. This class is kept so existing code runs
    unchanged, and warns once.
    """

    _warned = False

    def __init__(self, *args, **kwargs):
        if not ParallelScanDataAnalyzer._warned:
            warnings.warn("ParallelScanDataAnalyzer is deprecated: ScanDataAnalyzer.analyze_scan "
                          "is parallel by default (parallel=False for serial).",
                          DeprecationWarning, stacklevel=2)
            ParallelScanDataAnalyzer._warned = True
        super().__init__(*args, **kwargs)
