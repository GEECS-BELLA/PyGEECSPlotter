"""Scan-level analysis: load an sfile, filter / bin it, run analyzers over its shots.

``ScanDataAnalyzer`` is the hub; its ``analyze_scan`` runs shots in threads by
default (``parallel=False`` for serial; ``ParallelScanDataAnalyzer`` is a
deprecated alias). ``DataDirAnalyzer`` treats a plain folder of files as a
scan (no sfile). During a run, ``ScanWatcher`` (``live_analysis``) runs
analyzers and displayers on each scan as it finishes, and ``LiveShotWatcher``
(``live_shot_analysis``) analyses one analyzer's shots while the scan runs.
"""

from PyGEECSPlotter.scan_data_analyzers.scan_data_analysis import ScanDataAnalyzer
from PyGEECSPlotter.scan_data_analyzers.parallel_scan_data_analyzer import ParallelScanDataAnalyzer
from PyGEECSPlotter.scan_data_analyzers.data_dir_analysis import DataDirAnalyzer
from PyGEECSPlotter.scan_data_analyzers.live_analysis import LiveJob, ScanWatcher
from PyGEECSPlotter.scan_data_analyzers.live_shot_analysis import LiveShotWatcher

__all__ = ["ScanDataAnalyzer", "ParallelScanDataAnalyzer", "DataDirAnalyzer", "LiveJob", "ScanWatcher",
           "LiveShotWatcher"]
