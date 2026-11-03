"""Day trackers: figures for the whole experiment day, updated after each scan.

``DayImages`` shows one chosen shot of every scan with a shared colour scale,
``DayScalars`` per-scan mean ± std of some columns, ``ScanSheet`` (placeholder)
a one-row-per-scan table. ScanWatcher runs them from a ``ScanClassifier``'s
``day`` list and writes them to ``analysis/today/``.
"""

from PyGEECSPlotter.displayers.day.day_trackers import DayImages, DayScalars, ScanSheet

__all__ = ['DayImages', 'DayScalars', 'ScanSheet']
