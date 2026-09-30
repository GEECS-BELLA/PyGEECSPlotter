# Base class for scan-level displays.
# A ScanDisplayer takes a ScanDataAnalyzer and produces a (fig, ax) summary
# of the scan. Multiple displayers can be applied to the same scan; each one
# is composable and savable.

import os
import re
import time
from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# Characters Windows (and some network shares) reject in filenames. ``name``
# often comes from a scalar column like 'Peak Power (TW/J)', which contains
# one of these.
_UNSAFE_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|]')


class ScanDisplayer:
    """
    Base class for scan-level displays.

    Subclasses implement ``display(scan, fig=None, ax=None)`` which reads
    ``scan.active_data`` (and any other scan state), returns ``(fig, ax)``,
    and sets ``self.last_export`` to a flat dict of the arrays that made the
    plot (so it can be written out with ``export()`` without recomputing
    anything, or reused directly in a notebook).

    Use:

        scan.display_scan(MyDisplayer(...), save=True)

    ``output_subdir``, if given, nests saved/exported files under that
    subfolder of the scan's analysis directory (``analysis_dir/output_subdir``)
    instead of dropping them straight in ``analysis_dir``. ``timestamp_files``
    (default True) appends a run timestamp to the filename so re-running the
    same displayer doesn't silently overwrite a previous run's output.
    """

    def __init__(
        self,
        name: str = "scan",
        display_dict: Optional[Dict[str, Any]] = None,
        output_subdir: Optional[str] = None,
        timestamp_files: bool = True,
    ):
        self.name = name
        self.display_dict = dict(display_dict) if display_dict else {}
        self.last_export: Optional[Dict[str, Any]] = None
        self.output_subdir = output_subdir
        self.timestamp_files = timestamp_files

    # ------------------------------------------------------------------
    # Subclasses override this.
    # ------------------------------------------------------------------
    def display(self, scan, fig=None, ax=None):
        raise NotImplementedError(
            f"{type(self).__name__} must implement display(scan, fig, ax)."
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _new_fig(self, fig=None, ax=None, **defaults):
        """Make a new (fig, ax) if one wasn't supplied, using display_dict for figsize."""
        if fig is not None and ax is not None:
            return fig, ax
        figsize = self.display_dict.get('figsize', defaults.get('figsize', (6, 5)))
        return plt.subplots(constrained_layout=True, figsize=figsize)

    def _output_dir(self, scan):
        """Scan's analysis directory, nested under ``output_subdir`` if set."""
        analysis_dir = scan.get_scan_data_analysis_dir(make_dir=True)
        if self.output_subdir:
            analysis_dir = os.path.join(analysis_dir, self.output_subdir)
            os.makedirs(analysis_dir, exist_ok=True)
        return analysis_dir

    def _output_stem(self, scan, suffix: str = ""):
        """
        Filesystem-safe ``Scan{n:03d}_{name}{suffix}[_{timestamp}]`` stem,
        shared by ``save()`` and ``export()``.

        ``name`` is often built from a scalar column (e.g. ``'... Peak Power
        (TW/J) ...'``), which can contain characters Windows rejects in
        filenames — those are replaced with ``_`` rather than passed through.
        A run timestamp is appended when ``timestamp_files`` is True, so
        re-running the same displayer over the same scan doesn't overwrite a
        previous run's output.
        """
        safe_name = _UNSAFE_FILENAME_CHARS.sub('_', f"{self.name}{suffix}")
        stem = f"Scan{int(scan.scan):03d}_{safe_name}"
        if self.timestamp_files:
            stem += time.strftime("_%Y%m%d-%H%M%S")
        return stem

    def save(self, fig, scan, suffix: str = "", dpi: int = 200):
        """Save ``fig`` under the scan's analysis directory (or ``output_subdir`` within it)."""
        output_dir = self._output_dir(scan)
        path = os.path.join(output_dir, self._output_stem(scan, suffix) + ".png")
        fig.savefig(path, dpi=dpi)
        print(f'Fig saved to {path}')
        return path

    def export(self, scan, suffix: str = ""):
        """
        Write ``self.last_export`` to an ``.npz`` alongside where ``save()``
        puts the PNG, so the plotted arrays can be reloaded without rerunning
        the analysis.

        Must be called after ``display()`` has populated ``last_export``.
        """
        if self.last_export is None:
            raise RuntimeError(
                f"{type(self).__name__}.export() called before display() "
                "populated last_export."
            )
        output_dir = self._output_dir(scan)
        path = os.path.join(output_dir, self._output_stem(scan, suffix) + ".npz")
        np.savez(path, **self.last_export)
        print(f'Data exported to {path}')
        return path
