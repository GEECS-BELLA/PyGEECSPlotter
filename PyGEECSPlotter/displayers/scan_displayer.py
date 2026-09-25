# Base class for scan-level displays.
# A ScanDisplayer takes a ScanDataAnalyzer and produces a (fig, ax) summary
# of the scan. Multiple displayers can be applied to the same scan; each one
# is composable and savable.

import os
import re
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
    """

    def __init__(
        self,
        name: str = "scan",
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.display_dict = dict(display_dict) if display_dict else {}
        self.last_export: Optional[Dict[str, Any]] = None

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

    def _output_stem(self, scan, suffix: str = ""):
        """
        Filesystem-safe ``Scan{n:03d}_{name}{suffix}`` stem, shared by
        ``save()`` and ``export()``.

        ``name`` is often built from a scalar column (e.g. ``'... Peak Power
        (TW/J) ...'``), which can contain characters Windows rejects in
        filenames — those are replaced with ``_`` rather than passed through.
        """
        safe_name = _UNSAFE_FILENAME_CHARS.sub('_', f"{self.name}{suffix}")
        return f"Scan{int(scan.scan):03d}_{safe_name}"

    def save(self, fig, scan, suffix: str = "", dpi: int = 200):
        """Save ``fig`` under the scan's analysis directory."""
        analysis_dir = scan.get_scan_data_analysis_dir(make_dir=True)
        path = os.path.join(analysis_dir, self._output_stem(scan, suffix) + ".png")
        fig.savefig(path, dpi=dpi)
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
        analysis_dir = scan.get_scan_data_analysis_dir(make_dir=True)
        path = os.path.join(analysis_dir, self._output_stem(scan, suffix) + ".npz")
        np.savez(path, **self.last_export)
        return path
