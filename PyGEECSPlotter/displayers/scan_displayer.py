# Base class for scan-level displays.
# A ScanDisplayer takes a ScanDataAnalyzer and produces a (fig, ax) summary
# of the scan. Multiple displayers can be applied to the same scan; each one
# is composable and savable.

import os
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt


class ScanDisplayer:
    """
    Base class for scan-level displays.

    Subclasses implement ``display(scan, fig=None, ax=None)`` which reads
    ``scan.active_data`` (and any other scan state) and returns ``(fig, ax)``.

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

    def save(self, fig, scan, suffix: str = "", dpi: int = 200):
        """Save ``fig`` under the scan's analysis directory."""
        analysis_dir = scan.get_scan_data_analysis_dir(make_dir=True)
        fname = f"Scan{int(scan.scan):03d}_{self.name}{suffix}.png"
        path = os.path.join(analysis_dir, fname)
        fig.savefig(path, dpi=dpi)
        return path
