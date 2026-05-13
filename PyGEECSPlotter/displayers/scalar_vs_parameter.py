from typing import Optional, Dict, Any

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer


class ScalarVsParameter(ScanDisplayer):
    """
    Scatter / errorbar plot of one scalar column vs the scan parameter
    (or any other column).

    Parameters
    ----------
    y_col : str
        Column to plot on the y-axis. Must exist in ``scan.active_data``.
    x_col : str, optional
        Column to plot on the x-axis. Defaults to ``scan.scan_parameter``.
    bin_summary : {'none', 'mean', 'median', 'std_err'}, optional
        If not ``'none'``, also overlays a per-bin summary using
        ``scan.compute_bin_summary``.
    display_dict : dict, optional
        Style overrides: ``color``, ``alpha``, ``marker``, ``bin_color``,
        ``figsize``, ``ylabel``, ``xlabel``.
    """

    def __init__(
        self,
        y_col: str,
        x_col: Optional[str] = None,
        bin_summary: str = 'none',
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=f"{y_col}_vs_param", display_dict=display_dict)
        self.y_col = y_col
        self.x_col = x_col
        self.bin_summary = bin_summary

    def display(self, scan, fig=None, ax=None):
        fig, ax = self._new_fig(fig, ax)
        df = scan.active_data
        x_col = self.x_col or scan.scan_parameter

        if self.y_col not in df.columns:
            raise KeyError(f"Column '{self.y_col}' not in scan.active_data.")
        if x_col not in df.columns:
            raise KeyError(f"Column '{x_col}' not in scan.active_data.")

        ax.scatter(
            df[x_col],
            df[self.y_col],
            color=self.display_dict.get('color', 'k'),
            marker=self.display_dict.get('marker', 'o'),
            alpha=self.display_dict.get('alpha', 0.5),
            label='shots',
        )

        if self.bin_summary != 'none':
            center, spread = scan.compute_bin_summary(mode=self.bin_summary)
            if x_col in center.columns and self.y_col in center.columns:
                ax.errorbar(
                    center[x_col],
                    center[self.y_col],
                    yerr=spread[self.y_col],
                    fmt='o',
                    color=self.display_dict.get('bin_color', 'C3'),
                    capsize=3,
                    label=f'per-bin {self.bin_summary}',
                )
                ax.legend()

        ax.set_xlabel(self.display_dict.get('xlabel', x_col))
        ax.set_ylabel(self.display_dict.get('ylabel', self.y_col))
        ax.set_title(scan.scan_data_title())
        return fig, ax
