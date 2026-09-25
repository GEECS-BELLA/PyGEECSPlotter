from typing import Optional, Dict, Any, Iterable

import numpy as np

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer


class CorrelationHeatmap(ScanDisplayer):
    """
    Heatmap of pairwise correlations between selected scalar columns.

    Parameters
    ----------
    columns : iterable of str
        Columns to include. Missing columns are silently dropped.
    method : {'pearson', 'spearman', 'kendall'}, optional
        Correlation method (forwarded to ``DataFrame.corr``).
    annotate : bool, optional
        If True, write the numeric correlation in each cell.
    display_dict : dict, optional
        Style overrides: ``cmap``, ``figsize``, ``vmin``, ``vmax``.
    """

    def __init__(
        self,
        columns: Iterable[str],
        method: str = 'pearson',
        annotate: bool = True,
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=f"corr_{method}", display_dict=display_dict)
        self.columns = list(columns)
        self.method = method
        self.annotate = annotate

    def display(self, scan, fig=None, ax=None):
        n = max(len(self.columns), 4)
        fig, ax = self._new_fig(fig, ax, figsize=(0.6 * n + 2, 0.6 * n + 2))

        cols = [c for c in self.columns if c in scan.active_data.columns]
        if not cols:
            raise ValueError("None of the requested columns are present in scan.active_data.")

        corr = scan.active_data[cols].corr(method=self.method)
        vmin = self.display_dict.get('vmin', -1)
        vmax = self.display_dict.get('vmax', 1)
        cmap = self.display_dict.get('cmap', 'RdBu_r')

        im = ax.imshow(corr.values, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols)

        if self.annotate:
            for i in range(len(cols)):
                for j in range(len(cols)):
                    val = corr.values[i, j]
                    if np.isfinite(val):
                        ax.text(
                            j, i, f"{val:.2f}",
                            ha='center', va='center',
                            color='white' if abs(val) > 0.5 else 'black',
                            fontsize=8,
                        )

        fig.colorbar(im, ax=ax, label=f'{self.method} correlation')
        ax.set_title(scan.scan_data_title('Correlation'))
        return fig, ax
