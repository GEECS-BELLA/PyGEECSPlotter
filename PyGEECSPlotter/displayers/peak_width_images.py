from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from PyGEECSPlotter.displayers.representative_image_per_bin import RepresentativeImagePerBin


def _width_name(width_at):
    """'FWHM' for 0.5, 'FW at 0.135 max' otherwise."""
    return 'FWHM' if np.isclose(width_at, 0.5) else f'FW at {width_at:g} max'


def find_peak_bins(scan, peak_column, width_at=0.5, width_multiple=1.0, x_col=None):
    """
    Locate the peak of a per-bin mean curve and the bins either side of it.

    The curve is the per-bin mean of ``peak_column`` against the per-bin
    mean of ``x_col`` (default ``scan.scan_parameter``). Its peak is the bin
    with the largest mean; the width is measured at
    ``min + width_at*(max - min)`` (``width_at=0.5`` is the FWHM), with the
    crossings either side linearly interpolated between bins. If the curve
    doesn't drop to that level on one side, the half width comes from the
    other side alone.

    Returns a dict:
      'bins', 'x', 'y'          the per-bin curve (sorted by x)
      'level'                   the level the width is measured at
      'x_peak', 'hwhm'          peak position and half width (x units)
      'x_low', 'x_high'         x_peak -/+ width_multiple*hwhm
      'bin_low', 'bin_peak', 'bin_high'
                                bins whose mean x is nearest those positions
    """
    x_col = x_col or scan.scan_parameter
    active = scan.active_data
    for c in (x_col, peak_column):
        if c not in active.columns:
            raise KeyError(f"Column '{c}' not in scan.active_data.")

    center, _ = scan.compute_bin_summary(mode='mean')
    curve = center[['temp Bin number', x_col, peak_column]].dropna()
    if len(curve) < 3:
        raise RuntimeError(f"find_peak_bins needs at least 3 bins with finite "
                           f"'{x_col}' and '{peak_column}', got {len(curve)}.")
    curve = curve.sort_values(x_col)
    bins = curve['temp Bin number'].to_numpy()
    x = curve[x_col].to_numpy(float)
    y = curve[peak_column].to_numpy(float)

    i_pk = int(np.argmax(y))
    level = y.min() + width_at*(y.max() - y.min())

    def crossing(indices):
        # first pair (peak side -> outward) that straddles the level
        for a, b in zip(indices[:-1], indices[1:]):
            if y[b] < level <= y[a]:
                return x[a] + (level - y[a])*(x[b] - x[a])/(y[b] - y[a])
        return None

    x_left = crossing(list(range(i_pk, -1, -1)))
    x_right = crossing(list(range(i_pk, len(y))))
    if x_left is None and x_right is None:
        raise RuntimeError(f"'{peak_column}' never drops below {width_at:g} of its "
                           f"peak on either side; can't measure a width.")
    if x_left is not None and x_right is not None:
        hwhm = 0.5*(x_right - x_left)
    else:
        hwhm = abs((x_right if x_right is not None else x_left) - x[i_pk])

    x_peak = x[i_pk]
    x_low, x_high = x_peak - width_multiple*hwhm, x_peak + width_multiple*hwhm
    nearest = lambda v: bins[int(np.argmin(np.abs(x - v)))]
    return {
        'bins': bins, 'x': x, 'y': y, 'level': level,
        'x_peak': x_peak, 'hwhm': hwhm, 'x_low': x_low, 'x_high': x_high,
        'bin_low': nearest(x_low), 'bin_peak': bins[i_pk], 'bin_high': nearest(x_high),
    }


class PeakWidthImages(RepresentativeImagePerBin):
    """
    Three representative images: the peak bin, and the bins
    ``width_multiple`` half widths below and above it.

    The peak and width come from the per-bin mean of ``peak_column``
    (default ``parameter``) against the scan parameter (see
    ``find_peak_bins``). Each panel shows the shot with the largest
    ``parameter`` in its bin. Typical use: a focal scan, with ``parameter``
    the max counts, gives best focus and the images +-1 HWHM either side.

    Parameters
    ----------
    analyzer : DiagnosticAnalyzer
    parameter : str
        Column picking the shot shown in each bin (largest value).
    width_at : float, optional
        Fraction of the peak (above the minimum) the width is measured at;
        0.5 (default) is the FWHM.
    width_multiple : float, optional
        Distance of the outer panels from the peak, in half widths. Default 1.
    peak_column : str, optional
        Column whose per-bin mean defines peak and width. Default ``parameter``.
    bg, label_column, label_fmt, ncols, use_analyzer_display,
    suppress_labels, display_dict :
        See ``RepresentativeImagePerBin``. ``label_column`` defaults to the
        scan parameter.

    After ``display``, ``self.last_peak`` holds the ``find_peak_bins`` dict.
    """

    def __init__(
        self,
        analyzer,
        parameter: str,
        width_at: float = 0.5,
        width_multiple: float = 1.0,
        peak_column: Optional[str] = None,
        bg=None,
        label_column: Optional[str] = None,
        label_fmt: str = '{:.4g}',
        ncols: int = 3,
        use_analyzer_display: bool = True,
        suppress_labels: bool = True,
        display_dict: Optional[Dict[str, Any]] = None,
        output_subdir: Optional[str] = None,
        timestamp_files: bool = True,
    ):
        super().__init__(
            analyzer,
            mode='max',
            parameter=parameter,
            bg=bg,
            label_column=label_column,
            label_fmt=label_fmt,
            ncols=ncols,
            use_analyzer_display=use_analyzer_display,
            suppress_labels=suppress_labels,
            display_dict=display_dict,
            output_subdir=output_subdir,
            timestamp_files=timestamp_files,
        )
        self.name = f"{analyzer.output_diagnostic or analyzer.diagnostic}_peak_width_images"
        self.width_at = width_at
        self.width_multiple = width_multiple
        self.peak_column = peak_column or parameter
        self.last_peak = None

    def _select_rows(self, scan) -> List[Tuple[str, int]]:
        active = scan.active_data
        if self.parameter not in active.columns:
            raise KeyError(f"parameter column {self.parameter!r} not in active_data.")
        peak = find_peak_bins(scan, self.peak_column, width_at=self.width_at,
                              width_multiple=self.width_multiple,
                              x_col=self.label_column)
        self.last_peak = peak

        hw = _width_name(self.width_at).replace('FW', 'HW')
        m = f'{self.width_multiple:g}·' if self.width_multiple != 1 else ''
        tags = (f'−{m}{hw}', 'peak', f'+{m}{hw}')
        bin_col = active['temp Bin number'].to_numpy()
        selection: List[Tuple[str, int]] = []
        for tag, b in zip(tags, (peak['bin_low'], peak['bin_peak'], peak['bin_high'])):
            in_bin = np.where(bin_col == b)[0]
            if in_bin.size == 0:
                continue
            x_b = peak['x'][list(peak['bins']).index(b)]
            selection.append((f'{tag}\n{self.label_fmt.format(x_b)}',
                              self._pick_in_bin(active, in_bin)))
        return selection

    def _suptitle(self, scan) -> str:
        diag = self.analyzer.output_diagnostic or self.analyzer.diagnostic
        title = scan.scan_data_title(f'{diag} max {self._short_param()}, peak '
                                     f'+- {self.width_multiple:g} {_width_name(self.width_at)}/2')
        if self.last_peak is not None:
            title += f'\n{_width_name(self.width_at)} = {2*self.last_peak["hwhm"]:.3g}'
        return title
