"""Deprecated: the old ``PyGEECSPlotter.utils`` module.

Its functions now live in ``analysis_1d_utils``, ``analysis_2d_utils``,
``io_utils`` and ``plotting``; they are re-exported here so old imports keep
working. ``find_matching_row_index`` and ``polynomial_fit`` were unused and
have been removed (``fitting.poly_fit`` replaces the latter);
``merge_dicts_overwrite`` is ``{**a, **b}``.
"""

from PyGEECSPlotter.utils.analysis_1d_utils import (  # noqa: F401
    super_gaussian, calculate_moving_average_and_std, get_lineout_width,
)
from PyGEECSPlotter.utils.analysis_2d_utils import gaussian_2d, fit_gaussian_2d  # noqa: F401
from PyGEECSPlotter.utils.io_utils import (  # noqa: F401
    _convert_to_serializable, write_controls_from_python, parse_controls_from_python,
    save_lineouts_to_txt, load_lineouts_from_txt,
)
from PyGEECSPlotter.utils.plotting import plot_target_ellipse, plot_alignment_overview  # noqa: F401


def merge_dicts_overwrite(*dicts):
    """Merge dictionaries, later ones overwriting earlier keys."""
    result = {}
    for d in dicts:
        result.update(d)
    return result
