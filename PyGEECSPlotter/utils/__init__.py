"""Helpers shared across the package.

    fitting            curve fits, polynomial fits, linear regression
    navigation_utils   GEECS directory layout: top dirs, sfiles, scan / analysis dirs
    sfile_utils        masterlog helpers
    ecs_live_dumps     reading ECS live dump files
    ni_imread          12-bit / IMAQ PNG reader
    plotting, geecs_cmaps, pix_cmaps   matplotlib configuration and colormaps
    widgets            ipywidgets pickers
    analysis_2d_utils  image maths: ROI, threshold, centroids, moments, lineouts, masks
    analysis_1d_utils  lineout / trace / spectrum maths: widths, centroids, resampling
    io_utils           controls JSON, lineout text files, scaled 16-bit PNGs, file_lock
    tdms_utils         reading a scan's .tdms (one row per shot) while it is being written
    general            deprecated re-exports of the old ``PyGEECSPlotter.utils`` module

``from PyGEECSPlotter.utils import <name>`` still works for names from the old
``utils`` module, with a ``DeprecationWarning``.
"""

import importlib as _importlib
import warnings as _warnings


def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    try:  # a submodule, e.g. `from PyGEECSPlotter.utils import ni_imread`
        return _importlib.import_module(f'PyGEECSPlotter.utils.{name}')
    except ModuleNotFoundError as e:
        if e.name != f'PyGEECSPlotter.utils.{name}':
            raise
    general = _importlib.import_module('PyGEECSPlotter.utils.general')
    if hasattr(general, name):
        _warnings.warn(f"'PyGEECSPlotter.utils.{name}' has moved to "
                       f"its new module (see PyGEECSPlotter.utils.general); the old path is deprecated.",
                       DeprecationWarning, stacklevel=2)
        return getattr(general, name)
    raise AttributeError(f"module 'PyGEECSPlotter.utils' has no attribute {name!r}")
