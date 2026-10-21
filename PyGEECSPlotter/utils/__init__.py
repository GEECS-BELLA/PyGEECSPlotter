"""Helpers shared across the package.

    fitting            curve fits, polynomial fits, linear regression
    navigation_utils   GEECS directory layout: top dirs, sfiles, scan / analysis dirs
    sfile_utils        masterlog helpers
    ecs_live_dumps     reading ECS live dump files
    ni_imread          12-bit / IMAQ PNG reader
    plotting, geecs_cmaps, pix_cmaps   matplotlib configuration and colormaps
    widgets            ipywidgets pickers
    general            miscellaneous helpers (the old ``PyGEECSPlotter.utils`` module)

``from PyGEECSPlotter.utils import <name>`` still works for names from the old
``utils`` module, with a ``DeprecationWarning``.
"""

import importlib as _importlib
import warnings as _warnings


def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    general = _importlib.import_module('PyGEECSPlotter.utils.general')
    if hasattr(general, name):
        _warnings.warn(f"'PyGEECSPlotter.utils.{name}' has moved to "
                       f"'PyGEECSPlotter.utils.general.{name}'; the old path is deprecated.",
                       DeprecationWarning, stacklevel=2)
        return getattr(general, name)
    raise AttributeError(f"module 'PyGEECSPlotter.utils' has no attribute {name!r}")
