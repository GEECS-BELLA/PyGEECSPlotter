"""Building blocks of the live-display message of a shot (see
``DiagnosticAnalyzer.live_payload`` and ``PyGEECSPlotter.displayers.live_gui``):
compact, picklable pieces, so a shot reaches the display process quickly."""

import math

import numpy as np

# lineout-style aux pairs (<axis>, <axis>_lo), most specific first: magspec
# momentum, spectrometer wavelength, image x
LINEOUT_AXES = ('p', 'wl', 'x')


def scalars(results):
    """The numeric entries of ``results`` as floats."""
    out = {}
    for k, v in (results or {}).items():
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out[k] = float(v)
    return out


def downsample(image, max_px=512):
    """A 2-D array block-averaged to at most ``max_px`` per side, as float32."""
    a = np.asarray(image, dtype=np.float32)
    k = max(1, math.ceil(max(a.shape) / max_px))
    if k == 1:
        return a
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def image_panel(name, data, results=None, max_px=512):
    """``{'name', 'image', 'extent'}`` for a 2-D image (None otherwise). The
    extent is ``results['imshow_extent']`` if the analyzer set one, else pixels."""
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.size == 0:
        return None
    extent = (results or {}).get('imshow_extent')
    if extent is None or len(np.ravel(extent)) != 4:
        extent = (-0.5, data.shape[1] - 0.5, -0.5, data.shape[0] - 0.5)
    return {'name': name, 'image': downsample(data, max_px),
            'extent': tuple(float(v) for v in np.ravel(extent))}


def lineout(aux, data=None):
    """``{'x', 'y'}`` (float32) for the shot's 1-D view: the first lineout-style
    pair in ``aux`` (``'p'``/``'p_lo'``, ``'wl'``/``'wl_lo'``, ``'x'``/``'x_lo'``,
    else any ``k``/``k_lo``), or a 1-D ``data`` itself. None if there is none."""
    aux = aux if isinstance(aux, dict) else {}
    keys = list(LINEOUT_AXES) + [k for k in aux if isinstance(k, str) and f'{k}_lo' in aux
                                 and k not in LINEOUT_AXES]
    for k in keys:
        x, y = aux.get(k), aux.get(f'{k}_lo')
        if x is None or y is None:
            continue
        x, y = np.ravel(np.asarray(x, dtype=np.float32)), np.ravel(np.asarray(y, dtype=np.float32))
        if x.size and x.size == y.size:
            return {'x': x, 'y': y}
    if isinstance(data, np.ndarray) and data.ndim == 1 and data.size:
        return {'x': np.arange(data.size, dtype=np.float32), 'y': data.astype(np.float32)}
    return None
