"""
Shared per-bin lineout averaging, used by ``LineoutMeanPerBin`` and
``LineoutMeanWaterfall``. Not part of the public displayer API.

Unlike ``_trace_binning.py`` (which averages the dict-of-DataFrames a trace
analyzer like ``FrogAnalyzer`` returns as ``data``), this averages the flat
``aux`` dict an ``ImageAnalyzer``-style analyzer returns alongside its image
(e.g. ``{'x': x, 'y': y, 'x_lo': x_lo, 'y_lo': y_lo}``).
"""

import numpy as np


def mean_lineouts_per_bin(scan, analyzer, bg=None, bins=None, axes=None):
    """
    Per-bin ``{axis: (coord, mean, std)}`` dicts, in bin order.

    ``axes`` restricts averaging to the named coordinates (e.g. ``['x']``);
    defaults to every coordinate found in a shot's ``aux`` that has a
    matching ``'{axis}_lo'`` entry.
    """
    if bins is None:
        bins = np.unique(scan.active_data['temp Bin number'])
    else:
        bins = np.asarray(list(bins))

    per_bin = []
    saved = scan.save_mask()
    try:
        for b in bins:
            scan.restore_mask(saved)
            scan.filter_scan_data('temp Bin number', b - 0.1, b + 0.1)
            per_bin.append(_mean_lineouts_over_active(scan, analyzer, bg, axes))
    finally:
        scan.restore_mask(saved)

    return bins, per_bin


def _mean_lineouts_over_active(scan, analyzer, bg, axes):
    """Stack the currently-active shots' lineouts and average them, per axis."""
    stacks = {}
    for _, _, _, aux in scan._iter_shots(analyzer, bg=bg, show_progress=False):
        if not aux:
            continue
        found_axes = axes if axes is not None else [
            k for k in aux if not k.endswith('_lo') and f'{k}_lo' in aux
        ]
        for axis in found_axes:
            lo_key = f'{axis}_lo'
            if axis not in aux or lo_key not in aux:
                continue
            coord = np.asarray(aux[axis], dtype=float)
            lo = np.asarray(aux[lo_key], dtype=float)
            stacks.setdefault(axis, []).append((coord, lo))

    if not stacks:
        return None

    result = {}
    for axis, entries in stacks.items():
        shapes = {c.shape for c, _ in entries}
        if len(shapes) > 1:
            raise ValueError(
                f"Per-bin lineout averaging needs every shot on the same "
                f"'{axis}' coordinate, but the lineouts have shapes "
                f"{sorted(shapes)}."
            )
        coord = entries[0][0]
        lo_stack = np.stack([lo for _, lo in entries], axis=0)
        result[axis] = (coord, np.nanmean(lo_stack, axis=0), np.nanstd(lo_stack, axis=0))

    return result
