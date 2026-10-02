"""
Shared per-bin lineout averaging, used by ``LineoutMeanPerBin`` and
``LineoutMeanWaterfall``. Not part of the public displayer API.

Unlike ``_trace_binning.py`` (which averages the dict-of-DataFrames a trace
analyzer like ``FrogAnalyzer`` returns as ``data``), this averages the flat
``aux`` dict an ``ImageAnalyzer``-style analyzer returns alongside its image
(e.g. ``{'x': x, 'y': y, 'x_lo': x_lo, 'y_lo': y_lo}``).
"""

import numpy as np


_ALIGN = ('snap', 'interp')


def on_grid(coord, lo, ref, align='snap'):
    """
    Lineout ``lo`` (sampled at ``coord``) on the reference axis ``ref``.

    Shots' axes differ when the origin is set per shot (ImageAnalyzer
    ``centroid_method='centroid'`` puts x = 0 at each shot's centroid).

    - ``align='snap'`` (default): shift by a whole number of samples, no
      interpolation. Needs both axes on one lattice, which ImageAnalyzer's
      ``round_centroid=True`` (its default) guarantees; raises otherwise.
    - ``align='interp'``: linear interpolation onto ``ref``, for axes off
      the lattice (e.g. ``round_centroid=False``, sub-pixel origins).

    Points of ``ref`` outside ``coord`` become NaN either way.
    """
    if align not in _ALIGN:
        raise ValueError(f"align must be one of {_ALIGN}, got {align!r}.")
    if coord.shape == ref.shape and np.allclose(coord, ref):
        return lo
    if align == 'interp':
        if coord[0] > coord[-1]:
            coord, lo = coord[::-1], lo[::-1]
        return np.interp(ref, coord, lo, left=np.nan, right=np.nan)

    step = ref[1] - ref[0]
    k = int(np.round((coord[0] - ref[0]) / step))
    if not np.allclose(coord, ref[0] + (np.arange(len(coord)) + k) * step):
        raise ValueError(
            f"Lineout axes are not on a common sample lattice (offset "
            f"{(coord[0] - ref[0]) / step:.3f} samples), so they can't be "
            f"aligned by a whole-sample shift. Use align='interp', or keep "
            f"ImageAnalyzer's analyzer_dict['round_centroid'] True."
        )
    out = np.full(len(ref), np.nan)
    ref_idx = np.arange(len(coord)) + k
    ok = (ref_idx >= 0) & (ref_idx < len(ref))
    out[ref_idx[ok]] = lo[ok]
    return out


def mean_lineouts_per_bin(scan, analyzer, bg=None, bins=None, axes=None, align='snap'):
    """
    Per-bin ``{axis: (coord, mean, std)}`` dicts, in bin order.

    ``axes`` restricts averaging to the named coordinates (e.g. ``['x']``);
    defaults to every coordinate found in a shot's ``aux`` that has a
    matching ``'{axis}_lo'`` entry. ``align`` is how shots on shifted
    axes are put on one axis; see ``on_grid``.
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
            per_bin.append(_mean_lineouts_over_active(scan, analyzer, bg, axes, align))
    finally:
        scan.restore_mask(saved)

    return bins, per_bin


def _mean_lineouts_over_active(scan, analyzer, bg, axes, align):
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
        # first shot's axis is the bin's reference; the rest are interpolated onto it
        coord = entries[0][0]
        lo_stack = np.stack([on_grid(c, lo, coord, align) for c, lo in entries], axis=0)
        result[axis] = (coord, np.nanmean(lo_stack, axis=0), np.nanstd(lo_stack, axis=0))

    return result
