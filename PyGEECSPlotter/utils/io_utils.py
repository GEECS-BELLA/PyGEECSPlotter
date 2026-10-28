"""File helpers: controls JSON, lineout text files, scaled 16-bit PNGs, file locks."""

import contextlib
import json
import os
import time
import numpy as np
import pandas as pd
import imageio as imio

from PyGEECSPlotter.utils import ni_imread


@contextlib.contextmanager
def file_lock(path, timeout=60.0, stale_after=300.0, poll=0.1):
    """
    Hold ``<path>.lock`` while the block runs, so two processes (e.g. a live
    watcher and someone re-analysing by hand) don't rewrite ``path`` at once.

    The lock file is created atomically (``O_CREAT | O_EXCL``), which works on
    local disks and SMB shares. A lock older than ``stale_after`` seconds is
    taken to be left over from a crashed process and is removed. Raises
    ``TimeoutError`` if the lock can't be had within ``timeout`` seconds.
    """
    lock = os.fspath(path) + '.lock'
    t0 = time.monotonic()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > stale_after:
                    os.remove(lock)
                    continue
            except OSError:
                pass        # released (or removed) between the two calls
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"Could not lock {path} within {timeout:g} s ({lock} exists).")
            time.sleep(poll)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def _convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj


def write_controls_from_python(filename, controls_dict, print_data=False):
    controls_dict_serializable = _convert_to_serializable(controls_dict)
    with open(filename, 'w') as file:
        json.dump(controls_dict_serializable, file, indent=4)
    if print_data:
        print(f'Controls saved to : {filename}')


def parse_controls_from_python(filename):
    with open(filename, 'r') as file:
        controls_dict = json.load(file)
    return controls_dict


def save_lineouts_to_txt(lineouts, filename="lineouts.txt",
                         cols=None, float_format="%.3f", na_rep="nan"):
    """
    Save lineouts dict (arrays of different lengths) to a tab-delimited text file,
    padding with NaNs.

    Parameters
    ----------
    lineouts : dict
        Dict of 1D arrays, e.g. {'x': ..., 'x_lo': ..., 'y': ..., ...}
        Some keys may be missing (e.g. only 'x' and 'x_lo').
    filename : str
        Output filename.
    cols : list of str or None
        Column order to use. If None, use lineouts.keys() in their existing order.
        If provided, any names not found in lineouts are silently skipped.
    float_format : str
        Format for floats, e.g. '%.3f'.
    na_rep : str
        Representation of NaN in the file.
    """
    if cols is None:
        cols = list(lineouts.keys())
    else:
        # Only keep columns that actually exist in lineouts
        cols = [c for c in cols if c in lineouts]

    # Convert to arrays and find max length
    arrays = {k: np.asarray(lineouts[k]) for k in cols}
    max_len = max(len(arr) for arr in arrays.values())

    # Pad with NaN
    padded = {
        k: np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
        for k, arr in arrays.items()
    }

    df = pd.DataFrame(padded, columns=cols)
    df.to_csv(filename, sep="\t", index=False,
              float_format=float_format, na_rep=na_rep)
    return df


def load_lineouts_from_txt(filename):
    """
    Load lineouts from a tab-delimited file created by save_lineouts_to_txt.

    Automatically detects coord/value pairs of the form:
        <coord>, <coord> + '_lo'
    and uses the coord column's NaNs to remove padding from both.

    Returns
    -------
    lineouts : dict of np.ndarray
        Dict with original-length arrays.
    df : pandas.DataFrame
        The raw DataFrame as read from file (still padded with NaNs).
    """
    df = pd.read_csv(filename, sep="\t")

    cols = list(df.columns)
    lineouts = {}

    # --- Detect coord/value pairs automatically ---
    coord_pairs = []
    for col in cols:
        if not col.endswith("_lo"):
            lo_col = col + "_lo"
            if lo_col in df.columns:
                coord_pairs.append((col, lo_col))

    # Process each coord/value pair
    used_cols = set()
    for coord, val in coord_pairs:
        coord_vals = df[coord].to_numpy()
        mask = ~np.isnan(coord_vals)  # keep only real entries, drop padded NaNs

        lineouts[coord] = coord_vals[mask]
        lineouts[val] = df[val].to_numpy()[mask]

        used_cols.add(coord)
        used_cols.add(val)

    # Any remaining columns (not part of coord pairs) are returned as-is
    for col in cols:
        if col not in used_cols:
            lineouts[col] = df[col].to_numpy()

    return lineouts


def read_scaled_png(filename):
    """
    Read a 16-bit PNG written by ``write_scaled_png`` and undo the scaling,
    using the ``[Scaling] min / max`` lines of the ``.txt`` beside it.
    """
    data = ni_imread.read_imaq_image('%s' % filename)
    with open('%s.txt' % filename[:-4]) as f:
        lines = f.readlines()
    scale_min = float(lines[1].split(' ')[2])
    scale_max = float(lines[2].split(' ')[2])
    return data * (scale_max - scale_min) / (2**16 - 1) + scale_min


def write_scaled_png(save_path, data, nan_value=0):
    """
    Write float ``data`` as ``save_path + '.png'`` rescaled to the full 16-bit
    range, plus ``save_path + '.txt'`` holding the min / max for
    ``read_scaled_png``. NaNs become ``nan_value`` first.
    """
    data = np.nan_to_num(data, nan=nan_value)

    # Scale to the full 16-bit range; a flat frame has nothing to scale
    data_min, data_max = np.min(data), np.max(data)
    span = data_max - data_min
    if span > 0:
        data_int = (65535 * ((data - data_min) / span)).astype(np.uint16)
    else:
        data_int = np.zeros_like(data, dtype=np.uint16)

    lines = ['[Scaling]', 'min = %f' % data_min, 'max = %f' % data_max]
    with open(save_path + '.txt', 'w') as f:
        f.write('\n'.join(lines))
    imio.imwrite(save_path + '.png', data_int)
