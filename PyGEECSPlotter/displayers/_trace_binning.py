"""
Shared per-bin trace averaging, used by ``TraceMeanPerBin`` and
``TraceMeanWaterfall``. Not part of the public displayer API.
"""

import numpy as np
import pandas as pd


def mean_traces_per_bin(scan, analyzer, bg=None, bins=None):
    """Mean (and std) trace dict for each bin, in bin order."""
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
            per_bin.append(_mean_over_active(scan, analyzer, bg))
    finally:
        scan.restore_mask(saved)

    return bins, per_bin


def _mean_over_active(scan, analyzer, bg):
    """Stack the currently-active shots and average them column-wise."""
    stacks = {}
    for _, data, _, _ in scan._iter_shots(analyzer, bg=bg, show_progress=False):
        if data is None:
            continue
        for key, df in data.items():
            stacks.setdefault(key, []).append(df)

    if not stacks:
        return None

    mean_data, std_data = {}, {}
    for key, dfs in stacks.items():
        shapes = {df.shape for df in dfs}
        if len(shapes) > 1:
            raise ValueError(
                f"Per-bin averaging needs every shot on the same axis, but "
                f"the '{key}' traces have shapes {sorted(shapes)}. Give the "
                f"analyzer a common grid (e.g. analyzer_dict "
                f"{{'t_grid': (lo, hi, n)}})."
            )
        arr = np.stack([df.values.astype(float) for df in dfs], axis=0)
        cols = dfs[0].columns
        mean_data[key] = pd.DataFrame(np.nanmean(arr, axis=0), columns=cols)
        std_data[key] = pd.DataFrame(np.nanstd(arr, axis=0), columns=cols)

    return mean_data, std_data


def bin_labels(scan, bins, label_column=None, label_fmt='{:.4g}'):
    """Human labels for each bin: 'Bin {n}', or a summarised column value."""
    if label_column is False:
        return [f'Bin {int(b)}' for b in bins]

    col = label_column or scan.scan_parameter
    try:
        center_df, _ = scan.compute_bin_summary(mode='mean')
    except Exception:
        return [f'Bin {int(b)}' for b in bins]

    if col not in center_df.columns:
        return [f'Bin {int(b)}' for b in bins]

    by_bin = center_df.set_index('temp Bin number')[col]
    labels = []
    for b in bins:
        if b in by_bin.index:
            labels.append(label_fmt.format(by_bin.loc[b]))
        else:
            labels.append(f'Bin {int(b)}')
    return labels
