# Binning helpers for ScanDataAnalyzer.
#
# Each helper returns a 1-D numpy int array of bin numbers, the same length
# as the input. Bins are 1-indexed; positions with NaN/non-finite input get
# bin 0 so they're easy to spot and skip downstream.

import numpy as np
import pandas as pd


def compute_bin_numbers(values, method='unique', **kwargs):
    """
    Return bin numbers for ``values`` using the requested method.

    Parameters
    ----------
    values : array-like
        Per-shot values to bin.
    method : str or callable, optional
        Either a registered method name or a callable
        ``f(values, **kwargs) -> ndarray``. Default ``'unique'``.

        Registered methods and their kwargs:

        ``'unique'``
            One bin per unique value, ordered ascending.
        ``'rounding'`` — ``rounding_factor``
            Round to a multiple of ``rounding_factor`` then ``unique``.
        ``'zscore'`` — ``z_threshold``
            Split sorted values at gaps with z-score above the threshold.
        ``'kmeans'`` — ``n_bins``, optional ``random_state``
            1-D K-means; bins ordered by cluster center.
        ``'edges'`` — ``bin_edges``
            Bin using explicit ``bin_edges`` array. Out-of-range → 0.
        ``'quantile'`` — ``n_bins``
            Equal-count (quantile) bins via ``pd.qcut``.
        ``'width'`` — ``bin_width`` or ``n_bins``
            Equal-width bins. ``bin_width`` may be a number or a string
            understood by ``np.histogram_bin_edges`` (``'auto'``,
            ``'fd'``, ``'scott'``, ``'sturges'``, ...). If neither
            argument is given, defaults to ``bin_width='auto'``.

    Returns
    -------
    ndarray of int
        1-indexed bin numbers, same length as ``values``. NaN /
        non-finite / out-of-range positions get ``0``.
    """
    if callable(method):
        result = method(values, **kwargs)
        return np.asarray(result, dtype=int)

    if method not in _METHODS:
        raise ValueError(
            f"Unknown binning method {method!r}. "
            f"Choose from {sorted(_METHODS)} or pass a callable."
        )
    return _METHODS[method](values, **kwargs)


def _bin_unique(values):
    """
    Assign one bin per unique value, ordered ascending.

    NaN / non-numeric entries map to bin 0.
    """
    series = pd.Series(values)
    codes, _ = pd.factorize(series, sort=True, use_na_sentinel=True)
    # codes: -1 for NaN, 0+ for valid (sorted).
    return np.where(codes >= 0, codes + 1, 0).astype(int)


def _bin_rounding(values, rounding_factor=1.0):
    """
    Round each value to the nearest multiple of ``rounding_factor`` and
    then bin by uniqueness.

    Useful for noisy continuous scan parameters where commanded values
    cluster around discrete targets.
    """
    if rounding_factor <= 0:
        raise ValueError(f"rounding_factor must be > 0, got {rounding_factor!r}.")
    values = np.asarray(values, dtype=float)
    rounded = np.round(values / rounding_factor) * rounding_factor
    return _bin_unique(rounded)


def _bin_zscore(values, z_threshold=1.0):
    """
    Sort ``values`` and split at gaps whose Z-score (computed across all
    consecutive differences) exceeds ``z_threshold``.

    Useful for parametric scans where the discrete commanded values are
    well-separated relative to their internal scatter, even when the
    spacing between targets is irregular.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    bin_numbers = np.zeros(n, dtype=int)

    finite_mask = np.isfinite(values)
    n_finite = int(finite_mask.sum())
    if n_finite == 0:
        return bin_numbers

    finite_positions = np.where(finite_mask)[0]
    finite_values = values[finite_positions]

    if n_finite == 1:
        bin_numbers[finite_positions] = 1
        return bin_numbers

    order = np.argsort(finite_values)
    sorted_values = finite_values[order]

    diffs = np.diff(sorted_values)
    if diffs.size == 0 or np.all(diffs == 0):
        bin_numbers[finite_positions] = 1
        return bin_numbers

    std_diff = diffs.std()
    if std_diff == 0:
        # all gaps equal → single bin
        bin_numbers[finite_positions] = 1
        return bin_numbers

    z_scores = (diffs - diffs.mean()) / std_diff
    gap_indices = np.where(z_scores > z_threshold)[0]

    sorted_bins = np.ones(n_finite, dtype=int)
    bin_num = 1
    cursor = 0
    for gap_idx in gap_indices:
        sorted_bins[cursor:gap_idx + 1] = bin_num
        cursor = gap_idx + 1
        bin_num += 1
    sorted_bins[cursor:] = bin_num

    # scatter back to original positions
    bin_numbers[finite_positions[order]] = sorted_bins
    return bin_numbers


def _bin_kmeans(values, n_bins=None, random_state=0):
    """
    1-D K-means clustering. Bins are ordered ascending by cluster center,
    so bin 1 corresponds to the lowest-valued cluster.

    Parameters
    ----------
    n_bins : int
        Target number of bins. Capped at the number of finite samples.
    random_state : int, optional
        Seed for K-means initialisation (default 0 for reproducibility).
    """
    if n_bins is None:
        raise ValueError("method='kmeans' requires n_bins.")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins!r}.")

    from sklearn.cluster import KMeans  # lazy import

    values = np.asarray(values, dtype=float)
    n = len(values)
    bin_numbers = np.zeros(n, dtype=int)

    finite_mask = np.isfinite(values)
    n_finite = int(finite_mask.sum())
    if n_finite == 0:
        return bin_numbers

    finite_positions = np.where(finite_mask)[0]
    finite_values = values[finite_positions].reshape(-1, 1)

    k = min(int(n_bins), n_finite)
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    raw_labels = km.fit_predict(finite_values)

    # Re-label so bin 1 is the lowest cluster center, etc.
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)
    label_map = np.empty(k, dtype=int)
    label_map[order] = np.arange(k)
    sorted_labels = label_map[raw_labels] + 1  # 1-indexed

    bin_numbers[finite_positions] = sorted_labels
    return bin_numbers


def _bin_edges(values, bin_edges=None):
    """
    Bin values into intervals defined by an explicit ``bin_edges`` array.

    ``bin_edges`` must be strictly increasing with at least two entries.
    For N edges, N-1 bins are produced. Values in
    ``[bin_edges[i-1], bin_edges[i])`` go to bin ``i``. Values outside
    the full range (below the first edge or at/above the last edge) go
    to bin ``0``.
    """
    if bin_edges is None:
        raise ValueError("method='edges' requires bin_edges.")
    bin_edges = np.asarray(bin_edges, dtype=float)
    if bin_edges.ndim != 1 or len(bin_edges) < 2:
        raise ValueError("bin_edges must be a 1-D array with at least 2 entries.")
    if not np.all(np.diff(bin_edges) > 0):
        raise ValueError("bin_edges must be strictly increasing.")

    values = np.asarray(values, dtype=float)
    indices = np.digitize(values, bin_edges)
    n_bins = len(bin_edges) - 1
    return np.where(
        (indices >= 1) & (indices <= n_bins) & np.isfinite(values),
        indices,
        0,
    ).astype(int)


def _bin_quantile(values, n_bins=None):
    """
    Equal-count (quantile) bins via ``pd.qcut``.

    If the data has duplicate values that prevent ``n_bins`` distinct
    quantile edges, fewer bins are produced (``duplicates='drop'``).
    """
    if n_bins is None:
        raise ValueError("method='quantile' requires n_bins.")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins!r}.")

    series = pd.Series(values)
    codes = pd.qcut(series, q=int(n_bins), labels=False, duplicates='drop')

    result = np.zeros(len(series), dtype=int)
    valid_mask = codes.notna().to_numpy()
    result[valid_mask] = codes[valid_mask].astype(int).to_numpy() + 1
    return result


def _bin_width(values, bin_width=None, n_bins=None):
    """
    Equal-width bins.

    Exactly one of ``bin_width`` or ``n_bins`` should be supplied:

    - ``n_bins`` (int): divide the data range into that many equal-width bins.
    - ``bin_width`` (float): bins of that width, starting from min(values).
    - ``bin_width`` (str): forwarded to ``np.histogram_bin_edges`` —
      accepts ``'auto'``, ``'fd'``, ``'scott'``, ``'sturges'``, etc.

    If both arguments are ``None``, defaults to ``bin_width='auto'``.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    bin_numbers = np.zeros(n, dtype=int)

    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return bin_numbers
    finite_values = values[finite_mask]
    vmin, vmax = float(finite_values.min()), float(finite_values.max())

    if n_bins is not None:
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins!r}.")
        if vmin == vmax:
            edges = np.array([vmin, vmin + 1.0])
        else:
            edges = np.linspace(vmin, vmax, int(n_bins) + 1)
    elif bin_width is not None:
        if isinstance(bin_width, str):
            edges = np.histogram_bin_edges(finite_values, bins=bin_width)
        else:
            if bin_width <= 0:
                raise ValueError(f"bin_width must be > 0, got {bin_width!r}.")
            if vmin == vmax:
                edges = np.array([vmin, vmin + float(bin_width)])
            else:
                edges = np.arange(vmin, vmax + float(bin_width), float(bin_width))
                if edges[-1] < vmax:
                    edges = np.append(edges, edges[-1] + float(bin_width))
    else:
        edges = np.histogram_bin_edges(finite_values, bins='auto')

    # Bump the rightmost edge so that vmax falls inside the last bin
    # (np.digitize with right=False treats it as out-of-range otherwise).
    edges = edges.astype(float).copy()
    edges[-1] = np.nextafter(edges[-1], np.inf)

    return _bin_edges(values, bin_edges=edges)


_METHODS = {
    'unique': _bin_unique,
    'rounding': _bin_rounding,
    'zscore': _bin_zscore,
    'kmeans': _bin_kmeans,
    'edges': _bin_edges,
    'quantile': _bin_quantile,
    'width': _bin_width,
}
