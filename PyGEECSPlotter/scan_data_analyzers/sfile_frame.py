"""DataFrame helpers for sfile tables (no scan state).

Used by ScanDataAnalyzer for loading, renaming and merging per-shot results.
Rows are identified by the ``(scan, Shotnumber)`` key throughout.
"""

import numpy as np
import pandas as pd

KEYS = ['scan', 'Shotnumber']


def drop_incomplete_rows(sfile_data):
    """
    Drop rows with no ``scan`` or ``Shotnumber`` (e.g. a partial last row
    logged as the scan stopped). They can't be matched to any diagnostic
    file, and building file names from them fails.
    """
    keys = [c for c in KEYS if c in sfile_data.columns]
    if not keys:
        return sfile_data
    incomplete = sfile_data[keys].isna().any(axis=1)
    if incomplete.any():
        # +2: file line number (1-based, after the header line)
        lines = ', '.join(str(int(i) + 2) for i in np.flatnonzero(incomplete))
        print('Warning: dropping %d sfile row(s) with no scan / Shotnumber (file line %s)'
              % (incomplete.sum(), lines))
        sfile_data = sfile_data[~incomplete].reset_index(drop=True)
    return sfile_data


def rename_result_columns(results, diagnostic=None, analysis_label=None):
    """
    Keep the numeric columns of a per-shot results frame and name them
    ``'<diagnostic> <column> <analysis_label>'`` (``scan`` / ``Shotnumber``
    unchanged). The same names are used in memory and in the sfile.
    """
    results = results.select_dtypes(include=[np.number, 'bool'])
    if not diagnostic and not analysis_label:
        return results

    def rename(col):
        if col in KEYS:
            return col
        if diagnostic:
            col = f'{diagnostic} {col}'
        if analysis_label:
            col = f'{col} {analysis_label}'
        return col

    return results.rename(columns=rename)


def check_unique_keys(df, what):
    """Raise if ``(scan, Shotnumber)`` repeats in ``df`` (a merge would duplicate rows)."""
    dup = df.duplicated(KEYS, keep=False)
    if dup.any():
        pairs = df.loc[dup, KEYS].drop_duplicates().head(5).to_records(index=False).tolist()
        raise ValueError(f"{what} has repeated (scan, Shotnumber) rows, e.g. {pairs}; "
                         f"cannot merge results unambiguously.")


def merge_results(base, results, keep_unanalysed=True):
    """
    ``base`` with the columns of ``results`` merged in on ``(scan, Shotnumber)``.

    keep_unanalysed=True (default): for columns ``base`` already has, only
        the rows present in ``results`` are replaced; other rows keep their
        values. New columns are NaN outside ``results``.
    keep_unanalysed=False: each result column replaces ``base``'s whole
        column, so rows absent from ``results`` become NaN.

    Row order and count of ``base`` are preserved.
    """
    check_unique_keys(results, 'The results')
    check_unique_keys(base, 'The scan data')
    cols = [c for c in results.columns if c not in KEYS]
    replaced = [c for c in cols if c in base.columns]

    merged = base.drop(columns=replaced).merge(results, on=KEYS, how='left')
    merged.index = base.index
    if keep_unanalysed and replaced:
        analysed = base.set_index(KEYS).index.isin(results.set_index(KEYS).index)
        for c in replaced:
            merged[c] = merged[c].where(analysed, base[c])
    return merged[list(base.columns) + [c for c in cols if c not in base.columns]]
