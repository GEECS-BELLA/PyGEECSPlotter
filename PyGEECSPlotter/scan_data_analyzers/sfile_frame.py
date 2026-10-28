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
    repeated = [c for c in replaced if (base.columns == c).sum() > 1]
    if repeated:
        raise ValueError(f"The scan data has more than one column named {repeated}; "
                         f"cannot replace them with the results.")

    merged = base.drop(columns=replaced).merge(results, on=KEYS, how='left')
    merged.index = base.index
    if keep_unanalysed and replaced:
        analysed = base.set_index(KEYS).index.isin(results.set_index(KEYS).index)
        for c in replaced:
            merged[c] = merged[c].where(analysed, base[c])
    # Back to base's column order, by position: merged is base's other columns
    # (in order) followed by the result columns. (Selecting by label would
    # multiply any column name that base repeats.)
    kept = len(base.columns) - len(replaced)
    new_pos = {c: kept + i for i, c in enumerate(cols)}
    order, k = [], 0
    for c in base.columns:
        if c in new_pos and c in replaced:
            order.append(new_pos[c])
        else:
            order.append(k)
            k += 1
    order += [new_pos[c] for c in cols if c not in replaced]
    return merged.iloc[:, order]



# ---------------------------------------------------------------- controls
def read_controls(filename, kind, parse_from='python'):
    """
    Search / replace pairs (``kind='search_replace'``) or column-math
    clusters (``kind='column_math'``) from a controls file:
    ``parse_from='python'`` reads JSON written by ``write_controls_from_python``,
    ``'labview'`` a LabVIEW controls dump.
    """
    if parse_from == 'python':
        from PyGEECSPlotter.utils.io_utils import parse_controls_from_python
        return parse_controls_from_python(filename)
    if parse_from == 'labview':
        if kind == 'search_replace':
            return parse_search_replace_from_lv_controls(filename)
        return parse_column_math_from_lv_controls(filename)
    raise ValueError(f"parse_from must be 'python' or 'labview', got {parse_from!r}")


def apply_search_replace(df, pairs):
    """``df`` with each pair's ``Search`` replaced by ``Replace`` in every column name."""
    def rename(col):
        for pair in pairs:
            col = col.replace(pair['Search'], pair['Replace'])
        return col
    return df.rename(columns=rename)


def search_replace_text(text, pairs):
    """``text`` with every search / replace pair applied (e.g. the scan parameter)."""
    for pair in pairs:
        text = text.replace(pair['Search'], pair['Replace'])
    return text


_FORMULA_FUNCS = ["sqrt", "exp", "sin", "cos", "abs", "min", "max"]


def apply_column_math(df, clusters, verbose=False):
    """
    ``df`` with one new column per column-math cluster
    (``{'column_name', 'formula', 'variables': [{'vars', 'column_name'}]}``).
    A cluster whose input columns are missing, or whose formula fails, is
    skipped (reported if ``verbose``).
    """
    df = df.copy()
    for cluster in clusters:
        name = cluster['column_name']
        if not all(v['column_name'] in df.columns for v in cluster['variables']):
            if verbose:
                print(f"Column {name} not added — missing input columns.")
            continue
        formula = cluster['formula'].replace('^', '**')
        for func in _FORMULA_FUNCS:
            formula = formula.replace(func, f"np.{func}")
        local_vars = {v['vars']: df[v['column_name']] for v in cluster['variables']}
        try:
            df[name] = eval(formula, {"np": np}, local_vars)
        except Exception as e:
            if verbose:
                print(f"Error evaluating formula for {name}: {e}")
    return df


# ------------------------------------------------------- LabVIEW controls
def parse_column_math_from_lv_controls(file_path):
    clusters = []
    current_cluster = {}
    variables = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('Value') and '.Cluster.New (or old) column name' in line:
                if current_cluster:
                    current_cluster['variables'] = variables
                    clusters.append(current_cluster)
                    variables = []
                current_cluster = {'column_name': line.split('=')[1].strip()}
            elif line.startswith('Value') and '.Cluster.Formula' in line:
                current_cluster['formula'] = line.split('=')[1].strip()
            elif '.Cluster.vars' in line and '.Cluster.Column name' in line:
                # This is not expected to happen as per the provided format,
                # it seems to be a misunderstanding of the format.
                pass
            elif '.Cluster.vars' in line or '.Cluster.Column name' in line:
                var_name = line.split('=')[1].strip()
                var_key = 'vars' if '.Cluster.vars' in line else 'column_name'
                if variables and var_key in variables[-1]:
                    # If the last variable already has this key, start a new variable
                    variables.append({var_key: var_name})
                elif variables:
                    variables[-1][var_key] = var_name
                else:
                    variables.append({var_key: var_name})

        # Add the last cluster if it exists
        if current_cluster:
            current_cluster['variables'] = variables
            clusters.append(current_cluster)

    return clusters


def parse_search_replace_from_lv_controls(file_path):
    search_replace_pairs = []

    with open(file_path, 'r') as file:
        text_content = file.read()

    lines = text_content.strip().split('\n')
    for line in lines:
        if 'Cluster.Search=' in line:
            search_term = line.split('=')[1]
            if "Bella" in search_term:
                continue
            replace_term = next((l.split('=')[1] for l in lines if l.startswith(line.split('.Search')[0] + '.Replace=')), '')
            search_replace_pairs.append({'Search': search_term, 'Replace': replace_term})

    return search_replace_pairs
