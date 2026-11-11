"""DataFrame helpers for sfile tables (no scan state).

Used by ScanDataAnalyzer for loading, renaming and merging per-shot results.
Rows are identified by the ``(scan, Shotnumber)`` key throughout.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

KEYS = ['scan', 'Shotnumber']


def summary_filename(scan, diagnostic=None, label=None):
    """``'Scan003_<diagnostic>_<label>_Summary.txt'``, leaving out an empty or
    None diagnostic / label (e.g. ``'Scan003_CAM-X_Summary.txt'``)."""
    parts = [f'Scan{int(scan):03d}'] + [str(p) for p in (diagnostic, label) if p not in (None, '')]
    return '_'.join(parts + ['Summary.txt'])


def provenance_filename(scan, diagnostic=None, label=None):
    """``summary_filename`` with ``_provenance.json`` in place of ``_Summary.txt``."""
    return summary_filename(scan, diagnostic, label)[:-len('Summary.txt')] + 'provenance.json'


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
PARSE_FROM = ('auto', 'python', 'labview')
_REQUIRED_KEYS = {'search_replace': ('Search', 'Replace'),
                  'column_math': ('column_name', 'formula', 'variables')}


def detect_controls_format(filename):
    """
    ``'python'`` if ``filename`` is JSON (as written by
    ``write_controls_from_python``), ``'labview'`` if it is a LabVIEW controls
    dump (``...Cluster.<name>=<value>`` lines); otherwise ValueError.
    """
    with open(filename, 'r') as file:
        text = file.read()
    try:
        json.loads(text)
        return 'python'
    except ValueError:
        pass
    if 'Cluster.' in text:
        return 'labview'
    raise ValueError(f"{filename} is neither a JSON controls file nor a LabVIEW controls dump.")


def read_controls(filename, kind, parse_from='auto'):
    """
    Search / replace pairs (``kind='search_replace'``) or column-math
    clusters (``kind='column_math'``) from one controls file:
    ``parse_from='python'`` reads JSON written by ``write_controls_from_python``,
    ``'labview'`` a LabVIEW controls dump, ``'auto'`` (default) detects which.
    """
    if parse_from == 'auto':
        parse_from = detect_controls_format(filename)
    if parse_from == 'python':
        from PyGEECSPlotter.utils.io_utils import parse_controls_from_python
        return parse_controls_from_python(filename)
    if parse_from == 'labview':
        if kind == 'search_replace':
            return parse_search_replace_from_lv_controls(filename)
        return parse_column_math_from_lv_controls(filename)
    raise ValueError(f"parse_from must be one of {PARSE_FROM}, got {parse_from!r}")


def _is_inline(spec):
    return (isinstance(spec, (list, tuple)) and len(spec) > 0
            and all(isinstance(e, dict) for e in spec))


def _is_file_pair(spec):
    return (isinstance(spec, tuple) and len(spec) == 2
            and isinstance(spec[0], (str, os.PathLike))
            and isinstance(spec[1], str) and spec[1] in PARSE_FROM)


def iter_control_sources(spec, parse_from='auto'):
    """
    ``spec`` as a list of ``(label, parse_from, payload)``, in order.

    ``spec`` is None (nothing), a path (read with ``parse_from``), a
    ``(path, parse_from)`` tuple, an inline list of dicts (already-parsed
    controls; ``parse_from`` None, label ``'<inline>'``), or a list / tuple of
    any of these. A list whose elements are all dicts is one inline source;
    a 2-tuple whose second item is one of ``PARSE_FROM`` is a file pair.
    """
    if spec is None:
        return []
    if isinstance(spec, dict):
        spec = [spec]
    if _is_inline(spec):
        return [('<inline>', None, list(spec))]
    if _is_file_pair(spec):
        return [(os.fspath(spec[0]), spec[1], os.fspath(spec[0]))]
    if isinstance(spec, (str, os.PathLike)):
        return [(os.fspath(spec), parse_from, os.fspath(spec))]
    if isinstance(spec, (list, tuple)):
        return [source for item in spec for source in iter_control_sources(item, parse_from)]
    raise TypeError(f"Can't read controls from {spec!r}: give a path, (path, parse_from), "
                    f"a list of dicts, or a list of these.")


def _check_entries(entries, kind, label):
    if not isinstance(entries, list):
        raise ValueError(f"{label}: expected a list of {kind} entries, got {type(entries).__name__}.")
    for i, entry in enumerate(entries):
        missing = [k for k in _REQUIRED_KEYS[kind] if not isinstance(entry, dict) or k not in entry]
        if kind == 'column_math' and not missing:
            missing = [f"variables[{j}].{k}" for j, v in enumerate(entry['variables'])
                       for k in ('vars', 'column_name') if k not in v]
        if missing:
            raise ValueError(f"{label}: {kind} entry {i} is missing {missing}.")


def load_controls(spec, kind, parse_from='auto'):
    """
    ``(entries, labels)``: the search / replace pairs or column-math clusters
    of every source in ``spec`` (see ``iter_control_sources``), concatenated
    in order, and one label per source (its path, or ``'<inline>'``).
    Each entry is checked for the keys ``kind`` needs.
    """
    if kind not in _REQUIRED_KEYS:
        raise ValueError(f"kind must be one of {tuple(_REQUIRED_KEYS)}, got {kind!r}")
    entries, labels = [], []
    for label, source_parse_from, payload in iter_control_sources(spec, parse_from):
        found = payload if source_parse_from is None else read_controls(payload, kind, source_parse_from)
        _check_entries(found, kind, label)
        entries += found
        labels.append(label)
    return entries, labels


def apply_search_replace(df, pairs):
    """
    ``df`` with each pair's ``Search`` replaced by ``Replace`` in every column
    name, pairs applied in order. Warns if the renames give two columns the
    same name.
    """
    renamed = df.rename(columns=lambda col: search_replace_text(col, pairs))
    before = set(df.columns[df.columns.duplicated()])
    clashes = sorted(set(renamed.columns[renamed.columns.duplicated()]) - before)
    if clashes:
        warnings.warn(f"Search / replace gave more than one column each of the names {clashes}.",
                      stacklevel=2)
    return renamed


def search_replace_text(text, pairs):
    """``text`` with every search / replace pair applied (e.g. the scan parameter)."""
    for pair in pairs:
        text = text.replace(pair['Search'], pair['Replace'])
    return text


# Names available in column-math formulas (besides the formula's variables).
FORMULA_NAMESPACE = {'np': np, 'sqrt': np.sqrt, 'exp': np.exp, 'log': np.log,
                     'log10': np.log10, 'sin': np.sin, 'cos': np.cos, 'abs': np.abs,
                     'min': np.min, 'max': np.max, 'pi': np.pi}


def apply_column_math(df, clusters, verbose=False):
    """
    ``df`` with one new column per column-math cluster
    (``{'column_name', 'formula', 'variables': [{'vars', 'column_name'}]}``),
    in order, so a formula can use columns made by earlier clusters.
    Formulas may use ``^`` for powers and the names in ``FORMULA_NAMESPACE``
    (``min`` / ``max`` are numpy reductions). A cluster whose input columns are
    missing, or whose formula fails, is skipped (reported if ``verbose``).
    """
    df = df.copy()
    for cluster in clusters:
        name = cluster['column_name']
        if not all(v['column_name'] in df.columns for v in cluster['variables']):
            if verbose:
                print(f"Column {name} not added — missing input columns.")
            continue
        formula = cluster['formula'].replace('^', '**')
        local_vars = {v['vars']: df[v['column_name']] for v in cluster['variables']}
        try:
            df[name] = eval(formula, dict(FORMULA_NAMESPACE), local_vars)
        except Exception as e:
            if verbose:
                print(f"Error evaluating formula for {name}: {e}")
    return df


# ------------------------------------------------------- LabVIEW controls
def parse_column_math_from_lv_controls(file_path):
    """Column-math clusters from a LabVIEW controls dump."""
    clusters = []
    current_cluster = {}
    variables = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if '=' not in line:
                continue
            value = line.split('=', 1)[1].strip()
            if line.startswith('Value') and '.Cluster.New (or old) column name' in line:
                if current_cluster:
                    current_cluster['variables'] = variables
                    clusters.append(current_cluster)
                    variables = []
                current_cluster = {'column_name': value}
            elif line.startswith('Value') and '.Cluster.Formula' in line:
                current_cluster['formula'] = value
            elif '.Cluster.vars' in line and '.Cluster.Column name' in line:
                pass
            elif '.Cluster.vars' in line or '.Cluster.Column name' in line:
                var_key = 'vars' if '.Cluster.vars' in line else 'column_name'
                if variables and var_key in variables[-1]:
                    # The last variable already has this key: start a new one
                    variables.append({var_key: value})
                elif variables:
                    variables[-1][var_key] = value
                else:
                    variables.append({var_key: value})

        if current_cluster:
            current_cluster['variables'] = variables
            clusters.append(current_cluster)

    return clusters


def parse_search_replace_from_lv_controls(file_path):
    """
    Search / replace pairs from a LabVIEW controls dump. Search terms
    containing ``'Bella'`` are skipped (as in the LabVIEW tool), and a Search
    with no matching Replace line gets ``''``.
    """
    with open(file_path, 'r') as file:
        lines = file.read().strip().split('\n')
    values = {line.split('=', 1)[0]: line.split('=', 1)[1] for line in lines if '=' in line}

    search_replace_pairs = []
    for line in lines:
        if 'Cluster.Search=' in line:
            search_term = line.split('=', 1)[1]
            if "Bella" in search_term:
                continue
            replace_term = values.get(line.split('.Search')[0] + '.Replace', '')
            search_replace_pairs.append({'Search': search_term, 'Replace': replace_term})

    return search_replace_pairs
