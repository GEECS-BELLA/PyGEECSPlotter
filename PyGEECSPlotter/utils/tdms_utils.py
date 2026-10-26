"""Reading a scan's ``.tdms`` (one row per shot) while GEECS is still writing it.

GEECS writes ``scans/ScanNNN/ScanNNN.tdms`` during a scan, one group per
device, one channel per variable, named as the sfile columns are (e.g.
``'HEX-PL1-1 xpos'``, plus ``'Shotnumber'``, ``'Bin #'`` and ``'scan'``).
"""

import datetime
import os
import re
import time

import pandas as pd

from PyGEECSPlotter.utils.navigation_utils import alias_column_names, get_top_dir


def scan_tdms_path(top_dir, scan):
    """``<top_dir>/scans/ScanNNN/ScanNNN.tdms``."""
    name = f'Scan{int(scan):03d}'
    return os.path.join(os.fspath(top_dir), 'scans', name, f'{name}.tdms')


def alias_map(table_path):
    """
    ``{raw name: alias}`` from the header of an sfile or ScanData file
    (columns written ``'<raw> Alias:<alias>'``). The ``.tdms`` only has the
    raw names; GEECS adds the aliases when it writes the sfile at the end of
    the scan, so an earlier scan's sfile is used to name columns live.
    """
    header = pd.read_csv(os.fspath(table_path), sep='\t', nrows=0).columns
    return {c.split(' Alias:')[0]: name for c, name in alias_column_names(header).items()
            if ' Alias:' in c}


def latest_alias_map(top_dir, before_scan=None):
    """``alias_map`` of the newest sfile in ``top_dir/analysis`` (below
    ``before_scan`` if given), or ``{}`` if there is none."""
    analysis = os.path.join(os.fspath(top_dir), 'analysis')
    try:
        names = os.listdir(analysis)
    except FileNotFoundError:
        return {}
    scans = []
    for name in names:
        if name.startswith('s') and name.endswith('.txt') and name[1:-4].isdigit():
            n = int(name[1:-4])
            if before_scan is None or n < before_scan:
                scans.append(n)
    for n in sorted(scans, reverse=True):
        try:
            return alias_map(os.path.join(analysis, f's{n}.txt'))
        except Exception:
            continue
    return {}


def alias_map_for_scan(top_dir, scan, days_back=31):
    """
    ``latest_alias_map`` of the sfiles before ``scan`` in ``top_dir``; for the
    first scan of a day, that of the newest earlier day folder (up to
    ``days_back`` days back, same ``experiment_dir/Yyyyy/MM-Mon/yy_mmdd``
    layout) with an sfile. ``{}`` if none is found.
    """
    found = latest_alias_map(top_dir, before_scan=scan)
    if found:
        return found
    top_dir = os.path.normpath(os.fspath(top_dir))
    m = re.fullmatch(r'(\d{2})_(\d{2})(\d{2})', os.path.basename(top_dir))
    if not m:
        return {}
    experiment_dir = os.path.dirname(os.path.dirname(os.path.dirname(top_dir)))
    day = datetime.date(2000 + int(m.group(1)), int(m.group(2)), int(m.group(3)))
    for back in range(1, days_back + 1):
        d = day - datetime.timedelta(days=back)
        found = latest_alias_map(get_top_dir(experiment_dir, d.year, d.month, d.day))
        if found:
            return found
    return {}


def read_scan_tdms(path, aliases=None, retries=3, wait_s=0.2):
    """
    The ``.tdms`` at ``path`` as a DataFrame, one row per shot, with numeric
    values (GEECS stores them as text) and columns named as in a loaded sfile:
    channels in ``aliases`` (``{raw name: alias}``, see ``latest_alias_map``)
    are renamed to their alias.

    The file may be mid-write: a read that fails (e.g. a partly written last
    segment) is retried ``retries`` times, ``wait_s`` apart, and channels of
    unequal length (a device's value not written yet for the newest shot) are
    padded with NaN up to the longest. While GEECS is still writing the
    file it has no ``Shotnumber`` channel yet; it is then taken from the
    devices' ``'<device> shot #'`` channels. Returns None if the file doesn't
    exist or can't be read.
    """
    from nptdms import TdmsFile          # optional dependency (the 'live' extra)

    path = os.fspath(path)
    if not os.path.exists(path):
        return None
    aliases = aliases or {}
    for attempt in range(retries + 1):
        try:
            tdms = TdmsFile.read(path)
            columns = {}
            for group in tdms.groups():
                for channel in group.channels():
                    columns[aliases.get(channel.name, channel.name)] = channel[:]
            break
        except Exception:
            if attempt == retries:
                return None
            time.sleep(wait_s)
    if not columns:
        return pd.DataFrame()
    n = max(len(v) for v in columns.values())
    data = {}
    for name, values in columns.items():
        values = pd.Series(values)
        numeric = pd.to_numeric(values, errors='coerce')
        text = values.astype(str).str.strip()
        filled = values.notna() & ~text.isin(['', 'NaN', 'nan'])
        if numeric[filled].notna().all():
            values = numeric             # every entry is a number (blanks -> NaN)
        data[name] = values.reindex(range(n))
    df = pd.DataFrame(data)
    if 'Shotnumber' not in df.columns:
        # GEECS adds the Shotnumber, Bin # and scan groups only when it closes
        # the file; while the scan runs, each device's '<device> shot #'
        # channel numbers the rows (a device behind by a row is NaN there)
        shot_cols = [c for c in df.columns
                     if str(c).endswith(' shot #') and pd.api.types.is_numeric_dtype(df[c])]
        if shot_cols:
            df['Shotnumber'] = df[shot_cols].max(axis=1)
    return df
