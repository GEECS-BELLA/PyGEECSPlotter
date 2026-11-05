"""Reading a scan's ``.tdms`` (one row per shot) while GEECS is still writing it.

GEECS writes ``scans/ScanNNN/ScanNNN.tdms`` during a scan, one group per
device, one channel per variable, named as the sfile columns are (e.g.
``'HEX-PL1-1 xpos'``, plus ``'Shotnumber'``, ``'Bin #'`` and ``'scan'``).
"""

import datetime
import os
import re
import struct
import time

import numpy as np
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


def _numeric_frame(columns):
    """``{name: values}`` (channels, possibly of unequal length) as a DataFrame
    padded with NaN to the longest. A text channel whose every filled entry
    (not blank / 'NaN') is a number becomes numeric, parsed by
    ``pd.to_numeric`` as ``read_csv`` parses the sfile (so the values are
    bit-identical to the sfile's); int64 if every entry is an integer, as
    ``pd.to_numeric`` gives. All text channels are parsed in one call: per
    channel it costs ~0.6 ms, i.e. ~0.25 s per read of a 400-channel file."""
    n = max(len(v) for v in columns.values())
    names = list(columns)
    text_names = [k for k in names if np.asarray(columns[k]).dtype.kind in 'OUS']
    parsed = {}
    if text_names:
        lengths = [len(columns[k]) for k in text_names]
        cells = pd.Series(np.concatenate([np.asarray(columns[k], dtype=object) for k in text_names]))
        number = pd.to_numeric(cells, errors='coerce').to_numpy(dtype=float)
        # text checks only where needed: unparsed cells (blank, or text?) ...
        nan = np.isnan(number)
        bad = np.zeros(len(cells), bool)
        if nan.any():
            text = cells[nan].astype(str).str.strip()
            bad[nan] = (cells[nan].notna() & ~text.isin(['', 'NaN', 'nan'])).to_numpy()
        # ... and the channels that may be integers (pd.to_numeric gives int64
        # if every entry is written as an integer)
        ends = np.cumsum(lengths)
        spans = [(k, b, e) for k, b, e in zip(text_names, ends - lengths, ends) if not bad[b:e].any()]
        whole = np.zeros(len(cells), bool)
        with np.errstate(invalid='ignore'):
            whole[np.isfinite(number)] = np.mod(number[np.isfinite(number)], 1) == 0
        maybe = [(k, b, e) for k, b, e in spans if whole[b:e].all()]
        integer = np.zeros(len(cells), bool)
        if maybe:
            idx = np.concatenate([np.arange(b, e) for _, b, e in maybe])
            integer[idx] = cells.iloc[idx].astype(str).str.strip().str.fullmatch(r'[+-]?\d+').to_numpy(dtype=bool)
        for k, b, e in spans:
            values = number[b:e]
            parsed[k] = values.astype(np.int64) if integer[b:e].all() else values
    def padded(values):
        values = np.asarray(values)
        if len(values) == n:
            return values
        if values.dtype.kind in 'biuf':
            return np.concatenate([values.astype(float), np.full(n - len(values), np.nan)])
        return np.concatenate([values.astype(object), np.full(n - len(values), np.nan, dtype=object)])
    return pd.DataFrame({k: padded(parsed.get(k, columns[k])) for k in names})


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
    return _scan_frame(columns)


def _scan_frame(columns):
    """``{name: values}`` of the channels as the scan's DataFrame (see ``read_scan_tdms``)."""
    if not columns:
        return pd.DataFrame()
    df = _numeric_frame(columns)
    if 'Shotnumber' not in df.columns:
        # GEECS adds the Shotnumber, Bin # and scan groups only when it closes
        # the file; while the scan runs, each device's '<device> shot #'
        # channel numbers the rows (a device behind by a row is NaN there)
        shot_cols = [c for c in df.columns
                     if str(c).endswith(' shot #') and pd.api.types.is_numeric_dtype(df[c])]
        if shot_cols:
            df['Shotnumber'] = df[shot_cols].max(axis=1)
    return df


class TdmsTail:
    """
    Reads a ``.tdms`` that is still being written, parsing only the segments
    added since the last read: GEECS writes a segment per shot, and parsing
    the whole file again costs ~3 ms per segment (~0.7 s at shot 250 of a
    400-channel scan), plus copying all of it over the network.

    ``frame(aliases)`` gives what ``read_scan_tdms`` gives. A segment still
    being written is left for the next read. If the file shrinks or its
    already-read part changes, it is read again from the start; a file this
    reader can't follow (scaled or DAQmx channels, or a segment that still
    fails once later ones exist) is read whole with ``read_scan_tdms`` from
    then on. Uses nptdms' segment reader (``nptdms.reader``).
    """

    def __init__(self, path):
        self.path = os.fspath(path)
        self.full = False                # fall back to read_scan_tdms
        self._reset()

    def _reset(self):
        self._reader = None
        self._pos = 0                    # end of the last complete segment read
        self._mark = None                # (position, lead-in bytes) of that segment
        self._previous = None            # that segment (its object list carries on)
        self._data = {}                  # object path -> [arrays]
        self._stuck = 0                  # failed reads with more file after the failing segment

    def _read_new(self):
        from nptdms.reader import TdmsReader
        with open(self.path, 'rb') as f:
            size = os.fstat(f.fileno()).st_size
            if self._mark is not None:
                f.seek(self._mark[0])
                if size < self._pos or f.read(28) != self._mark[1]:
                    self._reset()        # rewritten
            if size < 28:
                return                   # just created
            if self._reader is None:
                f.seek(0)                # TdmsReader checks the tag where the file is
                self._reader = TdmsReader(f)
            reader = self._reader
            reader._file, reader._data_file_size = f, size
            while self._pos + 28 <= size:
                f.seek(self._pos)
                lead_in = f.read(28)
                big = struct.unpack('<l', lead_in[4:8])[0] & (1 << 6)       # kTocBigEndian
                next_offset = struct.unpack(('>' if big else '<') + 'Q', lead_in[12:20])[0]
                if next_offset == 0xFFFFFFFFFFFFFFFF or self._pos + 28 + next_offset > size:
                    return               # still being written
                f.seek(self._pos)
                try:
                    segment, properties = reader._read_segment_metadata(f, self._pos, None, self._previous, False)
                    chunks = list(segment.read_raw_data(f))
                except EOFError:
                    return
                except Exception:
                    # a partly written segment; if later data exists, the file
                    # is one this reader can't follow
                    self._stuck += 1
                    if self._stuck >= 3:
                        raise
                    return
                self._stuck = 0
                reader._update_object_metadata(segment)
                reader._update_object_properties(properties)
                for chunk in chunks:
                    for path, data in chunk.channel_data.items():
                        if data.scaler_data:
                            raise ValueError('DAQmx data')
                        if data.data is not None and len(data.data):
                            self._data.setdefault(path, []).append(data.data)
                f.seek(self._pos)
                self._mark = (self._pos, f.read(28))
                self._previous = segment
                self._pos = segment.next_segment_pos

    def frame(self, aliases=None):
        """The scan's DataFrame so far (None if the file doesn't exist or can't be read)."""
        if not self.full:
            try:
                if not os.path.exists(self.path):
                    return None
                self._read_new()
                if self._reader is None:
                    return None
                from nptdms.common import ObjectPath
                aliases = aliases or {}
                # channels in nptdms' order: by group (groups with their own
                # object first, in file order), then in file order
                groups, channels = {}, {}
                for path, meta in self._reader.object_metadata.items():
                    if any(k.startswith('NI_Scal') or k == 'NI_Number_Of_Scales' for k in meta.properties):
                        raise ValueError('scaled channel')
                    p = ObjectPath.from_string(path)
                    if p.is_group:
                        groups.setdefault(p.group, None)
                    elif p.channel is not None:
                        channels.setdefault(p.group, []).append((p.channel, path))
                columns = {}
                for group in list(groups) + [g for g in channels if g not in groups]:
                    for name, path in channels.get(group, []):
                        parts = self._data.get(path)
                        if parts and len(parts) > 1:
                            parts[:] = [np.concatenate(parts)]      # joined once
                        columns[aliases.get(name, name)] = parts[0] if parts else np.array([])
                return _scan_frame(columns)
            except OSError:
                return None              # e.g. locked: next time
            except Exception:
                self.full = True         # read it whole from now on
        return read_scan_tdms(self.path, aliases=aliases, retries=0)
