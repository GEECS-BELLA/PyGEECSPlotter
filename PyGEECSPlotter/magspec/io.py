# File I/O for the magspec port: tab-separated calibration / log tables,
# 12-bit camera PNGs and the integer-aC PNGs written by the MATLAB code.

import io as _io
import os

import numpy as np
import pandas as pd
import png

from PyGEECSPlotter.magspec.matlab_compat import mround


def read_log(path):
    """Read a tab-separated table with a header row (``fLogReadV07``).

    All cells are returned as strings-parsed-to-float where possible;
    trailing empty columns (old LabVIEW logs end lines with a tab) are
    dropped."""
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith('Unnamed')]]
    return df


def find_column(columns, prefix, required=True):
    """First column whose name starts with ``prefix`` (``fLogClmnFindV01``).

    MATLAB compares the first ``len(prefix)`` characters, so a column
    named exactly ``prefix`` or ``prefix + anything`` matches."""
    for c in columns:
        if str(c)[:len(prefix)] == prefix:
            return c
    if required:
        raise KeyError(f'column starting with {prefix!r} not found')
    return None


def log_column(df, prefix):
    """Numeric column by prefix (``str2double(data(:, fLogClmnFindV01(...)))``)."""
    return pd.to_numeric(df[find_column(df.columns, prefix)], errors='coerce').to_numpy(float)


def _png_chunks(path):
    return list(png.Reader(filename=path).chunks())


def png_text(path, key):
    """Value of a tEXt chunk (e.g. ``'Comment'``), or None."""
    for ctype, data in _png_chunks(path):
        if ctype == b'tEXt':
            k, _, v = data.partition(b'\x00')
            if k.decode('latin-1') == key:
                return v.decode('latin-1')
    return None


def png_significant_bits(path):
    """sBIT of a greyscale PNG, or None if absent."""
    for ctype, data in _png_chunks(path):
        if ctype == b'sBIT':
            return int(data[0])
    return None


def read_png_raw(path):
    """Raw PNG samples as a 2-D uint array (MATLAB ``imread``)."""
    w, h, rows, meta = png.Reader(filename=path).read()
    arr = np.vstack([np.asarray(r) for r in rows])
    if meta.get('planes', 1) != 1:
        raise ValueError(f'{path}: expected a greyscale PNG')
    return arr


def open_12bit_png(path):
    """``f12bitPngOpnV04``: undo the IMAQ bit-depth shift.

    MATLAB does ``double(uint16_img / 2^(16 - sBIT))``; integer division in
    MATLAB rounds half away from zero, so this is not a plain right shift."""
    img = read_png_raw(path).astype(float)
    sbit = png_significant_bits(path)
    if sbit is None:
        return img
    return mround(img / 2.0 ** (16 - sbit))


def read_int_ac_png(path):
    """Read an integer-aC PNG written by ``fIntImgSave`` and return aC.

    The ``Comment`` chunk is ``'<N> aC/count'``; MATLAB parses
    ``str2double(cnvS(1:end-9))``."""
    img = read_png_raw(path).astype(float)
    cmnt = png_text(path, 'Comment')
    conv = float(cmnt[:-9]) if cmnt else 1.0
    return img * conv


def int_ac_factor(img):
    """``fIntImgSave`` scale choice: returns (multiplier, comment)."""
    int_fl = np.ceil(np.nanmax(img) / 2 ** 16)
    if int_fl > 10:
        return 0.01, '100 aC/count'
    if int_fl > 1:
        return 0.1, '10 aC/count'
    return 1.0, '1 aC/count'


def quantize_int_ac(img):
    """What ``fIntImgSave`` + ``fBellaReadPrcImgV01`` round-trip does to an
    aC image: scale, round, clip to uint16 (NaN -> 0), scale back."""
    fac, cmnt = int_ac_factor(img)
    q = mround(np.nan_to_num(fac * np.asarray(img, float), nan=0.0))
    q = np.clip(q, 0, 65535)
    return q / fac, cmnt


def write_int_ac_png(path, img):
    """``fIntImgSave``: uint16 PNG of an aC image with a ``Comment`` chunk."""
    q, cmnt = quantize_int_ac(img)
    fac = float(cmnt.split()[0])
    arr = mround(q / fac).astype(np.uint16)
    buf = _io.BytesIO()
    png.Writer(width=arr.shape[1], height=arr.shape[0], greyscale=True,
               bitdepth=16).write(buf, arr.tolist())
    buf.seek(0)
    chunks = list(png.Reader(file=buf).chunks())
    text = (b'tEXt', b'Comment\x00' + cmnt.encode('latin-1'))
    chunks.insert(1, text)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as fh:
        png.write_chunks(fh, chunks)
    return cmnt


def write_table(path, titles, columns, digits=8):
    """``fTxtOutV01``: tab-separated header + ``%.<digits>e`` columns."""
    data = np.column_stack([np.asarray(c, float).ravel() for c in columns])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fmt = '\t'.join([f'%.{digits}e'] * len(titles))
    with open(path, 'w', newline='\n') as fh:
        fh.write('\t'.join(titles) + '\n')
        for row in data:
            fh.write(fmt % tuple(row) + '\n')
