# Day trackers: one figure per diagnostic, updated after every scan of the day,
# to spot drift (analysis/today/<name>_today.png).
#
# A tracker keeps a small per-scan cache in analysis/today/.cache/, so each
# update only analyses the scan that just ended and the figure is redrawn
# from the cache. ScanWatcher calls ``update(scan, today_dir, tags)`` after a
# scan's jobs; a restart or ``reanalyze`` refills the cache.

import glob
import math
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyGEECSPlotter.displayers.scan_displayer import ScanDisplayer, _UNSAFE_FILENAME_CHARS


def _cache_dir(today_dir):
    path = os.path.join(today_dir, '.cache')
    os.makedirs(path, exist_ok=True)
    return path


def _day_label(scan):
    return str(scan.date) if getattr(scan, 'date', None) else os.path.basename(scan.top_dir or '')


class DayImages(ScanDisplayer):
    """
    One shot of every scan of the day, side by side, with a shared colour
    scale: ``analysis/today/<name>_today.png``.

    analyzer : ImageAnalyzer (or any analyzer returning a 2-D image)
    select : 'first', 'max' or 'representative'
        Which shot of each scan: the first one; the one with the largest
        ``column``; or the one whose ``column`` is closest to the scan median.
    column : str
        Needed for 'max' / 'representative', e.g. ``'CAM-X max_counts'``
        (a column the scan's jobs have already added is fine).
    ncols : int, optional
        Panels per row (default: about square).
    name : str, optional
        File / cache name (default: the analyzer's diagnostic).
    display_dict : dict, optional
        ``cmap``, ``figsize``, ``max_figsize``, ``clim`` (fixed (lo, hi)).
    """

    def __init__(self, analyzer, select='first', column=None, ncols=None, name=None,
                 display_dict=None):
        if select not in ('first', 'max', 'representative'):
            raise ValueError("select must be 'first', 'max' or 'representative'.")
        if select != 'first' and column is None:
            raise ValueError(f"select='{select}' needs column=.")
        super().__init__(name=name or analyzer.diagnostic, display_dict=display_dict,
                         timestamp_files=False)
        self.analyzer = analyzer
        self.select = select
        self.column = column
        self.ncols = ncols

    @property
    def file_name(self):
        return _UNSAFE_FILENAME_CHARS.sub('_', self.name) + '_today.png'

    def _choose(self, scan):
        rows = scan.active_data
        if len(rows) == 0:
            return None, None
        if self.select == 'first':
            return rows.iloc[[0]], 'first shot'
        if self.column not in rows.columns:
            print(f"{scan.scan_label}: {self.name} today: no column {self.column!r}; using the first shot")
            return rows.iloc[[0]], 'first shot'
        values = rows[self.column]
        if values.notna().sum() == 0:
            return rows.iloc[[0]], 'first shot'
        if self.select == 'max':
            return rows.loc[[values.idxmax()]], f'max {self.column}'
        return rows.loc[[(values - values.median()).abs().idxmin()]], f'median {self.column}'

    def update(self, scan, today_dir, tags=()):
        """Add ``scan``'s chosen shot to the cache and redraw the day's figure."""
        self.analyzer.register_with_scan(scan)
        row, how = self._choose(scan)
        if row is None:
            print(f"{scan.scan_label}: {self.name} today: no shots with files; skipped")
            return None
        _, data, results, _ = next(scan._iter_shots(self.analyzer, rows=row, show_progress=False))
        if data is None:
            return None
        shot = int(row['Shotnumber'].iloc[0])
        extent = (results or {}).get('imshow_extent')
        np.savez_compressed(
            os.path.join(_cache_dir(today_dir), f'{self.file_name[:-4]}_Scan{int(scan.scan):03d}.npz'),
            image=np.asarray(data, dtype=np.float32), scan=int(scan.scan), shot=shot,
            how=how, tags=' '.join('#' + t for t in tags), day=_day_label(scan),
            extent=np.asarray(extent if extent is not None else [], dtype=float))
        return self.redraw(today_dir)

    def redraw(self, today_dir):
        """Draw every cached scan of the day into ``today_dir/<name>_today.png``."""
        files = sorted(glob.glob(os.path.join(_cache_dir(today_dir), f'{self.file_name[:-4]}_Scan*.npz')))
        if not files:
            return None
        entries = [dict(np.load(f, allow_pickle=False)) for f in files]
        n = len(entries)
        ncols = self.ncols or math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
        clim = self.display_dict.get('clim')
        if clim is None:
            values = np.concatenate([e['image'][np.isfinite(e['image'])].ravel()[::7] for e in entries])
            clim = (np.percentile(values, 0.5), np.percentile(values, 99.9)) if values.size else (0, 1)
        fig, axes = plt.subplots(nrows, ncols, squeeze=False, constrained_layout=True,
                                 figsize=self._grid_figsize(ncols, nrows))
        how = 'first shot' if self.select == 'first' else (
            f"{'max' if self.select == 'max' else 'median'} {self.column}")
        im = None
        for ax, e in zip(axes.ravel(), entries):
            extent = e['extent'] if e['extent'].size == 4 else None
            im = ax.imshow(e['image'], cmap=self.display_dict.get('cmap', 'viridis'),
                           vmin=clim[0], vmax=clim[1], extent=extent, origin='upper')
            title = f"Scan{int(e['scan']):03d} shot {int(e['shot'])}"
            if str(e['how']) != how:            # e.g. fell back to the first shot
                title += f" ({e['how']})"
            if str(e['tags']):
                title += f"\n{e['tags']}"
            ax.set_title(title, fontsize='small')
            self._hide_ticks(ax)
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        if im is not None:
            self._top_colorbar(fig, axes, im)
        fig.suptitle(f"{entries[-1]['day']}  {self.name}: {how} of each scan")
        os.makedirs(today_dir, exist_ok=True)
        path = os.path.join(today_dir, self.file_name)
        fig.savefig(path, dpi=self.display_dict.get('dpi', 120))
        plt.close(fig)
        print(f'Day figure updated: {path}')
        return path


class DayScalars:
    """
    Per-scan mean ± std of some columns, against scan number, for the whole
    day: ``analysis/today/<name>_today.png`` (one panel per column). The
    numbers behind it are in ``analysis/today/.cache/<name>.tsv``.

    columns : list of str, e.g. centroids and max counts
    """

    def __init__(self, columns, name='scalars', ncols=2):
        self.columns = list(columns)
        self.name = name
        self.ncols = ncols

    @property
    def file_name(self):
        return _UNSAFE_FILENAME_CHARS.sub('_', self.name) + '_today.png'

    def update(self, scan, today_dir, tags=()):
        rows = scan.active_data
        entry = {'scan': int(scan.scan), 'shots': len(rows), 'tags': ' '.join('#' + t for t in tags)}
        for c in self.columns:
            v = pd.to_numeric(rows[c], errors='coerce') if c in rows.columns else pd.Series(dtype=float)
            entry[f'{c} mean'] = v.mean()
            entry[f'{c} std'] = v.std()
        path = os.path.join(_cache_dir(today_dir), self.file_name[:-4] + '.tsv')
        table = pd.read_csv(path, sep='\t') if os.path.exists(path) else pd.DataFrame()
        if len(table):
            table = table[table['scan'] != entry['scan']]
        table = pd.concat([table, pd.DataFrame([entry])], ignore_index=True).sort_values('scan')
        table.to_csv(path, sep='\t', index=False)
        return self.redraw(today_dir, table, _day_label(scan))

    def redraw(self, today_dir, table, day=''):
        n = len(self.columns)
        ncols = min(self.ncols, n)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, squeeze=False, constrained_layout=True,
                                 figsize=(4.0 * ncols, 2.6 * nrows + 0.6))
        for ax, c in zip(axes.ravel(), self.columns):
            if f'{c} mean' in table:
                ax.errorbar(table['scan'], table[f'{c} mean'], yerr=table[f'{c} std'],
                            fmt='o-', ms=4, capsize=2)
            ax.set_title(c, fontsize='small')
            ax.set_xlabel('scan')
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        fig.suptitle(f'{day}  {self.name}: per-scan mean ± std')
        os.makedirs(today_dir, exist_ok=True)
        path = os.path.join(today_dir, self.file_name)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f'Day figure updated: {path}')
        return path


class ScanSheet:
    """
    Placeholder scan sheet: one row per scan in ``analysis/today/scan_sheet.tsv``
    (scan, time, shots, scan parameter and its range, tags, scan info).
    To be replaced by the experiment's own scan-sheet format.
    """

    name = 'scan_sheet'

    def update(self, scan, today_dir, tags=()):
        data = scan.data
        param = scan.scan_parameter
        values = pd.to_numeric(data[param], errors='coerce') if param in data.columns else pd.Series(dtype=float)
        try:
            info = scan.get_scan_info_text()
        except Exception:
            info = ''
        row = {'scan': int(scan.scan),
               'time': time.strftime('%H:%M', time.localtime(os.path.getmtime(scan.sfilename))),
               'shots': len(data), 'scan parameter': param,
               'min': values.min(), 'max': values.max(),
               'tags': ' '.join('#' + t for t in tags), 'info': info}
        os.makedirs(today_dir, exist_ok=True)
        path = os.path.join(today_dir, 'scan_sheet.tsv')
        table = pd.read_csv(path, sep='\t') if os.path.exists(path) else pd.DataFrame()
        if len(table):
            table = table[table['scan'] != row['scan']]
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True).sort_values('scan')
        table.to_csv(path, sep='\t', index=False)
        return path
