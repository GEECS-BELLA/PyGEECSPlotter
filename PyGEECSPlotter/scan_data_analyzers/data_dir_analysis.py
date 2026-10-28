# A plain folder of diagnostic files treated as a scan (no GEECS sfile).
# Author: Alex Picksley
# Version 0.2
# Created: 2024-02-26
# Last Modified: 2026-10-22

import glob
import os

import numpy as np
import pandas as pd

from PyGEECSPlotter.scan_data_analyzers.scan_data_analysis import ScanDataAnalyzer
from PyGEECSPlotter.scan_data_analyzers.sfile_frame import merge_results, rename_result_columns
from PyGEECSPlotter.utils.io_utils import file_lock


class DataDirAnalyzer(ScanDataAnalyzer):
    """
    Analyse a folder of files (e.g. images saved outside GEECS) like a scan.

    Each file matching ``*<file_ext>`` (sorted by name) becomes one shot,
    numbered from 1, all in bin 1 and with ``Shotnumber`` as the scan
    parameter, so analyzers, filters, ``rebin`` and every displayer work as
    on a GEECS scan. There is no sfile: ``write_columns_to_sfile=True``
    writes the results to ``<data_dir>/analysis/scalar_analysis.txt``
    instead, and figures are saved under ``<data_dir>/analysis``.

    Parameters
    ----------
    data_dir : str or Path
        Folder holding the files.
    file_ext : str
        Extension of the files to use, e.g. ``'.png'``.
    analyzer : DiagnosticAnalyzer, optional
        The analyzer you'll run; its ``diagnostic`` names the file-list
        column (``'<diagnostic> file_list'``) that it reads. Or give
        ``diagnostic`` directly.
    diagnostic : str, optional
        Name for the file-list column if no ``analyzer`` is given
        (default ``'diagnostic'``).
    scan : int
        Scan number recorded for every shot (default 1).
    """

    def __init__(self, data_dir, file_ext='.png', analyzer=None, diagnostic=None, scan=1):
        self._init_common()
        self.data_dir = os.fspath(data_dir)
        self.file_ext = file_ext
        self.scan = scan
        self.top_dir = self.data_dir
        if diagnostic is None:
            diagnostic = getattr(analyzer, 'diagnostic', None) or 'diagnostic'
        self.diagnostic = diagnostic

        file_list = sorted(glob.glob(os.path.join(self.data_dir, f'*{file_ext}')))
        if not file_list:
            raise FileNotFoundError(f"No '*{file_ext}' files in {self.data_dir}")
        n = len(file_list)
        self.data = pd.DataFrame({
            'scan': np.full(n, self.scan, dtype=np.int64),
            'Shotnumber': np.arange(1, n + 1),
            'Bin #': np.ones(n, dtype=np.int64),
            f'{diagnostic} file_list': file_list,
            f'{diagnostic} file_exists': np.ones(n, dtype=np.int64),
        })
        self.data['temp Bin number'] = self.data['Bin #']

        self.analysis_dir = os.path.join(self.data_dir, 'analysis')
        os.makedirs(self.analysis_dir, exist_ok=True)
        self.data_out_filename = os.path.join(self.analysis_dir, 'scalar_analysis.txt')
        self.scan_title = self.scan_data_title()

    def __repr__(self):
        return (f"DataDirAnalyzer(data_dir={self.data_dir}, diagnostic={self.diagnostic}, "
                f"num_shots={len(self.data)})")

    def scan_data_title(self, append_text=None):
        """The folder name, plus `` - append_text`` if given."""
        name = os.path.basename(os.path.normpath(self.data_dir)) if getattr(self, 'data_dir', None) else ''
        return ' - '.join(p for p in (name, append_text) if p)

    def load_scan_data(self, *args, **kwargs):
        """Nothing to load: the file list is built in the constructor."""
        analyzer = kwargs.get('analyzer')
        if analyzer is not None and getattr(analyzer, 'diagnostic', None) not in (None, self.diagnostic):
            raise ValueError(f"This DataDirAnalyzer's files are registered as {self.diagnostic!r}, "
                             f"but the analyzer reads {analyzer.diagnostic!r}; pass analyzer= "
                             f"(or diagnostic=) to DataDirAnalyzer.")

    def add_file_list_to_scan_data(self, diagnostic, file_ext, remove_missing_files=True):
        """Files come from the folder, not the GEECS layout; nothing to add."""
        if diagnostic != self.diagnostic:
            raise ValueError(f"This DataDirAnalyzer only has {self.diagnostic!r} files.")

    def get_scan_data_analysis_dir(self, make_dir=True):
        return self.analysis_dir

    def _shot_analysis_dir(self, scan):
        return self.analysis_dir

    def merge_data_frame_to_sfile(self,
                                  add_columns_df,
                                  diagnostic,
                                  overwrite_columns=True,
                                  analysis_label=None,
                                  keep_unanalysed=True,
                                  target_filename=None,
                                  ):
        """
        Write the results to ``scalar_analysis.txt`` (there is no sfile),
        merged with any results already there, as ``merge_data_frame_to_sfile``
        does for an sfile (``target_filename`` writes elsewhere instead).
        """
        target = os.fspath(target_filename) if target_filename is not None else self.data_out_filename
        new = rename_result_columns(add_columns_df, diagnostic, analysis_label)
        with file_lock(target):
            if os.path.exists(target):
                base = pd.read_csv(target, sep='\t')
            else:
                base = self.data[['scan', 'Shotnumber']].reset_index(drop=True)
            merged = merge_results(base, new, keep_unanalysed=keep_unanalysed)
            merged.to_csv(target, index=False, sep='\t')
        print(f'Columns added to {target}')
