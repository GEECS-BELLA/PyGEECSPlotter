# Base class for readers: per-shot "analyzers" that only load files.
# A reader computes nothing; it hands each shot's data to displayers (and to
# a MultiDiagnosticAnalyzer as a sub-analyzer) unchanged.

import os

import pandas as pd

from PyGEECSPlotter.diagnostic_analyzer import DiagnosticAnalyzer
from PyGEECSPlotter import ni_imread


class DataReader(DiagnosticAnalyzer):
    """
    Load one shot's file and pass it through unanalysed.

    ``load_data`` picks the loader from the file extension:

    - ``.png`` : ``ni_imread.read_imaq_image`` (GEECS camera PNGs, shifted
      by their ``sBIT`` depth), as float.
    - ``.txt`` : ``pd.read_csv(sep='\\t')`` (GEECS tab-separated tables).

    Any other extension raises ``ValueError``; subclass and override
    ``load_data`` for other formats, or for files that need special
    decoding (e.g. ``MagSpecAllEReader``'s integer-aC PNGs).

    ``analyze_data`` returns ``(data, {}, {})``: no scalars, no aux.
    Subclasses that build displayer-friendly ``aux`` from the loaded files
    override it.

    Use it on its own for raw files, as a sub-analyzer of a
    ``MultiDiagnosticAnalyzer``, or with ``diagnostic`` set to an
    analyzer's ``output_diagnostic`` to read back saved outputs
    (``add_file_list_to_scan_data`` falls back from ``scans/ScanNNN/<diag>``
    to ``analysis/ScanNNN/<diag>``).
    """

    def load_data(self, filename):
        if not isinstance(filename, str) or not os.path.exists(filename):
            return None
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.png':
            return ni_imread.read_imaq_image(filename).astype(float)
        if ext == '.txt':
            return pd.read_csv(filename, sep='\t')
        raise ValueError(f"{type(self).__name__} can't read {ext!r} files "
                         f"({filename}); override load_data.")

    def analyze_data(self, data, bg=None, context=None, analyzer_dict=None):
        return data, {}, {}
