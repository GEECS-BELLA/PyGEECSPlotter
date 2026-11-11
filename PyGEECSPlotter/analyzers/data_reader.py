# Base class for readers: per-shot "analyzers" that only load files.
# A reader computes nothing; it hands each shot's data to displayers (and to
# a MultiDiagnosticAnalyzer as a sub-analyzer) unchanged.

from PyGEECSPlotter.analyzers.diagnostic_analyzer import DiagnosticAnalyzer


class DataReader(DiagnosticAnalyzer):
    """
    Load one shot's file and pass it through unanalysed.

    ``load_data`` is ``DiagnosticAnalyzer``'s default: ``.png`` images via
    ``ni_imread`` (as float) and ``.txt`` tables (tab-separated with a
    header, or whitespace-separated without one). Any other extension
    raises ``ValueError``; subclass and override ``load_data`` for other
    formats, or for files that need special decoding (e.g.
    ``MagSpecAllEReader``'s integer-aC PNGs).

    ``analyze_data`` returns ``(data, {}, {})``: no scalars, no aux.
    Subclasses that build displayer-friendly ``aux`` from the loaded files
    override it.

    Use it on its own for raw files, as a sub-analyzer of a
    ``MultiDiagnosticAnalyzer``, or with ``diagnostic`` set to an
    analyzer's ``output_diagnostic`` to read back saved outputs
    (``add_file_list_to_scan_data`` falls back from ``scans/ScanNNN/<diag>``
    to ``analysis/ScanNNN/<diag>``).
    """

    def analyze_data(self, data, bg=None, context=None, analyzer_dict=None):
        return data, {}, {}
