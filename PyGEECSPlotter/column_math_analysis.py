from PyGEECSPlotter.diagnostic_analyzer import DiagnosticAnalyzer


class ColumnMathAnalyzer(DiagnosticAnalyzer):
    """
    Base class for analyzers that compute new columns from existing scan data
    without loading any diagnostic file.

    Subclass this and implement
    ``analyze_data(self, data, *, bg=None, context=None)``. The ``context``
    dict is the full scan-data row (column_name -> value). Use it to read
    existing columns and return new ones in ``results``.

    Returns ``(None, results, {})`` — no per-shot image data, no auxiliary
    outputs.
    """

    def __init__(self, friendly_name=None):
        super().__init__(
            diagnostic=None,
            file_ext=None,
            analyzer_dict={
                "add_columns_to_masterlog": True,
                "write_analyzed": False,
                "analyze_raw_data": False,
                "update_scan_data_file": True,
            },
            display_dict={},
            output_diagnostic=None,
            output_file_ext=None,
        )
        self.friendly_name = friendly_name if friendly_name is not None else self.__class__.__name__

    def load_data(self, filename):
        return None

    def analyze_data(self, data, *, bg=None, context=None):
        """Override in subclasses to compute new columns from ``context``."""
        return None, {}, {}
