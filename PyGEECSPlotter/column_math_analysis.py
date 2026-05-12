class ColumnMathAnalyzer:
    """
    Base class for analyzers that compute new columns from existing scan data
    without loading any diagnostic file.

    Subclass this and implement analyze_data(data, context={}, bg=None).
    The `context` dict is the full scan-data row (column_name -> value). Use it
    to read existing columns and return new ones in results_dict.

    Returns (None, results_dict, None) - no image data, no lineouts.
    """

    def __init__(self, friendly_name=None):
        self.diagnostic = None
        self.file_ext = None
        self.output_diagnostic = None
        self.output_file_ext = None
        self.analyzer_dict = {
            "add_columns_to_masterlog": True,
            "write_analyzed": False,
            "analyze_raw_data": False,
            "update_scan_data_file": True,
        }
        self.display_dict = {}
        self.friendly_name = friendly_name if friendly_name is not None else self.__class__.__name__

    def load_data(self, filename):
        return None

    def analyze_data(self, data, context={}, bg=None):
        """Override this in subclasses to compute new columns from context."""
        return None, {}, None
