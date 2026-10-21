# Base class for per-shot diagnostic analyzers.
# Subclasses (ImageAnalyzer, ColumnMathAnalyzer, ...) implement the
# four-method pipeline below.

from typing import Optional, Dict, Tuple, Any


class DiagnosticAnalyzer:
    """
    Base class for per-shot diagnostic analyzers.

    The contract used by ``ScanDataAnalyzer.analyze_scan``:

    ``load_data(filename) -> data``
        Load a single shot's raw data from disk.

    ``analyze_data(data, *, bg=None, context=None) -> (data, results, aux)``
        Process one shot. ``results`` is a dict of scalars to be added to the
        sfile. ``aux`` is a dict of auxiliary per-shot outputs (e.g. lineouts)
        that the caller may write to disk.

    ``display_data(data, *, return_dict=None, title=None, fig=None, ax=None)
        -> (fig, ax)``
        Render a per-shot figure.

    ``write_analyzed_data(data, analysis_dir, scan, shot_num, *, context=None)``
        Write the analyzed shot data (typically an image or array) to disk
        under ``analysis_dir``.

    Subclasses are not required to implement every method. Methods that aren't
    relevant for a given analyzer (for example, ``display_data`` on a pure
    column-math analyzer) can be left as-is; calling them will raise
    ``NotImplementedError``.
    """

    def __init__(
        self,
        diagnostic: Optional[str] = None,
        file_ext: Optional[str] = None,
        analyzer_dict: Optional[Dict[str, Any]] = None,
        display_dict: Optional[Dict[str, Any]] = None,
        output_diagnostic: Optional[str] = None,
        output_file_ext: Optional[str] = None,
    ):
        self.diagnostic = diagnostic
        self.file_ext = file_ext
        self.analyzer_dict = dict(analyzer_dict) if analyzer_dict else {}
        self.display_dict = dict(display_dict) if display_dict else {}
        self.output_diagnostic = output_diagnostic
        self.output_file_ext = output_file_ext

    # ------------------------------------------------------------------
    # Pipeline contract
    # ------------------------------------------------------------------
    def load_data(self, filename):
        raise NotImplementedError(
            f"{type(self).__name__} must implement load_data(filename)."
        )

    def analyze_data(
        self,
        data,
        bg=None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """
        Process one shot.

        Returns
        -------
        data : Any
            Analyzed data (e.g. processed image, processed dataframe). May be
            ``None`` if the analyzer only produces scalars.
        results : dict
            Scalar outputs to merge back into the sfile.
        aux : dict
            Auxiliary outputs (e.g. lineouts) for optional disk writes.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement analyze_data(data, bg, context)."
        )

    def display_data(
        self,
        data,
        return_dict: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        fig=None,
        ax=None,
    ):
        raise NotImplementedError(
            f"{type(self).__name__} must implement display_data(data, ...)."
        )

    def write_analyzed_data(
        self,
        data,
        analysis_dir,
        scan,
        shot_num,
        context: Optional[Dict[str, Any]] = None,
    ):
        raise NotImplementedError(
            f"{type(self).__name__} must implement write_analyzed_data(...)."
        )

    # ------------------------------------------------------------------
    # Scan-level integration
    # ------------------------------------------------------------------
    def register_with_scan(self, scan, remove_missing_files=True):
        """
        Add the ``<diagnostic> file_list`` / ``<diagnostic> file_exists``
        columns this analyzer expects to ``scan.data``.

        Default implementation registers a single ``(diagnostic, file_ext)``
        pair. ``MultiDiagnosticAnalyzer`` overrides this to iterate over its
        declared inputs. Subclasses with no diagnostic (e.g.
        ``ColumnMathAnalyzer``) become a no-op automatically.
        """
        if self.diagnostic is None or self.file_ext is None:
            return
        scan.add_file_list_to_scan_data(self.diagnostic, self.file_ext, remove_missing_files)
