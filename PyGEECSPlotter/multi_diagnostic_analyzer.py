# Per-shot analyzer that consumes multiple diagnostics simultaneously.
# Designed for cases where the analysis of one diagnostic depends on data
# from another (e.g. combining VIS + NIR spectra, or normalising an image
# by a separate energy meter reading).

from typing import Optional, Dict, Any, List, Tuple

from PyGEECSPlotter.diagnostic_analyzer import DiagnosticAnalyzer


class MultiDiagnosticAnalyzer(DiagnosticAnalyzer):
    """
    Per-shot analyzer consuming several diagnostics at once.

    Declares its required diagnostics via ``inputs`` (a list of
    ``(diagnostic_name, file_ext)`` tuples) and composes one
    ``DiagnosticAnalyzer`` per input via ``sub_analyzers``. The base
    ``ScanDataAnalyzer._iter_shots`` detects the multi case, looks up the
    per-shot file path for each declared input from the scan's
    ``<diagnostic> file_list`` columns, and passes a dict
    ``{name: data}`` to ``analyze_data``.

    Contract for subclasses:

    - ``analyze_data(data, bg=None, context=None)`` — ``data`` is
      ``{name: per-diagnostic-data}``. ``bg`` is either ``None`` or a
      parallel dict ``{name: per-diagnostic-bg}``. Returns the standard
      3-tuple ``(combined_data, results, aux)``.

    - The subclass owns output column naming. Per-sub-analyzer scalars
      should be merged into ``results`` with explicit prefixes
      (e.g. ``'vis peak wl (nm)'``, ``'nir peak wl (nm)'``) alongside
      whatever cross-diagnostic scalars the combination produces. The
      top-level ``output_diagnostic`` then gets prefixed to every column
      by ``ScanDataAnalyzer.merge_data_frame_to_sfile`` at sfile-write time.

    - ``load_data`` is provided by default and delegates to each
      ``sub_analyzers[name].load_data(path)``. Override only if the
      multi analyzer needs cross-input information at load time.
    """

    def __init__(
        self,
        inputs: List[Tuple[str, str]],
        sub_analyzers: Optional[Dict[str, DiagnosticAnalyzer]] = None,
        output_diagnostic: Optional[str] = None,
        output_file_ext: Optional[str] = None,
        analyzer_dict: Optional[Dict[str, Any]] = None,
        display_dict: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            diagnostic=None,
            file_ext=None,
            analyzer_dict=analyzer_dict,
            display_dict=display_dict,
            output_diagnostic=output_diagnostic,
            output_file_ext=output_file_ext,
        )
        self.inputs = list(inputs)
        self.sub_analyzers = dict(sub_analyzers or {})

    # ------------------------------------------------------------------
    # Scan-level integration
    # ------------------------------------------------------------------
    def register_with_scan(self, scan, remove_missing_files=True):
        """Add file_list / file_exists columns for every declared input."""
        for name, ext in self.inputs:
            scan.add_file_list_to_scan_data(name, ext, remove_missing_files)

    # ------------------------------------------------------------------
    # Pipeline contract
    # ------------------------------------------------------------------
    def load_data(self, paths):
        """
        Load all diagnostics for one shot.

        Parameters
        ----------
        paths : dict
            ``{diagnostic_name: file_path}`` — one entry per declared input.

        Returns
        -------
        dict
            ``{diagnostic_name: data}``. Entries with no registered
            sub-analyzer become ``None``.
        """
        out = {}
        for name, path in paths.items():
            sub = self.sub_analyzers.get(name)
            if sub is None:
                out[name] = None
            else:
                out[name] = sub.load_data(path)
        return out

    def analyze_data(self, data, bg=None, context=None):
        """
        Combine multi-diagnostic data for one shot.

        Parameters
        ----------
        data : dict
            ``{diagnostic_name: per-diagnostic-data}`` as produced by
            ``load_data``.
        bg : dict or None
            ``{diagnostic_name: per-diagnostic-bg}`` or ``None``.
        context : dict or None
            The shot's row as a dict (scan, Shotnumber, scalars, ...).

        Subclasses must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement analyze_data(data, bg, context). "
            f"`data` and `bg` are dicts keyed by diagnostic name."
        )

    # ------------------------------------------------------------------
    # Helper: pull the per-diagnostic background out of a bg dict
    # ------------------------------------------------------------------
    @staticmethod
    def _bg_for(bg, name):
        """Return ``bg[name]`` if ``bg`` is a dict and contains ``name``, else None."""
        if bg is None:
            return None
        if isinstance(bg, dict):
            return bg.get(name)
        return None
