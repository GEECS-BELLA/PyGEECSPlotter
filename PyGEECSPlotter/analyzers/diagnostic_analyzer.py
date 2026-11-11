# Base class for per-shot diagnostic analyzers.
# Subclasses (ImageAnalyzer, ColumnMathAnalyzer, ...) implement the
# four-method pipeline below.

import os
import re
from typing import Optional, Dict, Tuple, Any

import pandas as pd

_NUMERIC_LINE = re.compile(r'^[\s\d.eE+\-,;]*$')


class DiagnosticAnalyzer:
    """
    Base class for per-shot diagnostic analyzers.

    The contract used by ``ScanDataAnalyzer.analyze_scan``:

    ``load_data(filename) -> data``
        Load a single shot's raw data from disk. The default reads GEECS
        ``.png`` images and ``.txt`` tables (see ``load_data``).

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

    # analyze_scan runs load_data / analyze_data in a thread pool by default.
    # Set False in a subclass whose per-shot calls share unguarded mutable
    # state (it then needs analyze_scan(..., parallel=False)).
    thread_safe = True

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
        """
        Load one shot's file; None if it is missing. Default loaders:

        - ``.png``: ``ni_imread.read_imaq_image`` (GEECS camera PNGs, shifted
          by their ``sBIT`` depth), as float.
        - ``.txt``: a tab-separated table with a header row, or, when the
          first line is all numbers, a whitespace-separated table without
          one (e.g. optical spectrometers; columns 0, 1, ...).

        Override for other formats or for files that need decoding.
        """
        return load_shot_file(filename)

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

    # ------------------------------------------------------------------
    # Online analysis
    # ------------------------------------------------------------------
    def check_ready(self):
        """
        Raise (with the missing file names) if the analyzer can't analyse
        shots, e.g. default backgrounds or calibrations are missing. A live
        watcher calls it before watching. Default: ``check_backgrounds()`` if
        the analyzer has one, else nothing.
        """
        check = getattr(self, 'check_backgrounds', None)
        if callable(check):
            check()

    def live_payload(self, scan, shot, data, results, aux):
        """
        One shot for a live display in another process (see
        ``PyGEECSPlotter.displayers.live_gui``): the scalar ``results``, a 2-D
        ``data`` as a downsampled image, and the shot's 1-D view (a
        lineout-style ``aux`` pair such as ``'x'``/``'x_lo'``, or a 1-D
        ``data``) as the line and the waterfall row. Override to send
        something more specific (as the magspec analyzer does).
        """
        from PyGEECSPlotter.utils.live_payload import image_panel, lineout, scalars
        panel = image_panel(self.diagnostic or '', data, results)
        line = lineout(aux, data)
        return {'kind': 'shot', 'scan': scan, 'shot': shot, 'results': scalars(results),
                'images': [panel] if panel else [], 'line': line, 'wf': line}


def load_shot_file(filename):
    """The default ``DiagnosticAnalyzer.load_data``: a GEECS ``.png`` as a float
    image, or a ``.txt`` table as a DataFrame (header detected); None if the
    file is missing. Other extensions raise ``ValueError``."""
    if not isinstance(filename, (str, os.PathLike)) or not os.path.exists(filename):
        return None
    filename = os.fspath(filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.png':
        from PyGEECSPlotter.utils import ni_imread
        return ni_imread.read_imaq_image(filename).astype(float)
    if ext == '.txt':
        with open(filename, errors='replace') as f:
            first = f.readline()
        if _NUMERIC_LINE.match(first.strip()):
            return pd.read_csv(filename, sep=r'\s+', header=None)
        return pd.read_csv(filename, sep='\t')
    raise ValueError(f"No default loader for {ext!r} files ({filename}); override load_data.")
