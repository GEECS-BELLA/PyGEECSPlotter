"""Per-shot diagnostic analyzers and readers.

Framework (generic, diagnostic-agnostic):
    DiagnosticAnalyzer, MultiDiagnosticAnalyzer   base classes / the per-shot contract
    DataReader                                    load only, no analysis
    ImageAnalyzer, ScaledImageAnalyzer            configurable 2D image pipeline
    ColumnMathAnalyzer                            new sfile columns from existing ones
    ShotCompareReader                             several analyzers on the same shot

Diagnostic-specific modules (import from their modules):
    spectrum_analysis, combined_vis_nir_spectra, frog_analysis,
    wavefront_analysis, camera_energy_analysis, alignment_analysis,
    magspec/ (needs the private pw_py_magspec package)
"""

from PyGEECSPlotter.analyzers.diagnostic_analyzer import DiagnosticAnalyzer
from PyGEECSPlotter.analyzers.multi_diagnostic_analyzer import MultiDiagnosticAnalyzer
from PyGEECSPlotter.analyzers.data_reader import DataReader
from PyGEECSPlotter.analyzers.image_analysis import ImageAnalyzer, ScaledImageAnalyzer
from PyGEECSPlotter.analyzers.column_math_analysis import ColumnMathAnalyzer
from PyGEECSPlotter.analyzers.shot_compare_analysis import ShotCompareReader

__all__ = [
    "DiagnosticAnalyzer",
    "MultiDiagnosticAnalyzer",
    "DataReader",
    "ImageAnalyzer",
    "ScaledImageAnalyzer",
    "ColumnMathAnalyzer",
    "ShotCompareReader",
]
