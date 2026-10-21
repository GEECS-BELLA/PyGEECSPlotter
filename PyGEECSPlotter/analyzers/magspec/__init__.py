"""Magnetic spectrometer (allE) and e-beam profile analyzers.

Thin framework wrappers around the private ``pw_py_magspec`` package
(BellaCenter/PW-py-magspec), which must be installed to import these modules:

    magspec_alle_analysis   MagSpecAllEAnalyzer (10 cameras -> allE image + scalars)
    magspec_alle_reader     MagSpecAllEReader (reads the saved outputs back, fast)
    ebeam_profile_analysis  EBeamProfileAnalyzer

Nothing is imported here, so this package stays importable without it.
"""
