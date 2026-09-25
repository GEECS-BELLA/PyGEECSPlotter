# Python port of the BELLA triangle-chamber magnetic spectrometer analysis
# (Kei Nakamura's MATLAB: bellaMagspcTri.m -> fBellaSShotTri.m). Pure
# numerics; the framework wrapper is PyGEECSPlotter.magspec_alle_analysis.
#
#   calibration  - ESMCalib file loading (cam / lanex / trajectory tables)
#   axes         - per-camera and stitched momentum / angle axes
#   image_proc   - bg subtraction, rotation, hot-pixel filter, stats
#   stage1       - raw frames -> highE / lowE / frontSL
#   stage2       - highE / lowE / frontSL -> allE + scalars
#   pipeline     - run_alle: both stages for one shot
#   io, matlab_compat - file formats and MATLAB-equivalent primitives

from PyGEECSPlotter.magspec.calibration import MAGSPEC_CAMERAS, MagSpecCalibration
from PyGEECSPlotter.magspec.pipeline import run_alle
