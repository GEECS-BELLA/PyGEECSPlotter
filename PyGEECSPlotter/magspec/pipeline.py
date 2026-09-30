# Raw magspec frames -> allE, chaining stage 1 (bellaMagspcTri) and
# stage 2 (fBellaSShotTri). In MATLAB the two stages communicate through
# files: integer-aC PNGs, '%.8e' Spec/Div tables and a '%.5e' frontSLX
# table. ``emulate_quantization=True`` reproduces that round trip so the
# Python results match the MATLAB outputs; set it False to keep full
# precision between the stages.

import numpy as np

from PyGEECSPlotter.magspec.io import quantize_int_ac
from PyGEECSPlotter.magspec.stage1 import run_stage1
from PyGEECSPlotter.magspec.stage2 import ProcessedWindow, run_stage2


def _sig(a, digits):
    """Round to what ``%.<digits>e`` + ``str2double`` gives back."""
    a = np.asarray(a, float)
    with np.errstate(invalid='ignore'):
        return np.array([float(f'{v:.{digits}e}') if np.isfinite(v) else v for v in a.ravel()]).reshape(a.shape)


def _window(img, win, field, quantize):
    mmt = 0.001 * field * win.mmt
    n_mmt = 0.001 * win.mmt
    dsp = field * win.dsp
    accp = win.accp
    if quantize:
        img, _ = quantize_int_ac(img)
        mmt, n_mmt, dsp, accp = (_sig(v, 8) for v in (mmt, n_mmt, dsp, accp))
    return ProcessedWindow(img, mmt, n_mmt, accp, dsp)


def run_alle(raw, bg, calib, field_T, ey_angle, roi=(0.01, 5.0), emulate_quantization=True):
    """Full allE analysis for one shot. Returns (stage1_result, stage2_result)."""
    s1 = run_stage1(raw, bg, calib, field_T, ey_angle)
    q = emulate_quantization
    high = _window(s1.high, s1.windows[0], field_T, q)
    low = _window(s1.low, s1.windows[1], field_T, q)
    angle = _sig(s1.angle.angl, 5) if q else s1.angle.angl
    fmm, fsig, fmmt = s1.front_x[0], s1.front_x[2], s1.front_x[3]
    if q:
        fmm, fsig, fmmt = _sig(fmm, 5), _sig(fsig, 5), _sig(fmmt, 5)
    s2 = run_stage2(high, low, angle, fmm, fsig, fmmt, calib.resolution, roi)
    return s1, s2
