# Backwards-compatible per-shot output writers.
#
# Ports `fBellaETxtSaveV02`, `fBellaATxtSaveV01`, and the integer-aC
# PNG block inlined in `bellaLiveMagspc2.m`. Column headers and units
# match the existing matlab quickE outputs exactly so downstream tooling
# keeps working.

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
from PIL import Image, PngImagePlugin

from PyGEECSPlotter.magspec.geometry import UniformAngleAxis, UniformMomentumWindow


# Header strings copied verbatim from `fBellaLabelV03` for byte-level
# backwards compatibility with existing analysis tooling.
_ENERGY_HEADERS = (
    'Momentum [GeV/c]',
    'Charge [fC]',
    'ChargeDen [pC/GeV/c]',
    'acepAngle [mrad]',
    'incAngle [deg]',
    'NrmMmt [GeV/c/T]',
    'dispersion [MeV/mm]',
)

_ANGLE_HEADERS = (
    'Angle [mrad]',
    'Charge [fC]',
    'ChargeDen [fC/mrad]',
)


# ----------------------------------------------------------------------
# Energy spectrum text file
# ----------------------------------------------------------------------
def write_energy_spectrum_txt(
    path: str,
    window: UniformMomentumWindow,
    image_aC: np.ndarray,
    field_T: float,
    fmt: str = '%.8g',
) -> None:
    """
    Write the 1-D energy spectrum text file. 7 columns:

      1. ``Momentum [GeV/c]`` = ``0.001 * field_T * window.mmt``
      2. ``Charge [fC]`` = column sum of ``image_aC`` × 0.001
      3. ``ChargeDen [pC/GeV/c]`` = column 2 / (``window.dp`` × ``field_T``)
      4. ``acepAngle [mrad]`` = ``window.acceptance``
      5. ``incAngle [deg]`` = ``window.inc_angle``
      6. ``NrmMmt [GeV/c/T]`` = ``0.001 * window.mmt``
      7. ``dispersion [MeV/mm]`` = ``field_T * window.dispersion``

    Ports ``fBellaETxtSaveV02``.

    Parameters
    ----------
    path : str
    window : UniformMomentumWindow
    image_aC : np.ndarray
        Stitched 2-D spectrum in aC per (mrad × MeV/c) bin, shape
        ``(angle_resolution, len(window.mmt))``.
    field_T : float
        Hall-probe field at this shot [T].
    fmt : str, optional
        Number formatter passed to ``np.savetxt``.
    """
    momentum_gev = 0.001 * field_T * window.mmt
    charge_fc = 0.001 * image_aC.sum(axis=0)
    denom = window.dp * field_T
    pc_per_gev = np.divide(
        charge_fc, denom,
        out=np.zeros_like(charge_fc),
        where=denom != 0,
    )
    table = np.column_stack([
        momentum_gev,
        charge_fc,
        pc_per_gev,
        window.acceptance,
        window.inc_angle,
        0.001 * window.mmt,
        field_T * window.dispersion,
    ])
    np.savetxt(
        path, table,
        delimiter='\t', fmt=fmt,
        header='\t'.join(_ENERGY_HEADERS), comments='',
    )


# ----------------------------------------------------------------------
# Angle distribution text file
# ----------------------------------------------------------------------
def write_angle_distribution_txt(
    path: str,
    uniform_angle: UniformAngleAxis,
    image_aC: np.ndarray,
    fmt: str = '%.8g',
) -> None:
    """
    Write the 1-D angular distribution text file. 3 columns:

      1. ``Angle [mrad]`` = ``uniform_angle.angle``
      2. ``Charge [fC]`` = row sum of ``image_aC`` × 0.001
      3. ``ChargeDen [fC/mrad]`` = column 2 / ``uniform_angle.d_angle``

    Ports ``fBellaATxtSaveV01``.
    """
    angle = uniform_angle.angle
    charge_fc = 0.001 * image_aC.sum(axis=1)
    da = uniform_angle.d_angle
    if da == 0:
        density = np.zeros_like(charge_fc)
    else:
        density = charge_fc / da
    table = np.column_stack([angle, charge_fc, density])
    np.savetxt(
        path, table,
        delimiter='\t', fmt=fmt,
        header='\t'.join(_ANGLE_HEADERS), comments='',
    )


# ----------------------------------------------------------------------
# Integer-aC PNG with embedded scale comment
# ----------------------------------------------------------------------
def _choose_scale_and_comment(image_aC: np.ndarray) -> Tuple[float, str]:
    """
    Pick the uint16 packing factor and matching ``Comment`` text, matching
    the inlined logic in ``bellaLiveMagspc2.m`` exactly.
    """
    max_val = float(np.nanmax(image_aC)) if image_aC.size else 0.0
    int_floor = int(np.ceil(max_val / (2 ** 16))) if max_val > 0 else 0
    if int_floor > 10:
        return 0.01, '100 aC/count'
    if int_floor > 1:
        return 0.1, '10 aC/count'
    return 1.0, '1 aC/count'


def write_integer_aC_png(path: str, image_aC: np.ndarray) -> str:
    """
    Save the 2-D stitched spectrum as a 16-bit PNG with a ``Comment``
    text chunk documenting the aC/count packing scale.

    Mirrors the ``intF`` / ``intCmnt`` block in ``bellaLiveMagspc2.m``:
    pick scale ∈ {1, 0.1, 0.01} so the largest value fits in uint16,
    write the resulting array, and embed e.g. ``'1 aC/count'`` as a PNG
    text chunk.

    Returns the comment that was written (so the caller can log it).
    """
    scale, comment = _choose_scale_and_comment(image_aC)
    packed = np.round(scale * image_aC).clip(0, 65535).astype(np.uint16)

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    pil_img = Image.fromarray(packed, mode='I;16')
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text('Comment', comment)
    pil_img.save(path, 'PNG', pnginfo=pnginfo)
    return comment
