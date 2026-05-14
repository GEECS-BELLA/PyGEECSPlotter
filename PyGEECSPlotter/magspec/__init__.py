"""
BELLA magnetic-spectrometer port.

Phase 1 (landed): calibration loaders + axis/geometry derivation.
Phase 2 (this commit): per-camera image processing + background loading.
Phase 3 (next): MagSpecAnalyzer(MultiDiagnosticAnalyzer) + stitching.

See MAGSPEC_PORT.md at the repo root for the function-by-function port map
and design decisions.
"""

from PyGEECSPlotter.magspec.calibrations import (
    CameraCalibration,
    TrajectoryCalibration,
    LanexCalibration,
    discover_calib_path,
    load_camera_calibration,
    load_trajectory_calibration,
    load_lanex_calibration_table,
    compute_c2c_and_vignette,
)
from PyGEECSPlotter.magspec.geometry import (
    CameraXAxis,
    CameraYAxis,
    AngleMap,
    UniformAngleAxis,
    UniformMomentumWindow,
    compute_camera_axis,
    compute_all_axes,
    compute_angle_maps,
    uniform_angle_axis,
    uniform_momentum_axes,
)
from PyGEECSPlotter.magspec.image_processing import (
    count_saturated_pixels,
    subtract_background_iterative,
    lowpass_xray_filter,
    rotate_around_center,
    process_camera_image,
)
from PyGEECSPlotter.magspec.backgrounds import (
    load_or_build_background,
)

__all__ = [
    # calibrations
    "CameraCalibration",
    "TrajectoryCalibration",
    "LanexCalibration",
    "discover_calib_path",
    "load_camera_calibration",
    "load_trajectory_calibration",
    "load_lanex_calibration_table",
    "compute_c2c_and_vignette",
    # geometry
    "CameraXAxis",
    "CameraYAxis",
    "AngleMap",
    "UniformAngleAxis",
    "UniformMomentumWindow",
    "compute_camera_axis",
    "compute_all_axes",
    "compute_angle_maps",
    "uniform_angle_axis",
    "uniform_momentum_axes",
    # image_processing
    "count_saturated_pixels",
    "subtract_background_iterative",
    "lowpass_xray_filter",
    "rotate_around_center",
    "process_camera_image",
    # backgrounds
    "load_or_build_background",
]
