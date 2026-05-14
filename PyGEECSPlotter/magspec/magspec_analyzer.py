# MagSpecAnalyzer — per-shot multi-camera magnetic-spectrometer pipeline.
#
# Pulls together everything from the earlier phases into one
# MultiDiagnosticAnalyzer subclass that loads N camera images per shot,
# stitches them onto uniform (angle, momentum) axes, scales the momentum
# axis by the per-shot Hall-probe field, and produces:
#   * the standard backwards-compatible quickE* outputs per window
#   * per-window scalar stats (peak / mean / fwhm / std) for the sfile

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from PyGEECSPlotter.multi_diagnostic_analyzer import MultiDiagnosticAnalyzer
from PyGEECSPlotter.ni_imread import read_imaq_image
from PyGEECSPlotter.magspec.analysis import spot_analysis
from PyGEECSPlotter.magspec.backgrounds import load_or_build_background
from PyGEECSPlotter.magspec.calibrations import (
    CameraCalibration,
    LanexCalibration,
    TrajectoryCalibration,
    compute_c2c_and_vignette,
    discover_calib_path,
    load_camera_calibration,
    load_lanex_calibration_table,
    load_trajectory_calibration,
)
from PyGEECSPlotter.magspec.geometry import (
    AngleMap,
    CameraXAxis,
    CameraYAxis,
    UniformAngleAxis,
    UniformMomentumWindow,
    compute_all_axes,
    compute_angle_maps,
    uniform_angle_axis,
    uniform_momentum_axes,
)
from PyGEECSPlotter.magspec.image_processing import process_camera_image
from PyGEECSPlotter.magspec.outputs import (
    write_angle_distribution_txt,
    write_energy_spectrum_txt,
    write_integer_aC_png,
)
from PyGEECSPlotter.magspec.stitch import stitch_all_windows


# Defaults pulled from bellaLiveMagspc2.m
_DEFAULT_LOWPASS = (2.0, 1, 1.0, 2)        # (factor, pit, min_counts, n_iter)
_DEFAULT_MMT_RES = {'front': 512, 'side': 1024}
_DEFAULT_ACCEPTANCE_MM = {'front': 33.0, 'side': 40.0}
_DEFAULT_ANGLE_RANGE = (-1.3, 1.3)


class MagSpecAnalyzer(MultiDiagnosticAnalyzer):
    """
    Multi-camera magnetic-spectrometer analyzer.

    Construction loads day-specific calibrations, computes per-camera +
    uniform axes once, builds per-camera vignette/c2c, and pre-loads (or
    rebuilds) the per-camera averaged backgrounds for the designated
    background scan.

    ``analyze_data`` then per-shot:
      1. Processes each camera image (bg / rotate / ROI / low-pass /
         vignette / cos(incAngle) / c2c / sensitivity).
      2. Stitches per-screen images onto the window's uniform axes.
      3. Reads the Hall-probe field from ``context`` and scales the
         momentum axis by it at output time.
      4. Returns ``(stitched_per_screen, results, aux)``.

    See ``MAGSPEC_PORT.md`` for the function-by-function map.
    """

    # ------------------------------------------------------------------
    # Construction — load calibrations / geometry / backgrounds eagerly.
    # ------------------------------------------------------------------
    def __init__(
        self,
        calib_dir: str,
        day: str,
        scan_dir: str,
        analysis_dir: str,
        bg_scan_num: int,
        angle_resolution: int = 128,
        momentum_resolutions: Optional[Dict[str, int]] = None,
        acceptance_mm: Optional[Dict[str, float]] = None,
        angle_range: Tuple[float, float] = _DEFAULT_ANGLE_RANGE,
        lowpass_params: Tuple[float, int, float, int] = _DEFAULT_LOWPASS,
        hall_probe_column: str = 'HALLPROBE-TEA-MAGSPEC Field',
        output_diagnostic: str = 'MagSpec',
    ):
        # ---- calibration files -------------------------------------------
        self.cameras: List[CameraCalibration] = self._load_cameras(calib_dir, day)
        self.front_traj, self.side_traj = self._load_trajectory(calib_dir, day)
        self.lanex_table: Dict[int, LanexCalibration] = self._load_lanex(calib_dir, day)

        # ---- per-camera c2c + vignette -----------------------------------
        self.c2c, self.vignettes = self._compute_lanex_corrections()

        # ---- axes + angle maps + uniform windows -------------------------
        self.x_axes, self.y_axes = compute_all_axes(
            self.cameras, self.front_traj, self.side_traj,
            acceptance_mm or _DEFAULT_ACCEPTANCE_MM,
        )
        self.angle_maps: List[AngleMap] = compute_angle_maps(self.x_axes, self.y_axes)
        self.uniform_angle: UniformAngleAxis = uniform_angle_axis(
            angle_resolution, angle_range,
        )
        self.windows: Dict[str, UniformMomentumWindow] = uniform_momentum_axes(
            self.cameras, self.x_axes,
            momentum_resolutions or _DEFAULT_MMT_RES,
        )

        # ---- pre-load (or build) backgrounds for this day ---------------
        self.backgrounds: Dict[str, np.ndarray] = load_or_build_background(
            self.cameras, scan_dir, analysis_dir, bg_scan_num, save=True,
        )

        # ---- wire MultiDiagnosticAnalyzer contract -----------------------
        super().__init__(
            inputs=[(cam.diagnostic, '.png') for cam in self.cameras],
            sub_analyzers={},   # we override load_data below
            output_diagnostic=output_diagnostic,
            output_file_ext='.png',
        )

        # ---- per-shot config retained as state ---------------------------
        self.lowpass_params = lowpass_params
        self.hall_probe_column = hall_probe_column

    # ------------------------------------------------------------------
    # __init__ helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_cameras(calib_dir: str, day: str) -> List[CameraCalibration]:
        path, _ = discover_calib_path(calib_dir, '*camCalib.txt', day)
        return load_camera_calibration(path)

    @staticmethod
    def _load_trajectory(
        calib_dir: str, day: str,
    ) -> Tuple[TrajectoryCalibration, TrajectoryCalibration]:
        path, _ = discover_calib_path(calib_dir, '*trjCalib*A0.txt', day)
        return load_trajectory_calibration(path)

    @staticmethod
    def _load_lanex(calib_dir: str, day: str) -> Dict[int, LanexCalibration]:
        path, _ = discover_calib_path(calib_dir, '*lanexCalib.txt', day)
        return load_lanex_calibration_table(path)

    def _compute_lanex_corrections(
        self,
    ) -> Tuple[List[float], List[np.ndarray]]:
        c2c: List[float] = []
        vignettes: List[np.ndarray] = []
        for cam in self.cameras:
            if cam.set_n not in self.lanex_table:
                raise KeyError(
                    f"No lanex calibration for set_n={cam.set_n} "
                    f"(camera {cam.diagnostic!r})."
                )
            factor, vgnt = compute_c2c_and_vignette(cam, self.lanex_table[cam.set_n])
            c2c.append(factor)
            vignettes.append(vgnt)
        return c2c, vignettes

    # ------------------------------------------------------------------
    # Per-shot pipeline
    # ------------------------------------------------------------------
    def load_data(self, paths: Dict[str, str]) -> Dict[str, Optional[np.ndarray]]:
        """
        Load raw 12-bit camera images for one shot. Overrides the default
        ``MultiDiagnosticAnalyzer.load_data`` to avoid needing sub-analyzers
        — we just want the raw arrays; processing happens in ``analyze_data``.
        """
        out: Dict[str, Optional[np.ndarray]] = {}
        for name, path in paths.items():
            if path is None or not os.path.exists(path):
                out[name] = None
                continue
            arr = read_imaq_image(path)
            out[name] = arr.astype(np.float64) if arr is not None else None
        return out

    def analyze_data(
        self,
        data: Dict[str, Optional[np.ndarray]],
        bg=None,
        context: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict, Dict]:
        """
        Per-shot pipeline:
          1) Process each camera image via ``process_camera_image``.
          2) Stitch per screen into ``{screen: 2D image [aC]}``.
          3) Compute per-screen scalar stats with the Hall-probe field
             from ``context``.

        ``bg`` is ignored — backgrounds are pre-loaded at construction.
        """
        if data is None:
            return {}, {}, {}

        processed, saturations = self._process_all_cameras(data)
        stitched = stitch_all_windows(
            self.cameras, processed, self.angle_maps, self.x_axes,
            self.uniform_angle, self.windows,
        )

        field_T = self._read_field(context)
        results = self._build_results(saturations, stitched, field_T)
        aux = {'field_T': field_T}
        return stitched, results, aux

    def _process_all_cameras(
        self,
        data: Dict[str, Optional[np.ndarray]],
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Run ``process_camera_image`` on each camera; return parallel lists."""
        processed: List[np.ndarray] = []
        saturations: List[int] = []
        for i, cam in enumerate(self.cameras):
            raw = data.get(cam.diagnostic)
            bg_img = self.backgrounds.get(cam.diagnostic)
            if bg_img is None:
                raise KeyError(
                    f"No background loaded for camera {cam.diagnostic!r}."
                )
            proc, sat = process_camera_image(
                raw, cam, bg_img,
                self.lowpass_params,
                self.c2c[i],
                self.vignettes[i],
                self.x_axes[i].inc_angle,
            )
            processed.append(proc)
            saturations.append(sat)
        return processed, saturations

    def _read_field(self, context: Optional[Dict]) -> float:
        """Pick the per-shot Hall-probe field [T] from the scan context."""
        if context is None:
            return 0.0
        return float(context.get(self.hall_probe_column, 0.0) or 0.0)

    def _build_results(
        self,
        saturations: List[int],
        stitched: Dict[str, np.ndarray],
        field_T: float,
    ) -> Dict[str, float]:
        """Per-camera saturation counts + per-screen spot-stat scalars."""
        results: Dict[str, float] = {}
        for cam, sat in zip(self.cameras, saturations):
            results[f'{cam.diagnostic} saturation'] = sat

        if field_T <= 0:
            # Without a field we can't scale the momentum axis; skip stats
            # but still surface the (zero) charge so the sfile carries a row.
            for screen, img in stitched.items():
                results[f'quickE {screen} charge [pC]'] = 1e-6 * float(img.sum())
            return results

        for screen, img in stitched.items():
            window = self.windows[screen]
            energy_gev = 0.001 * field_T * window.mmt
            stats = spot_analysis(energy_gev, self.uniform_angle.angle, img)
            results.update({
                f'quickE {screen} peak mmt [GeV/c]': stats['peak_x'],
                f'quickE {screen} peak angle x [mrad]': stats['peak_y'],
                f'quickE {screen} mean mmt [GeV/c]': stats['mean_x'],
                f'quickE {screen} mean angle x [mrad]': stats['mean_y'],
                f'quickE {screen} fwhm mmt [GeV/c]': stats['fwhm_x'],
                f'quickE {screen} fwhm div x [mrad]': stats['fwhm_y'],
                f'quickE {screen} std mmt [GeV/c]': stats['std_x'],
                f'quickE {screen} std div x [mrad]': stats['std_y'],
                f'quickE {screen} charge [pC]': 1e-6 * float(img.sum()),
            })
        return results

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    def write_analyzed_data(
        self,
        data: Dict[str, np.ndarray],
        analysis_dir: str,
        scan: int,
        shot_num: int,
        context: Optional[Dict] = None,
    ) -> None:
        """
        Write the per-shot quickE outputs (one set per screen window).

        Files (matches matlab ``bellaLiveMagspc2.m`` ``quickDir`` outputs):
          - ``Scan{NNN}_quickESpec[_<screen>]_<shot>.txt``
          - ``Scan{NNN}_quickEDiv[_<screen>]_<shot>.txt``
          - ``Scan{NNN}_quickE[_<screen>]_<shot>.png``  (16-bit aC PNG with comment)

        The ``_<screen>`` suffix is only inserted when more than one screen
        is being processed; the single-screen case matches matlab byte-for-byte.
        """
        if not data:
            return
        field_T = self._read_field(context)
        if field_T <= 0:
            return  # field unavailable → energy-axis ill-defined; skip output

        quick_dir = os.path.join(analysis_dir, 'quickE')
        os.makedirs(quick_dir, exist_ok=True)

        multi_screen = len(data) > 1
        for screen, img in data.items():
            paths = self._output_paths(quick_dir, scan, shot_num, screen, multi_screen)
            write_energy_spectrum_txt(paths['spec'], self.windows[screen], img, field_T)
            write_angle_distribution_txt(paths['div'], self.uniform_angle, img)
            write_integer_aC_png(paths['png'], img)

    @staticmethod
    def _output_paths(
        quick_dir: str,
        scan: int,
        shot_num: int,
        screen: str,
        multi_screen: bool,
    ) -> Dict[str, str]:
        suffix = f"_{screen}" if multi_screen else ""
        prefix = f"Scan{int(scan):03d}_quickE"
        shot = f"{int(shot_num):03d}"
        return {
            'spec': os.path.join(quick_dir, f"{prefix}Spec{suffix}_{shot}.txt"),
            'div':  os.path.join(quick_dir, f"{prefix}Div{suffix}_{shot}.txt"),
            'png':  os.path.join(quick_dir, f"{prefix}{suffix}_{shot}.png"),
        }

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def display_data(
        self,
        data: Dict[str, np.ndarray],
        display_dict: Optional[Dict] = None,
        return_dict: Optional[Dict] = None,
        title: Optional[str] = None,
        fig=None,
        ax=None,
    ):
        """
        One pcolormesh panel per screen, with the energy axis scaled by
        ``return_dict['field_T']`` (forwarded from ``analyze_data``).

        ``display_dict`` accepts ``figsize``, ``cmap``, ``vmin``, ``vmax``.
        """
        if not data:
            return None, None
        display_dict = display_dict or {}

        n = len(data)
        if fig is None:
            figsize = display_dict.get('figsize', (6 * n, 4))
            fig, axes = plt.subplots(
                1, n,
                figsize=figsize,
                constrained_layout=True,
                squeeze=False,
            )
        else:
            axes = np.atleast_2d(ax) if ax is not None else np.atleast_2d(fig.axes)

        field_T = float((return_dict or {}).get('field_T', 1.0)) or 1.0
        cmap = display_dict.get('cmap', 'jet')
        vmin = display_dict.get('vmin', None)
        vmax = display_dict.get('vmax', None)

        for k, (screen, img) in enumerate(data.items()):
            a = axes.flat[k]
            window = self.windows[screen]
            energy_gev = 0.001 * field_T * window.mmt
            # pC / mrad / (GeV/c) density: aC / (MeV/c × mrad) × 1e-3
            density = img / (window.dp * self.uniform_angle.d_angle) * 1e-3
            im = a.pcolormesh(
                energy_gev, self.uniform_angle.angle, density,
                shading='auto', cmap=cmap, vmin=vmin, vmax=vmax,
            )
            fig.colorbar(im, ax=a, label='pC / mrad / (GeV/c)')
            a.set_xlabel('Energy [GeV/c]')
            a.set_ylabel('Angle [mrad]')
            a.set_title(f'{screen} window')

        if title:
            fig.suptitle(title)
        return fig, axes
