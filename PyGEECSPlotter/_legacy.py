"""Old flat module paths -> their new homes (pre-0.1 layout).

Importing an old path, e.g. ``PyGEECSPlotter.image_analysis``, returns the new
module object itself (so classes, ``isinstance`` and pickles are unchanged)
and emits one ``DeprecationWarning``. Update imports to the new paths; these
aliases will be removed in a future release.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

MOVED_MODULES = {
    'PyGEECSPlotter.scan_data_analysis': 'PyGEECSPlotter.scan_data_analyzers.scan_data_analysis',
    'PyGEECSPlotter.parallel_scan_data_analyzer': 'PyGEECSPlotter.scan_data_analyzers.parallel_scan_data_analyzer',
    'PyGEECSPlotter.data_dir_analysis': 'PyGEECSPlotter.scan_data_analyzers.data_dir_analysis',
    'PyGEECSPlotter.live_analysis': 'PyGEECSPlotter.scan_data_analyzers.live_analysis',
    'PyGEECSPlotter.binning': 'PyGEECSPlotter.scan_data_analyzers.binning',
    'PyGEECSPlotter.diagnostic_analyzer': 'PyGEECSPlotter.analyzers.diagnostic_analyzer',
    'PyGEECSPlotter.multi_diagnostic_analyzer': 'PyGEECSPlotter.analyzers.multi_diagnostic_analyzer',
    'PyGEECSPlotter.data_reader': 'PyGEECSPlotter.analyzers.data_reader',
    'PyGEECSPlotter.column_math_analysis': 'PyGEECSPlotter.analyzers.column_math_analysis',
    'PyGEECSPlotter.image_analysis': 'PyGEECSPlotter.analyzers.image_analysis',
    'PyGEECSPlotter.shot_compare_analysis': 'PyGEECSPlotter.analyzers.shot_compare_analysis',
    'PyGEECSPlotter.spectrum_analysis': 'PyGEECSPlotter.analyzers.spectrum_analysis',
    'PyGEECSPlotter.combined_vis_nir_spectra': 'PyGEECSPlotter.analyzers.combined_vis_nir_spectra',
    'PyGEECSPlotter.frog_analysis': 'PyGEECSPlotter.analyzers.frog_analysis',
    'PyGEECSPlotter.wavefront_analysis': 'PyGEECSPlotter.analyzers.wavefront_analysis',
    'PyGEECSPlotter.camera_energy_analysis': 'PyGEECSPlotter.analyzers.camera_energy_analysis',
    'PyGEECSPlotter.alignment_analysis': 'PyGEECSPlotter.analyzers.alignment_analysis',
    'PyGEECSPlotter.magspec_alle_analysis': 'PyGEECSPlotter.analyzers.magspec.magspec_alle_analysis',
    'PyGEECSPlotter.magspec_alle_reader': 'PyGEECSPlotter.analyzers.magspec.magspec_alle_reader',
    'PyGEECSPlotter.ebeam_profile_analysis': 'PyGEECSPlotter.analyzers.magspec.ebeam_profile_analysis',
    'PyGEECSPlotter.fitting': 'PyGEECSPlotter.utils.fitting',
    'PyGEECSPlotter.navigation_utils': 'PyGEECSPlotter.utils.navigation_utils',
    'PyGEECSPlotter.sfile_utils': 'PyGEECSPlotter.utils.sfile_utils',
    'PyGEECSPlotter.ecs_live_dumps': 'PyGEECSPlotter.utils.ecs_live_dumps',
    'PyGEECSPlotter.ni_imread': 'PyGEECSPlotter.utils.ni_imread',
    'PyGEECSPlotter.plotting': 'PyGEECSPlotter.utils.plotting',
    'PyGEECSPlotter.geecs_cmaps': 'PyGEECSPlotter.utils.geecs_cmaps',
    'PyGEECSPlotter.pix_cmaps': 'PyGEECSPlotter.utils.pix_cmaps',
    'PyGEECSPlotter.widgets': 'PyGEECSPlotter.utils.widgets',
    'PyGEECSPlotter.displayers.scalar_vs_parameter': 'PyGEECSPlotter.displayers.scalars.scalar_vs_parameter',
    'PyGEECSPlotter.displayers.correlation_heatmap': 'PyGEECSPlotter.displayers.scalars.correlation_heatmap',
    'PyGEECSPlotter.displayers.scan_stats_table': 'PyGEECSPlotter.displayers.scalars.scan_stats_table',
    'PyGEECSPlotter.displayers.lineout_waterfall': 'PyGEECSPlotter.displayers.lineouts.lineout_waterfall',
    'PyGEECSPlotter.displayers.lineout_mean_per_bin': 'PyGEECSPlotter.displayers.lineouts.lineout_mean_per_bin',
    'PyGEECSPlotter.displayers.lineout_mean_waterfall': 'PyGEECSPlotter.displayers.lineouts.lineout_mean_waterfall',
    'PyGEECSPlotter.displayers.trace_waterfall': 'PyGEECSPlotter.displayers.lineouts.trace_waterfall',
    'PyGEECSPlotter.displayers.trace_mean_per_bin': 'PyGEECSPlotter.displayers.lineouts.trace_mean_per_bin',
    'PyGEECSPlotter.displayers.trace_mean_waterfall': 'PyGEECSPlotter.displayers.lineouts.trace_mean_waterfall',
    'PyGEECSPlotter.displayers.optical_spectrum': 'PyGEECSPlotter.displayers.lineouts.optical_spectrum',
    'PyGEECSPlotter.displayers.image_grid': 'PyGEECSPlotter.displayers.images.image_grid',
    'PyGEECSPlotter.displayers.shot_selection_grid': 'PyGEECSPlotter.displayers.images.shot_selection_grid',
    'PyGEECSPlotter.displayers.mean_image_per_bin': 'PyGEECSPlotter.displayers.images.mean_image_per_bin',
    'PyGEECSPlotter.displayers.sampled_images': 'PyGEECSPlotter.displayers.images.sampled_images',
    'PyGEECSPlotter.displayers.representative_image_per_bin': 'PyGEECSPlotter.displayers.images.representative_image_per_bin',
    'PyGEECSPlotter.displayers.peak_width_images': 'PyGEECSPlotter.displayers.images.peak_width_images',
    'PyGEECSPlotter.displayers.shot_compare': 'PyGEECSPlotter.displayers.hybrid.shot_compare',
    'PyGEECSPlotter.displayers.multi_diagnostic_alignment': 'PyGEECSPlotter.displayers.hybrid.multi_diagnostic_alignment',
    'PyGEECSPlotter.displayers.focal_scan': 'PyGEECSPlotter.displayers.hybrid.focal_scan',
    'PyGEECSPlotter.displayers.multi_scan_scalar_vs_parameter': 'PyGEECSPlotter.displayers.hybrid.multi_scan_scalar_vs_parameter',
    'PyGEECSPlotter.displayers.multi_scan_lineout_mean': 'PyGEECSPlotter.displayers.hybrid.multi_scan_lineout_mean',
    'PyGEECSPlotter.displayers.magspec_alle_waterfall': 'PyGEECSPlotter.displayers.magspec.magspec_alle_waterfall',
    'PyGEECSPlotter.displayers.magspec_alle_spectrum_per_bin': 'PyGEECSPlotter.displayers.magspec.magspec_alle_spectrum_per_bin',
    'PyGEECSPlotter.displayers.magspec_alle_gallery': 'PyGEECSPlotter.displayers.magspec.magspec_alle_gallery',
}



class _AliasLoader(importlib.abc.Loader):
    def __init__(self, new_name):
        self.new_name = new_name
        self._new_spec = None

    def create_module(self, spec):
        module = importlib.import_module(self.new_name)
        self._new_spec = module.__spec__
        return module

    def exec_module(self, module):
        # the import system stamped the alias spec on the shared module; restore it
        module.__spec__ = self._new_spec
        warnings.warn(f"This module has moved to '{self.new_name}'; importing it from "
                      f"its old path is deprecated.", DeprecationWarning, stacklevel=2)


class LegacyModuleFinder(importlib.abc.MetaPathFinder):
    """Resolve the old flat module paths listed in ``MOVED_MODULES``."""

    def find_spec(self, fullname, path=None, target=None):
        new_name = MOVED_MODULES.get(fullname)
        if new_name is None:
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(new_name))


def install():
    if not any(isinstance(f, LegacyModuleFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, LegacyModuleFinder())
