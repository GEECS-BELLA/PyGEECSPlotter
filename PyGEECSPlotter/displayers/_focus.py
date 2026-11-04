"""Per-shot laser-focus figures of merit shared by ``FocalScanDisplayer``
(detailed view) and ``FocusStatsDisplayer``: the pulse energy on target from
the sfile, and the unthresholded image whose counts hold that energy."""

import copy
import warnings

import numpy as np

import PyGEECSPlotter.utils.analysis_2d_utils as a2d

# 1BL compressor transmission before scans recorded it in the sfile
DEFAULT_TRANSMISSION = 0.586


def shot_energy(row, energy_col, transmission_col, energy_scale=None):
    """(``energy_col``, transmission) for one sfile row (Series or dict).
    The transmission is ``energy_scale`` if given, else ``transmission_col``,
    else ``DEFAULT_TRANSMISSION`` with a warning."""
    get = row.get
    raw = float(get(energy_col, np.nan))
    if not np.isfinite(raw):
        warnings.warn(f"No pulse energy in '{energy_col}'; fluence, intensity, a0 and "
                      f"Strehl are NaN.")
    if energy_scale is not None:
        return raw, float(energy_scale)
    t = float(get(transmission_col, np.nan))
    if not np.isfinite(t):
        warnings.warn(f"No '{transmission_col}' in the sfile; using {DEFAULT_TRANSMISSION:g}.")
        t = DEFAULT_TRANSMISSION
    return raw, t


def norm_analyzer(analyzer, norm_pipeline=None):
    """Copy of ``analyzer`` running ``norm_pipeline`` (default: its own
    pipeline without ``'threshold'`` steps). Its image's counts hold the
    pulse energy, so halo energy a threshold removes isn't credited to the
    peak."""
    ad = analyzer.analyzer_dict or {}
    if norm_pipeline is None:
        norm_pipeline = [(name, kw) for name, kw in analyzer._resolve_pipeline(ad)
                         if name != 'threshold']
    out = copy.copy(analyzer)
    out.analyzer_dict = {**ad, 'pipeline': norm_pipeline}
    return out


def shot_focus_stats(analyzer, data, results, norm_data, energy, tau_fs, wavelength_um):
    """``a2d.focal_spot_stats`` for one shot plus its 2nd moments
    ('2nd moment x' ...), taken from the pipeline's ``results`` when it
    computed them (its r_max etc.), else from ``data``."""
    ad = analyzer.analyzer_dict or {}
    dx, dy = ad.get('dx', 1), ad.get('dy', 1)
    stats = a2d.focal_spot_stats(data, dx=dx, dy=dy, energy=energy, tau_fs=tau_fs,
                                 wavelength_um=wavelength_um, norm_data=norm_data)
    results = results or {}
    keys = ('x', 'y', 'x45', 'y45')
    if not all(f'2nd moment {k}' in results for k in keys):
        m = a2d.compute_second_moments(data, dx=dx, dy=dy)
        results = {f'2nd moment {k}': m[k] for k in keys}
    stats.update({f'2nd moment {k}': results[f'2nd moment {k}'] for k in keys})
    return stats


def finish_ee_axes(a, analyzer, r_max, w0, ee_rmax=None):
    """Encircled-energy axes: a Gaussian of 1/e^2 radius ``w0`` with a
    dotted line at ``w0``, x range 0 to ``ee_rmax`` (default 10 w0, at most
    ``r_max``), labels, grid and legend."""
    if np.isfinite(w0):
        rr = np.linspace(0, r_max, 400)
        a.plot(rr, 1 - np.exp(-2*rr**2/w0**2), color='C3', ls='--', lw=1,
               label=f'Gaussian, $w_0$ = {w0:.3g}')
        a.axvline(w0, color='0.4', ls=':', lw=1)
    if ee_rmax is None:
        ee_rmax = 10*w0 if np.isfinite(w0) else r_max
    a.set_xlim(0, min(ee_rmax, r_max))
    a.set_ylim(0, 1.02)
    units = (analyzer.display_dict or {}).get('spatial_units', 'pixels')
    a.set_xlabel(rf'radius ($\mathrm{{{units}}}$)')
    a.set_ylabel('encircled energy fraction')
    a.grid(alpha=0.3)
    a.legend(fontsize='small', loc='lower right')


def encircled_energy(analyzer, norm_data, data_shape, stats):
    """(r, fraction) of ``norm_data``, centred on the stats' centre when
    the two images share a frame."""
    ad = analyzer.analyzer_dict or {}
    same = np.shape(norm_data) == tuple(data_shape)
    return a2d.encircled_energy_curve(norm_data, dx=ad.get('dx', 1), dy=ad.get('dy', 1),
                                      x0=stats['x0'] if same else None,
                                      y0=stats['y0'] if same else None)
