# The MATLAB "infoE" per-shot summary figure (fig1 of fBellaSShotTri.m):
# e-beam profile | allE charge density + spectrum | front-screen x-ray,
# with two text panels. Pure plotting: every input is passed in.

import numpy as np
import matplotlib.pyplot as plt

# EBeam-profile overlay drawn by fBellaSShotTri (screen outline, hole rows)
_SCREEN_BOX = ([-3.3, 2.65, 2.65, -3.3, -3.3], [-1.2, -1.2, 1.2, 1.2, -1.2])
_HOLE_ROWS = 0.75


def _acceptance_mask(ax, mmt, accp, ya):
    """Black patches outside +/- acceptance, as MATLAB's two ``patch`` calls."""
    x = np.r_[mmt[0], mmt, mmt[-1], mmt[0]]
    ax.fill(x, np.r_[ya[-1], accp, ya[-1], ya[-1]], color='k', lw=0)
    ax.fill(x, np.r_[ya[0], -accp, ya[0], ya[0]], color='k', lw=0)


def _text_panel(ax, lines, fontsize):
    """``fPltInfV02``: lines spread evenly top to bottom, axes hidden."""
    ax.set_axis_off()
    n = len(lines)
    for k, s in enumerate(lines):
        ax.text(0, 1 - k / max(n - 1, 1), s, fontsize=fontsize, va='center',
                transform=ax.transAxes)


def draw_infoe(mmt, ya, density, accp, spectrum, gap, scalars, xray_img, xray_x_mm,
               xray_y_mm, ebeam=None, ict_pC=np.nan, ey_angle=np.nan, scan=None, shot=None,
               roi=(0.01, 5.0), fontsize=10, figsize=(20, 6.67)):
    """
    Draw infoE and return ``(fig, axes_dict)``.

    Parameters
    ----------
    mmt, ya : momentum [GeV/c] (ROI, non-uniform) and angle [mrad] axes.
    density : charge density [pC/mrad/(GeV/c)] on (ya, mmt).
    accp : acceptance half-angle [mrad] on mmt.
    spectrum : charge density [pC/GeV] on mmt.
    gap : (low, high) momentum [GeV/c] of the screen gap, drawn black.
    scalars : the stage-2 scalar dict (charge_pC, peakMomentum_GeV/c, ...).
    xray_img : front-screen image [fC] (flipped up-down, as saved);
        ``xray_x_mm`` / ``xray_y_mm`` its axes.
    ebeam : optional dict ``{'image': pC per pixel, 'x': mrad, 'y': mrad}``
        for the left panel; None leaves it blank ("no EBeam profile").
    ict_pC, ey_angle : shown in the left text panel.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 9, left=0.04, right=0.97, top=0.9, bottom=0.08,
                          wspace=0.35, hspace=0.35)
    ax_eb = fig.add_subplot(gs[0:2, 0:2])
    ax_im = fig.add_subplot(gs[0:2, 2:7])
    ax_xr = fig.add_subplot(gs[0:2, 7:9])
    ax_txt_l = fig.add_subplot(gs[2, 0:2])
    ax_sp = fig.add_subplot(gs[2, 2:7], sharex=ax_im)
    ax_txt_r = fig.add_subplot(gs[2, 7:9])
    max_eng = scalars.get('maxMomentum_GeV/c', 0.0)

    # --- centre: allE charge density; colour max = peak density at the
    # energy-peak column (peakCD in fBellaSShotTri; linScl was [0 20] in the
    # repo copy but the reference figures use this automatic scale)
    ipe = int(np.nanargmin(np.abs(mmt - scalars.get('energyPeakMmt_GeV/c', mmt[0]))))
    peak_cd = np.nanmax(density[:, ipe]) or 0.1
    m = ax_im.pcolormesh(mmt, ya, density, cmap='jet', shading='auto', vmin=0, vmax=peak_cd,
                         rasterized=True)
    _acceptance_mask(ax_im, mmt, accp, ya)
    ax_im.fill([gap[0], gap[1], gap[1], gap[0]], [ya[-1], ya[-1], ya[0], ya[0]], color='k', lw=0)
    ax_im.axvline(max_eng, color='g', lw=1)
    xc = 0.5 * (mmt[0] + mmt[-1])
    ax_im.text(xc, -1, 'AA', color='w', fontsize=fontsize)
    ax_im.text(xc, 1, f'Max mmt = {max_eng:.3g} GeV/c', color='w', fontsize=fontsize)
    ax_im.set_ylim(-1.3, 1.3)
    ax_im.tick_params(labelbottom=True)
    fig.colorbar(m, ax=ax_im, location='top', fraction=0.08, pad=0.02)

    # --- bottom centre: log spectrum
    with np.errstate(divide='ignore', invalid='ignore'):
        ax_sp.plot(mmt, np.log10(spectrum), 'r-', lw=0.8)
    ax_sp.axvline(max_eng, color='g', lw=1)
    top = np.nanmax(np.log10(spectrum[spectrum > 0])) if np.any(spectrum > 0) else 1
    ax_sp.set_xlim(roi)
    ax_sp.set_ylim(-1, max(top, 1))
    ax_sp.set_xlabel('GeV/c')
    ax_sp.set_ylabel('log(pC/GeV)')

    # --- left: e-beam profile [pC/mrad^2], axes swapped as in MATLAB
    # (pcolor(lnxY, lnxX, img')), plus the screen outline and hole rows
    if ebeam is not None:
        ex, ey = np.asarray(ebeam['x']), np.asarray(ebeam['y'])
        dmrad = ex[1] - ex[0]
        mb = ax_eb.pcolormesh(ey, ex, np.asarray(ebeam['image']).T / dmrad ** 2, cmap='jet',
                              shading='auto', vmin=0, rasterized=True)
        fig.colorbar(mb, ax=ax_eb, location='top', fraction=0.08, pad=0.02)
        ax_eb.plot([ey_angle, ey_angle] if np.isfinite(ey_angle) else [0, 0], [-2, 2], 'w--', lw=1)
        ax_eb.plot(*_SCREEN_BOX, 'w-.', lw=1)
        for s in (-1, 1):
            ax_eb.plot([-3.3, 2.65], [s * _HOLE_ROWS] * 2, 'w-.', lw=1)
        ax_eb.text(0.98, 0.02, 'up, AA', color='w', ha='right', transform=ax_eb.transAxes,
                   fontsize=fontsize - 1)
        eb_img = np.asarray(ebeam['image'])
        xm, ym = np.meshgrid(ex, ey)   # lnxXM from meshgrid(lnxX, lnxY)
        hole = (xm / 1.2) ** 2 + (ym / 1.2) ** 2 <= 1
        hole_chg, all_chg = np.nansum(eb_img * hole), np.nansum(eb_img)
    else:
        ax_eb.set_facecolor('0.85')
        ax_eb.set_xticks([])
        ax_eb.set_yticks([])
        ax_eb.text(0.5, 0.5, 'no EBeam profile', ha='center', va='center',
                   transform=ax_eb.transAxes, fontsize=fontsize)
        hole_chg = all_chg = np.nan

    # --- right: front-screen x-ray image, rot90(fliplr(img), 2) on -x
    xr = np.fliplr(np.asarray(xray_img))
    # MATLAB plots the saved frontSL PNG counts (aC per pixel)
    mx = ax_xr.pcolormesh(-np.asarray(xray_x_mm), xray_y_mm, 1e3 * np.rot90(xr, 2), cmap='jet',
                          shading='auto', rasterized=True)
    ax_xr.yaxis.tick_right()
    fig.colorbar(mx, ax=ax_xr, location='top', fraction=0.08, pad=0.02)
    ax_xr.axvline(0, color='w', ls='-.', lw=1)
    ax_xr.text(0.98, 0.02, 'up, AA', color='w', ha='right', transform=ax_xr.transAxes,
               fontsize=fontsize - 1)

    # --- text panels (fPltInfV02)
    s3 = lambda v: f'{v:.3g}' if np.isfinite(v) else 'n/a'
    _text_panel(ax_txt_l, [
        f'scan: {int(scan):03d}' if scan is not None else 'scan:',
        f'shot: {int(shot):03d}' if shot is not None else 'shot:',
        f'EBeam hole charge [pC]: {s3(hole_chg)}',
        f'EBeam all charge [pC]: {s3(all_chg)}',
        f'ICT charge [pC]: {s3(ict_pC)}',
        f'e-beam y angle [mrad]: {s3(ey_angle)}',
    ], fontsize - 2)
    chg = scalars.get('charge_pC', np.nan)
    peak = scalars.get('peakMomentum_GeV/c', np.nan)
    fwhm = scalars.get('fwhmMomentum_GeV/c', np.nan)
    r = lambda v, n: round(v, n) if np.isfinite(v) else np.nan
    _text_panel(ax_txt_r, [
        f'MS charge [pC]: {r(chg, 3):g}',
        f'peakMmt[GeV/c]: {r(peak, 3):g}',
        f'fwhmMmt[GeV/c]: {r(fwhm, 3):g}',
        f'mmtSpread[%]: {r(100 * fwhm / peak, 1):g}' if peak else 'mmtSpread[%]: n/a',
        f"mmtRes[%]: {r(scalars.get('mmtRes_%', np.nan), 2):g}",
        f"peakAngl[mrad]: {r(scalars.get('peakAngle_mrad', np.nan), 1):g}",
        f"fwhmAngl[mrad]: {r(scalars.get('fwhmAngle_mrad', np.nan), 1):g}",
        f'XRay eqv charge [pC]: {1e-3 * np.nansum(xray_img):.2g}',
    ], fontsize - 2)

    axes = {'ebeam': ax_eb, 'alle': ax_im, 'xray': ax_xr, 'spectrum': ax_sp,
            'info_left': ax_txt_l, 'info_right': ax_txt_r}
    return fig, axes
