"""Reference pixel corrections for HxRG detectors.

The correction is split in an up-down part (per-channel offset estimated
from the top and bottom reference rows) and a left-right part (per-row
offset estimated from the left and right reference columns).
"""
import numpy as np
from scipy.signal import savgol_filter

def get_refpix_ud(ud_refpix, nframes, nchannel, npix_channel, npix_ref=4):
    """Estimate the up-down reference pixel correction.

    The top and bottom reference rows of each frame are concatenated and
    averaged (mean) per readout channel.

    Parameters
    ----------
    ud_refpix : ndarray
        Up and down reference rows, of shape
        ``(nframes, 2 * npix_ref, nx)``.
    nframes : int
        Number of frames (groups) in the ramp.
    nchannel : int
        Number of readout channels.
    npix_channel : int
        Number of pixel columns per channel.
    npix_ref : int, optional
        Number of reference rows on each side of the frame. Default is 4.

    Returns
    -------
    ndarray
        Up-down correction of shape ``(nframes, nx)``, constant within each
        channel, as float32.
    """
    npix_ref_both_side = npix_ref * 2
    ud_corr = np.mean(ud_refpix.reshape(nframes, npix_ref_both_side, nchannel, npix_channel),
                    axis=(1,3))
    return np.repeat(ud_corr, npix_channel, axis=1).astype(np.float32)


def get_refpix_lr(lr_refpix, npix_channel, polyorder=3, red_fac=0.875, npix_ref=4):
    """Estimate the left-right reference pixel correction.

    The left and right reference columns of the active rows are averaged
    per row, and the resulting profile is smoothed with a Savitzky-Golay
    filter.

    Parameters
    ----------
    lr_refpix : ndarray
        Up-down corrected frames of shape ``(nframes, ny, nx)``.
    npix_channel : int
        Number of pixel columns per channel. Sets the smoothing window
        length.
    polyorder : int, optional
        Polynomial order of the Savitzky-Golay filter. Default is 3.
    red_fac : float, optional
        Ratio between the smoothing window length and ``npix_channel``
        (capped at 128 pixels). If 1, the window is as long as a channel.
        Default is 0.875.
    npix_ref : int, optional
        Number of reference columns (and rows) on each side of the frame.
        Default is 4.

    Returns
    -------
    ndarray
        Left-right correction of shape ``(nframes, ny - 2 * npix_ref)``,
        as float32.

    See Also
    --------
    scipy.signal.savgol_filter
    """
    lr_corr = np.concatenate([lr_refpix[:, npix_ref:-npix_ref, :npix_ref],
                              lr_refpix[:, npix_ref:-npix_ref, -npix_ref:]],
                              axis=2 ).mean(axis=2)
    if npix_channel > 128:
        warnings.warn(f"npix_channel={npix_channel} > 128, capping to 128 for smoothing window length.")
        npix_channel = 128

    window_length = int(red_fac*npix_channel)
    return np.asarray( [ savgol_filter(lr_corr_, window_length=window_length, polyorder=polyorder).astype("float32") for lr_corr_ in lr_corr])
