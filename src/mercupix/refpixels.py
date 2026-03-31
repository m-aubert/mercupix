import numpy as np
from scipy.signal import savgol_filter

def get_refpix_ud(ud_refpix, nframes, nchannel, npix_channel, npix_ref=4):
    """
    Method to estimate the Up-Down reference pixel correction.
    Concatenates up and down pixel of each columns and average (mean) them on a channel basis.

    Parameters
    ----------
    ud_refpix : ndarray
        Up-Down reference pixels
    nframes : int
        Number of frames/groups in the ramp.
    nchannel : int
        Number of channel.
    npix_channel : int
        Number of pixels in each channel.
    npix_ref : int, optional
         Number of reference pixels, by default 8

    Returns
    -------
    ndarray
        Up-down reference pixel correction.
    """
    npix_ref_both_side = npix_ref * 2
    ud_corr = np.mean(ud_refpix.reshape(nframes, npix_ref_both_side, nchannel, npix_channel),
                    axis=(1,3)) 
    return np.repeat(ud_corr, npix_channel, axis=1).astype(np.float32) 
    
    
def get_refpix_lr(lr_refpix, npix_channel, polyorder=3, red_fac=0.875, npix_ref=4):
    """
     Method to estimate the Left-Right reference pixel correction.

    Parameters
    ----------
    lr_refpix : ndarray
        up-down corrected left-right reference pixels
    npix_channel : int
        number of channels in the detector 
    polyorder : int, optional
        polynomial order of smoothing, by default 3
    red_fac : float, optional
        Factor to control the pixel window.
        if 1, the pixel window is the same number as the number of pixels in a single channel,
        by default 0.875

    Returns
    -------
    ndarray
         Left-Right reference pixel correction
    """    
    lr_corr = np.concatenate([lr_refpix[:, npix_ref:-npix_ref, :npix_ref],
                              lr_refpix[:, npix_ref:-npix_ref, -npix_ref:]],
                              axis=2 ).mean(axis=2)
    
    window_length = int(red_fac*npix_channel)
    if window_length > 128 : 
        window_length = int(red_fac*128)
        
    return np.asarray( list( map(lambda x : savgol_filter(x,
                                                        window_length=window_length, 
                                                        polyorder=polyorder).astype(np.float32), lr_corr)
                            ))
