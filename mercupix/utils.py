""" This file should contain utilitary function """
import numpy as np
from scipy.signal import savgol_filter

def refpix_ud(ud_refpix,nframes, nchannel, npix_channel, npix_ref=8):    
    """ Method to estimate the Up-Down reference pixel correction.

    Args:
        ud_refpix (ndarray): Up-Down reference pixels
        nframes (_type_): Number of frames in the ramp.
        nchannel (_type_): Number of channel.
        npix_channel (_type_): Number of pixels in each channel.
        npix_ref (int, optional): Number of reference pixels. Defaults to 8.

    Returns:
        ndarray : Up-down reference pixel correction.
    """
    ud_corr = np.mean(ud_refpix.reshape(nframes, npix_ref, npix_channel, nchannel),
                    axis=(1,3)) 
    ud_corr = np.repeat(ud_corr, nchannel, axis=1).astype(np.float32) 
    
    return ud_corr
    
def refpix_lr(lr_refpix, npix_channel, polyorder=3, red_fac=0.875):    
    """ Method to estimate the Left-Right reference pixel correction.

    Args:
        lr_refpix (_type_): ud corrected left-right reference pixels
        npix_channel (_type_): number of channels in the detector 
        polyorder (int, optional): polynomial order of smoothing. Defaults to 3.
        red_fac (float, optional): Factor to control the pixel window. 
                                    If one, uses the same number as the number of pixel in a channel. 
                                    Defaults to 0.875.

    Returns:
        ndarray : Left-Right reference pixel correction
    """

    lr_corr = np.concatenate([lr_refpix[:,4:-4,:4], 
                         lr_refpix[:,4:-4,-4:]], 
                            axis=2).mean(axis=2)
    
    lr_corr = np.asarray(list(map(lambda x : savgol_filter(x,
                                                        window_length=int(red_fac*npix_channel), 
                                                        polyorder=polyorder).astype(np.float32), lr_corr)))

    return lr_corr