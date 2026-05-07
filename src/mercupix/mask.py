import numpy as np 
import warnings
from astropy.stats import mad_std


_MASK_DTYPE = 'uint8' 
_BITFLAGS_DICT = dict(DIS=1,
                      CR=2,
                      SATURATED= 4,
                      HOT=8,
                      WARM=16)    

def get_saturated(data=None, sat_threshold=1):
    """Identify saturated pixel in data.

    Parameters
    ----------
    data : ndarray, optional
        Raw datacube (in uint16)
        
    Returns:
        ndarray : Boolean (N, Npix, Npix) array where True means saturated pixels.

    """
    saturated_pixels = np.zeros(data.shape, dtype=_MASK_DTYPE) 
    mask = data >= int(sat_threshold*65535)
    saturated_pixels[mask] = _BITFLAGS_DICT['SATURATED']
    
    return saturated_pixels

def get_downpix(path=None):
    """Recover pixels markes as dark/disconnected.

    Parameters
    ----------
    path (str, optional) : Path to disconnected pixel mask from metadata.
        Data cube (Npix, Npix). by default None
        
    Returns
    -------
        ndarray : Boolean (N, Npix, Npix) array where True means disconnected.
    """
    from .io import load_downpix
    # path tbd
    if path is not None : 
        mask = load_downpix(path=path)
    else : 
        warnings.warn('No disconnected pixel map provided. \n No DIS Mask.')
        mask = 0
                
    return mask

def get_hotpix(data=None, path=None, **kwargs):
    """ Get identified noisy pixel in data.

    Parameters
    ----------
    data : ndarray, optional
        Data cube (N, Npix, Npix). by default None

    Returns
    -------
    ndarray
        Boolean (Npix, Npix) array where True means noisy pixels.
    """
    # Boolean
    # Get noisy pixels as bitmap or as boolean ?
    
    if path is not None : 
        from .io import load_hotpix
        mask = load_hotpix(path)
    else : 
        warnings.warn("No hot pixel map provided. \n"+ \
                      "No HOT and WARM Mask.")
        mask = 0
         
    return mask

def get_cosmics(data, to_mask=None, cr_threshold_mad=9, cr_threshold_std=5):
    from astropy.stats import sigma_clip 
    
    mask = np.zeros(data.shape, _MASK_DTYPE)
    
    if to_mask is None : 
        zscore = get_zscores(np.diff(data, axis=0),
                             threshold_mad=cr_threshold_mad, 
                             threshold_std=cr_threshold_std)
    
    else : 
        data_ = np.ma.array(np.diff(data, axis=0), mask=to_mask[1:])
        zscore = get_zscores(data_, 
                             threshold_mad=cr_threshold_mad, 
                             threshold_std=cr_threshold_std)
        zscore = zscore & ~to_mask[1:]
    
    # Special treatment for first frame.
    mask_frame0 = sigma_clip(data[0], cenfunc='median', stdfunc='mad_std', 
                             sigma_lower=100000, sigma_upper=cr_threshold_mad)

    mask[np.concatenate([mask_frame0.mask[None,:] , zscore], axis=0)] = _BITFLAGS_DICT['CR']    
    
    return mask 
        
def get_zscores(data, threshold_mad=9, threshold_std=5) :
    np_= np.ma if isinstance(data, np.ma.MaskedArray) else np 
        
    
    z_mad = (data - np_.median(data, axis=0, keepdims=True) )/mad_std(data, axis=0)
    z_std = (data - np_.median(data, axis=0, keepdims=True) )/data.std(axis=0, ddof=1, keepdims=True)
    
    mask = (z_mad >= threshold_mad) & (z_std >= threshold_std)
    return mask
            
def detect_hotpix(dark_slope, warm_threshold=0.15, hot_threshold=0.5):
    """
    Raw hot pixel detection from dark slope.
    
    Thresholds from Roman. Also need linearity tests.

    Parameters
    ----------
    dark_slope : ndarray
        (Nactpix, Nactpix) dark slope
    warm_threshold : float, optional
        Warm pixel detection threshold, by default 0.1
    hot_threshold : float, optional
        Hot pixel detection threshold, by default 0.5

    Returns
    -------
    ndarray
        Bitmask of hot & warm pixel
    """

    mask = np.zeros(dark_slope.shape, _MASK_DTYPE)
    mask[(dark_slope >= warm_threshold) & (dark_slope <hot_threshold )] += _BITFLAGS_DICT['WARM']
    mask[(dark_slope >= hot_threshold)] += _BITFLAGS_DICT['HOT']

    # Missing : 
    # Wacky pixels and linearity requirements
    # Stacked darks data.
    # What else ?
    
    return mask 

def detect_rc(data):
    """
    Detect RC pixels in Dark
    
    Thresholds from Roman. Also need linearity tests.

    Parameters
    ----------
    dark_slope : ndarray
        (Nactpix, Nactpix) dark slope
    warm_threshold : float, optional
        Warm pixel detection threshold, by default 0.1
    hot_threshold : float, optional
        Hot pixel detection threshold, by default 0.5

    Returns
    -------
    ndarray
        Bitmask of hot & warm pixel
    """
    pass


