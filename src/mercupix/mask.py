import numpy as np 
import warnings
import bottleneck as bn 
from scipy.stats import median_abs_deviation
from astropy.nddata.bitmask import bitfield_to_boolean_mask

_MASK_DTYPE = 'uint32' 
_BITFLAGS_DICT = dict(GOOD=0,
                      DIS=2**0, # Disconnected but might change to DEAD.
                      CR=2**1, # Cosmic rays but might change to JUMP.
                      SATURATED=2**2,
                      HOT=2**3,
                      WARM=2**4,
                      CM=2**5, #Basically candidate for CM & Snowball.
                      REF=2**6,
                      RC=2**7,
                      NL=2**8,
                      RTN=2**9)


def get_boolmask(mask, bitflags=_BITFLAGS_DICT, **kwargs):
    boolmask = bitfield_to_boolean_mask(mask, flag_name_map=bitflags, **kwargs)
    return boolmask

def get_saturated(data=None, sat_threshold=1):
    """Identify saturated pixel in data.

    Parameters
    ----------
    data : ndarray, optional
        Raw datacube (in uint16)
        
    sat_threshold : float, optional
        Saturation threshold if one wants to flag saturation lower than digital saturation.
        Defaults to 1. 
        
    Returns:
        ndarray : Boolean (N, Npix, Npix) array where bit 4 means saturated pixels.

    """
    saturated_pixels = np.zeros(data.shape, dtype=_MASK_DTYPE) 
    mask = data >= int(sat_threshold*65535)
    saturated_pixels[mask] = _BITFLAGS_DICT['SATURATED']
    
    return saturated_pixels

def get_charge_migration(mask, kernel=None):
    # Assume that mask will be issued directly from saturated pixels.
    from scipy.ndimage import binary_dilation, generate_binary_structure
    
    if kernel is None : 
        kernel = generate_binary_structure(2, 2)    
    new_mask = binary_dilation(mask, kernel)
    new_mask[mask.astype(bool)] = False
    
    mask[new_mask.astype(bool)] = _BITFLAGS_DICT['CM']
    return mask

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

    if path is not None : 
        mask = load_downpix(path=path)
    else : 
        warnings.warn('No disconnected pixel map provided. \n No DIS Mask.')
        mask = 0
                
    return mask

def get_hotpix(data=None, path=None, **kwargs):
    """ Get high known dark pixels identified in data.

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

def get_zscores(data, threshold_mad=9, threshold_std=5):
    z_mad = (data - np.nanmedian(data, axis=0, keepdims=True))
    z_std = z_mad/np.nanstd(data,axis=0, ddof=1, keepdims=True)
    
    z_mad /=median_abs_deviation(data, scale='normal', nan_policy='omit', axis=0)
    
    mask = (z_mad >= threshold_mad) & (z_std >= threshold_std)
    return mask

def get_window(data, stat_func='median', window_length=4, min_count=1, axis=0):
    window = getattr(bn.move, f'move_{stat_func}')
    rolled_stat = window(data, window=window_length, min_count=min_count, axis=axis)

    zscore = (data - rolled_stat)/median_abs_deviation(data, scale='normal', nan_policy='omit', axis=0)
    
    return zscore , rolled_stat

def get_cosmics(data, inpaint=True, cr_threshold_mad=7):
    """
    
    """
    from astropy.stats import sigma_clip 
    
    mask = np.zeros(data.shape, dtype=_MASK_DTYPE)
    
    diff_data = np.diff(data, axis=0)

    zscore, med = get_window(diff_data, stat_func='median', window_length=4, min_count=1, axis=0)
    cr = zscore >= cr_threshold_mad
    
    mask_frame0 = sigma_clip(data[0], 
                             cenfunc='median', 
                             stdfunc='mad_std', 
                             sigma_lower=100000, 
                             sigma_upper=cr_threshold_mad)
    
    mask[np.concatenate([ mask_frame0.mask[None,:] , cr], axis=0)] = _BITFLAGS_DICT['CR']    
    
    if inpaint : 
        diff_data = diff_data.copy()
        diff_data[cr] = med[cr]
        data = np.cumsum(np.append(diff_data[0][None,:,:] , diff_data, axis=0), axis=0)
    
    return mask, data
        
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

    return mask 

def detect_rc(data):
    """
    Detect RC pixels in Dark
    
    Thresholds from Roman. 
    Also need linearity tests.

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
        Bitmask of RC pixels.
    """
    
    # Strategy :  
    # From stack of ramp data & slope
    # QF based selection.
    # Will see how this goes. 
    pass


# Deprecated cosmics detection.
def get_cosmics_dep(data, to_mask=None, cr_threshold_mad=9, cr_threshold_std=5):
    from astropy.stats import sigma_clip 

    mask = np.zeros(data.shape, dtype=_MASK_DTYPE)

    if to_mask is None : 
        zscore = get_zscores(np.diff(data, axis=0), 
                             threshold_mad=cr_threshold_mad, 
                             threshold_std=cr_threshold_std)
   
    else : 
        data = np.ma.masked_where(to_mask, data) #Create masked array for easy diff

        # Then convert to NaN because faster than using the masked array itself (x2).
        zscore = get_zscores(np.ma.diff(data, axis=0).filled(fill_value=np.nan), 
                             threshold_mad=cr_threshold_mad, 
                             threshold_std=cr_threshold_std)

        zscore = zscore & ~to_mask[1:]
        
    # Special treatment for first frame.
    mask_frame0 = sigma_clip(data[0], 
                             cenfunc='median', 
                             stdfunc='mad_std', 
                             sigma_lower=100000, 
                             sigma_upper=cr_threshold_mad)
    
    mask[np.concatenate([ mask_frame0.mask[None,:] , zscore], axis=0)] = _BITFLAGS_DICT['CR']    
    return mask 

