"""Data quality bitmasks and pixel flagging.

Each flag is a power of two, defined in `_BITFLAGS_DICT`:

- ``GOOD`` (0): good pixel,
- ``DIS`` (1): disconnected (dead) pixel,
- ``CR`` (2): cosmic ray (jump),
- ``SATURATED`` (4): saturated pixel,
- ``HOT`` (8): hot pixel,
- ``WARM`` (16): warm pixel,
- ``CM`` (32): charge migration or snowball candidate,
- ``REF`` (64): reference pixel,
- ``RC`` (128): RC pixel,
- ``NL`` (256): non-linearity issue,
- ``RTN`` (512): random telegraph noise.
"""
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
    """Convert a bitmask to a boolean mask.

    Parameters
    ----------
    mask : ndarray
        Integer bitmask.
    bitflags : dict, optional
        Mapping between flag names and bit values. Default is
        `_BITFLAGS_DICT`.
    **kwargs
        Passed to `astropy.nddata.bitmask.bitfield_to_boolean_mask`, e.g.
        ``ignore_flags`` to select the flags to consider.

    Returns
    -------
    ndarray
        Boolean mask with the shape of ``mask``.
    """
    boolmask = bitfield_to_boolean_mask(mask, flag_name_map=bitflags, **kwargs)
    return boolmask

def get_saturated(data=None, sat_threshold=1):
    """Flag the saturated pixels.

    Parameters
    ----------
    data : ndarray
        Raw datacube in ADU (uint16).
    sat_threshold : float, optional
        Saturation threshold as a fraction of the digital saturation
        (65535 ADU), to flag saturation below the digital limit. Default
        is 1.

    Returns
    -------
    ndarray
        Bitmask with the shape of ``data``, set to ``SATURATED`` where
        ``data >= sat_threshold * 65535``.
    """
    saturated_pixels = np.zeros(data.shape, dtype=_MASK_DTYPE) 
    mask = data >= int(sat_threshold*65535)
    saturated_pixels[mask] = _BITFLAGS_DICT['SATURATED']
    
    return saturated_pixels

def get_charge_migration(mask, kernel=None):
    # Assume that mask will be issued directly from saturated pixels.
    """Flag the neighbors of saturated pixels as charge migration candidates.

    The flagged pixels are dilated with ``kernel`` and the new pixels are
    set to ``CM``.

    Parameters
    ----------
    mask : ndarray
        Saturation bitmask, as returned by `get_saturated`. Modified in
        place.
    kernel : ndarray, optional
        Structuring element of the dilation. If None, a 3x3 kernel with
        full connectivity is used. Default is None.

    Returns
    -------
    ndarray
        Input ``mask`` with the charge migration candidates set to ``CM``.

    See Also
    --------
    scipy.ndimage.binary_dilation
    """
    from scipy.ndimage import binary_dilation, generate_binary_structure
    
    if kernel is None : 
        kernel = generate_binary_structure(2, 2)    
    new_mask = binary_dilation(mask, kernel)
    new_mask[mask.astype(bool)] = False
    
    mask[new_mask.astype(bool)] = _BITFLAGS_DICT['CM']
    return mask

def get_downpix(path=None):
    """Get the disconnected (dead) pixel map.

    Parameters
    ----------
    path : str, optional
        Path to the disconnected pixel map, from the metadata. If None, a
        warning is issued and no pixel is flagged. Default is None.

    Returns
    -------
    ndarray or int
        Disconnected pixel map of shape ``(ny, nx)``, or 0 if ``path`` is
        None.

    See Also
    --------
    mercupix.io.load_downpix
    """
    from .io import load_downpix

    if path is not None : 
        mask = load_downpix(path=path)
    else : 
        warnings.warn('No disconnected pixel map provided. \n No DIS Mask.')
        mask = 0
                
    return mask

def get_hotpix(data=None, path=None, **kwargs):
    """Get the known hot and warm pixel map.

    Parameters
    ----------
    data : ndarray, optional
        Unused. Default is None.
    path : str, optional
        Path to the hot pixel map, from the metadata. If None, a warning is
        issued and no pixel is flagged. Default is None.
    **kwargs
        Unused.

    Returns
    -------
    ndarray or int
        Hot and warm pixel bitmask of shape ``(ny, nx)``, or 0 if ``path``
        is None.

    See Also
    --------
    detect_hotpix
    mercupix.io.load_hotpix
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
    """Flag outliers along the first axis.

    A value is an outlier if its deviation from the median along the first
    axis exceeds both ``threshold_mad`` normalized median absolute
    deviations and ``threshold_std`` standard deviations.

    Parameters
    ----------
    data : ndarray
        Data of shape ``(n, ...)``, e.g. group differences.
    threshold_mad : float, optional
        Threshold in units of the normalized median absolute deviation.
        Default is 9.
    threshold_std : float, optional
        Threshold in units of the standard deviation. Default is 5.

    Returns
    -------
    ndarray
        Boolean mask with the shape of ``data``, True for outliers.
    """
    z_mad = (data - np.nanmedian(data, axis=0, keepdims=True))
    z_std = z_mad/np.nanstd(data,axis=0, ddof=1, keepdims=True)
    
    z_mad /=median_abs_deviation(data, scale='normal', nan_policy='omit', axis=0)
    
    mask = (z_mad >= threshold_mad) & (z_std >= threshold_std)
    return mask

def get_window(data, stat_func='median', window_length=5, min_count=1, axis=0):
    """Compute z-scores relative to a moving statistic.

    Parameters
    ----------
    data : ndarray
        Input data, e.g. group differences.
    stat_func : str, optional
        Moving statistic, as named in `bottleneck` (``'median'``,
        ``'mean'``, ...). Default is ``'median'``.
    window_length : int, optional
        Length of the moving window. Default is 5.
    min_count : int, optional
        Minimum number of non-NaN values in a window. Default is 1.
    axis : int, optional
        Axis along which the window moves. Default is 0.

    Returns
    -------
    zscore : ndarray
        Deviation of ``data`` from the moving statistic, in units of the
        normalized median absolute deviation along ``axis``.
    rolled_stat : ndarray
        Moving statistic, with the shape of ``data``.
    """
    window = getattr(bn.move, f'move_{stat_func}')
    rolled_stat = window(data, window=window_length, min_count=min_count, axis=axis)

    zscore = (data - rolled_stat)/median_abs_deviation(data, scale='normal', nan_policy='omit', axis=axis)
    
    return zscore, rolled_stat

def get_cosmics(data, inpaint=True, cr_threshold_mad=7):
    """Detect, and optionally correct, the jumps (cosmic rays) in a ramp.

    Jumps are detected as outliers in the group differences, relative to a
    moving median of 5 differences. Outliers in the first group are
    detected by sigma clipping across the frame.

    Parameters
    ----------
    data : ndarray
        Datacube of shape ``(ngroups, ny, nx)``.
    inpaint : bool, optional
        Whether to replace the flagged differences by the moving median and
        rebuild the ramp. Default is True.
    cr_threshold_mad : float, optional
        Detection threshold, in units of the normalized median absolute
        deviation. Default is 7.

    Returns
    -------
    mask : ndarray
        Bitmask of shape ``(ngroups, ny, nx)``, set to ``CR`` for the groups
        affected by a jump.
    data : ndarray
        Datacube, corrected if ``inpaint`` is True.

    See Also
    --------
    get_window
    """
    from astropy.stats import sigma_clip 
    
    mask = np.zeros(data.shape, dtype=_MASK_DTYPE)
    
    diff_data = np.diff(data, axis=0)

    zscore, med = get_window(diff_data, stat_func='median', window_length=5, min_count=1, axis=0)
    cr = zscore >= cr_threshold_mad
    
    if inpaint :
         diff_data = diff_data.copy()
         diff_data[cr] = med[cr]
         data = np.cumsum(np.append(data[0][None,:,:], diff_data, axis=0), axis=0)

    mask_frame0 = sigma_clip(data[0], 
                             cenfunc='median', 
                             stdfunc='mad_std', 
                             sigma_lower=100000, 
                             sigma_upper=cr_threshold_mad)
    
    mask[np.append(mask_frame0.mask[None,:,:] , cr, axis=0)] = _BITFLAGS_DICT['CR']    
    
    return mask, data
        
def detect_hotpix(dark_slope, warm_threshold=0.15, hot_threshold=0.5):
    """Detect hot and warm pixels from a dark slope.

    Thresholds are taken from Roman.

    Parameters
    ----------
    dark_slope : ndarray
        Dark slope of shape ``(ny_act, nx_act)``.
    warm_threshold : float, optional
        Lower dark slope threshold for warm pixels. Default is 0.15.
    hot_threshold : float, optional
        Lower dark slope threshold for hot pixels. Default is 0.5.

    Returns
    -------
    ndarray
        Bitmask with the shape of ``dark_slope``, set to ``WARM`` or
        ``HOT``.

    Notes
    -----
    Linearity tests are still needed.
    """

    mask = np.zeros(dark_slope.shape, _MASK_DTYPE)
    mask[(dark_slope >= warm_threshold) & (dark_slope <hot_threshold )] += _BITFLAGS_DICT['WARM']
    mask[(dark_slope >= hot_threshold)] += _BITFLAGS_DICT['HOT']

    return mask 

def detect_rc(data):
    """Detect RC pixels in dark data.

    Parameters
    ----------
    data : ndarray
        Stack of dark ramps and slopes.

    Returns
    -------
    ndarray
        Bitmask set to ``RC`` for the RC pixels.

    Notes
    -----
    Not implemented yet: currently returns None. The intended strategy is
    a selection on the fit quality of a stack of dark ramps.
    """
    
    # Strategy :  
    # From stack of ramp data & slope
    # QF based selection.
    # Will see how this goes. 
    pass


# Deprecated cosmics detection.
def get_cosmics_dep(data, to_mask=None, cr_threshold_mad=9, cr_threshold_std=5):
    """Detect jumps (cosmic rays) in a ramp (deprecated).

    Deprecated, use `get_cosmics` instead.

    Parameters
    ----------
    data : ndarray
        Datacube of shape ``(ngroups, ny, nx)``.
    to_mask : ndarray, optional
        Boolean mask of the values to ignore, with the shape of ``data``.
        Default is None.
    cr_threshold_mad : float, optional
        Detection threshold in units of the normalized median absolute
        deviation. Default is 9.
    cr_threshold_std : float, optional
        Detection threshold in units of the standard deviation. Default
        is 5.

    Returns
    -------
    ndarray
        Bitmask of shape ``(ngroups, ny, nx)``, set to ``CR`` for the groups
        affected by a jump.

    See Also
    --------
    get_cosmics
    get_zscores
    """
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

