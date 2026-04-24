import numpy as np 
import warnings
from astropy.stats import mad_std
import bottleneck as bn  # For rolling median etc but useless actually.

_MASK_DTYPE = 'uint8' 

_BITFLAGS_DICT = dict(DIS=1,
                      CR=2,
                      SATURATED= 4,
                      HOT=8)    


def get_mask(
            disconnected=True,
            cosmics=False, 
            saturated=False,
            hotpix=False, 
            estimate_mask=False): 
        """Create a bitmask map of the ramp

        Parameters
        ----------
        disconnected : bool, optional
            Track disconnected or dark pixels, by default True
        cosmics : bool, optional
            Track cosmic tracks, by default False
        saturated : bool, optional
            Track saturated pixels, by default False
        noisy : bool, optional
            Track noisy or hot pixels, by default False
        estimate_mask : bool, optional
            Whether to estimate mask, by default False.
            Overrides all keyword parameters. 
            
        Returns
        -------
        ndarray
            A shape (N, Nactpix, Nactpix) bitmask.
        """        
        # Initialize mask
        mask = np.zeros(active_shape, dtype='uint8')

        if estimate_mask : 
            if disconnected : 
                mask += get_downpix() # From carac & to be udated in time
            
            if hotpix : 
                mask += get_hotpix() # From carac & to be udated in time
                
            if saturated : 
                mask += get_saturated()
            
            if cosmics : 
                mask += get_cosmics()

            if mask.sum() == 0 : 
                from astropy.stats import sigma_clip
                
                maskdata = sigma_clip(self.data, sigma=17, axis=(1,2))
                mask = maskdata.mask.astype('uint8')
                            
        return mask
    

def get_saturated(data=None):
    """Identify saturated pixel in data.

    Parameters
    ----------
    data : ndarray, optional
        Raw datacube (in uint16)
        
    Returns:
        ndarray : Boolean (N, Npix, Npix) array where True means saturated pixels.

    """
    saturated_pixels = np.zeros(data.shape, dtype=_MASK_DTYPE) 
    mask = data == 65535
    saturated_pixels[mask] = _BITFLAGS_DICT['SATURATED']
    
    return saturated_pixels

def get_downpix(path=None):
    """Recover pixels markes as dark/disconnected.

    Parameters
    ----------
    path (str, optional) : Path to disconnected pixel mask from metadata.
        Data cube (N, Npix, Npix). by default None
        
    Returns
    -------
        ndarray : Boolean (N, Npix, Npix) array where True means disconnected.
    """
    from .io import load_downpix
    # path tbd
    if path is not None : 
        mask = load_downpix(path=path)
    else : 
        warnings.warn('No disconnected pixel map provided. No DIS Mask.')
        mask = 0
                
    return mask

def get_hotpix(self, data=None):
    """ Get identified noisy pixel in data.

    Parameters
    ----------
    data : ndarray, optional
        Data cube (N, Npix, Npix). by default None

    Returns
    -------
    ndarray
        Boolean (N, Npix, Npix) array where True means noisy pixels.
    """
    # Boolean
    # Get noisy pixels as bitmap or as boolean ?
    from astropy.stats import sigma_clip
        
    maskdata = sigma_clip(np.diff(self.data, axis=0), sigma=17, axis=(1,2))
    mask = maskdata.mask.astype(bool)
    # For now this identifies noisy pixels in terms of delta. 
    # Sigma_clipping along time axis might help in detecting whacky pixels as well ?
    # See how this goes. 
    
    
    # Detect outliers in data that is not :
    # - CR
    # - DIS
    # - saturated
    # Or load from map of carac
    # Outlier rejection / frame ? 
    # Check things out and how that works basically.
    
    return mask

def get_cosmics(data, to_mask=None, threshold_mad=9, threshold_std=5):
    from astropy.stats import sigma_clip 
    
    mask = np.zeros(data.shape, _MASK_DTYPE)
    
    if to_mask is None : 
        zscore = get_zscores(np.diff(data, axis=0), threshold_mad=threshold_mad, threshold_std=threshold_std)
    
    else : 
        data_ = np.ma.array(np.diff(data, axis=0), mask=to_mask[1:])
        zscore = get_zscores(data_, threshold_mad=threshold_mad, threshold_std=threshold_std)
        zscore = mask.data & ~to_mask[1:]
    
    # Special treatment for first frame.
    mask_frame0 = sigma_clip(data[0], cenfunc='median', stdfunc='mad_std', sigma_lower=100000, sigma_upper=threshold_mad)

    mask[np.stack([mask_frame0.mask , zscore])] = _BITFLAGS_DICT['CR']    
    
    return mask 
        
def get_zscores(data, threshold_mad=9, threshold_std=5) :
    z_mad = (data - np.median(data, axis=0, keepdims=True) )/mad_std(data, axis=0)
    z_std = (data - np.median(data, axis=0, keepdims=True) )/data.std(axis=0, ddof=1, keepdims=True)
    
    mask = (z_mad >= threshold_mad) & (z_std >= threshold_std)
    return mask
            
        