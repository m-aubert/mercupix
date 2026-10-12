"""Detector characterization tools.

Includes photon transfer curve, readout-order utilities, noise power
spectra and classical non-linearity estimation.
"""
import numpy as np 

# --------------------
# ---- Utils func ----
# --------------------

def get_ptc(data):
    """Compute the photon transfer curve statistics of a datacube.

    Parameters
    ----------
    data : ndarray
        Datacube of shape ``(ngroups, ny, nx)``.

    Returns
    -------
    mean : ndarray
        Mean of each group, of shape ``(ngroups,)``.
    std : ndarray
        Standard deviation of each group, of shape ``(ngroups,)``.
    """
    return data.mean(axis=(1,2)) , data.std(axis=(1,2), ddof=1) 

def get_channel_slices(ramp, channel=0, corr_refpix=False): 
    """Get the slices selecting a readout channel in a frame.

    Parameters
    ----------
    ramp : mercupix.baseramp.BaseRamp
        Ramp giving the detector geometry.
    channel : int, optional
        Index of the readout channel. Default is 0.
    corr_refpix : bool, optional
        Whether the frame is cropped to the active area (reference pixel
        corrected). Default is False.

    Returns
    -------
    slice_ : slice
        Row slice of the frame.
    slice_channel : slice
        Column slice of the channel.
    """
    slice_ = ramp.actslice if corr_refpix else slice(0,ramp.frame_shape[0])
    offset_up, offset_down = 0,0

    slice_channel = slice(channel*ramp.npix_per_channel+offset_up , 
                        (channel+1)*ramp.npix_per_channel-offset_down,1)

    if corr_refpix :
        if channel == 0 : 
            offset_up, offset_down = ramp.nrefpixels,0
            slice_channel = slice(0, ramp.npix_per_channel-ramp.nrefpixels) 

        if channel == ramp.nchannels-1:
            offset_up, offset_down = 0,ramp.nrefpixels
            slice_channel = slice(channel*ramp.npix_per_channel-offset_down, ramp.active_shape[1]) 
            
    return slice_, slice_channel

def to_readout_order(data, ramp, channel=0, corr_refpix=True):    
    """Extract a readout channel in readout order.

    Odd channels are read in the opposite direction, so their columns are
    reversed.

    Parameters
    ----------
    data : ndarray
        Frame of shape ``(ny, nx)`` or datacube of shape
        ``(ngroups, ny, nx)``.
    ramp : mercupix.baseramp.BaseRamp
        Ramp giving the detector geometry.
    channel : int, optional
        Index of the readout channel. Default is 0.
    corr_refpix : bool, optional
        Whether ``data`` is cropped to the active area. Default is True.

    Returns
    -------
    ndarray
        Columns of ``channel`` in readout order.
    """
    _, slice_channel = get_channel_slices(ramp, channel=channel, corr_refpix=corr_refpix)
    
    if data.ndim == 3 : 
        return data[:,:, slice_channel] if channel%2 == 0 else data[:, :, slice_channel][:,:,::-1]
    elif data.ndim == 2 :        
        return data[:, slice_channel] if channel%2 == 0 else data[:,slice_channel][:,::-1]


def get_readout_sequence(ramp,data=None, channel=0, frame=0, corr_refpix=True, pad=12):
    """Rebuild the readout sequence of a channel for a single frame.

    The channel columns are placed in readout order in an array padded
    with ``pad`` zero columns per row, mimicking the row overhead.

    Parameters
    ----------
    ramp : mercupix.ramp.RawRamp
        Ramp giving the detector geometry.
    data : ndarray, optional
        Frame of shape ``(ny, nx)``. If None, group ``frame`` of the
        preprocessed ramp is used. Default is None.
    channel : int, optional
        Index of the readout channel. Default is 0.
    frame : int, optional
        Index of the group used when ``data`` is None. Default is 0.
    corr_refpix : bool, optional
        Whether ``data`` is cropped to the active area (reference pixel
        corrected). Default is True.
    pad : int, optional
        Number of overhead columns appended to each row. Default is 12.

    Returns
    -------
    ndarray
        Readout sequence of shape ``(ny, npix_per_channel + pad)``, as
        float32.
    """
    if data is None : 
        data = ramp.preprocess_ramp(corr_refpix=corr_refpix)[frame]

    slice_ = ramp.actslice if corr_refpix else slice(0,ramp.frame_shape[0])
    offset_up, offset_down = 0,0

    slice_channel = slice(channel*ramp.npix_per_channel+offset_up , 
                          (channel+1)*ramp.npix_per_channel-offset_down,1)

    if corr_refpix :
        if channel == 0 : 
            offset_up, offset_down = ramp.nrefpixels,0
            slice_channel = slice(0, ramp.npix_per_channel-ramp.nrefpixels) 

        if channel == ramp.nchannels-1:
            offset_up, offset_down = 0,ramp.nrefpixels
            slice_channel = slice(channel*ramp.npix_per_channel-offset_down, ramp.active_shape[1]) 
            
    test = np.zeros((ramp.frame_shape[0], ramp.npix_per_channel+pad), dtype='float32') 
    test[slice_, offset_up:ramp.npix_per_channel-offset_down] = data[:, slice_channel] if channel%2 == 0 else data[:, slice_channel][:,::-1]

    return test

    
def get_psd(ramp, data=None, channel=0, frame=0, corr_refpix=True, pad=12, readout_freq=203_000): 
    """Compute the noise power spectral density of a readout channel.

    For each group, the channel is unrolled in readout order (see
    `get_readout_sequence`) and its power spectrum is computed with a real
    FFT.

    Parameters
    ----------
    ramp : mercupix.ramp.RawRamp
        Ramp giving the detector geometry.
    data : ndarray, optional
        Datacube of shape ``(ngroups, ny, nx)``. If None, the preprocessed
        ramp is used. Default is None.
    channel : int, optional
        Index of the readout channel. Default is 0.
    frame : int, optional
        Unused. Default is 0.
    corr_refpix : bool, optional
        Whether ``data`` is cropped to the active area (reference pixel
        corrected). Default is True.
    pad : int, optional
        Number of overhead columns appended to each row. Default is 12.
    readout_freq : float, optional
        Pixel readout frequency, in Hz. Default is 203000.

    Returns
    -------
    k : ndarray
        Frequencies, in Hz.
    psd : ndarray
        Power spectral density of each group, of shape
        ``(ngroups, len(k))``, as float32.
    """
    if data is None : 
        data = ramp.preprocess_ramp(corr_refpix=corr_refpix)[0]

    slice_ = ramp.actslice if corr_refpix else slice(0,ramp.frame_shape[0])
    offset_up, offset_down = 0,0

    slice_channel = slice(channel*ramp.npix_per_channel+offset_up , 
                          (channel+1)*ramp.npix_per_channel-offset_down,1)

    if corr_refpix :
        if channel == 0 : 
            offset_up, offset_down = ramp.nrefpixels,0
            slice_channel = slice(0, ramp.npix_per_channel-ramp.nrefpixels) 

        if channel == ramp.nchannels-1:
            offset_up, offset_down = 0,ramp.nrefpixels
            slice_channel = slice(channel*ramp.npix_per_channel-offset_down, ramp.active_shape[1]) 
            
    psdsraw = []
    for i in range(0, ramp.ngroups):
        test = np.zeros((ramp.frame_shape[0], ramp.npix_per_channel+pad), dtype='float32') 
        test[slice_, offset_up:ramp.npix_per_channel-offset_down] = data[i, :, slice_channel] if channel%2 == 0 else data[i, :, slice_channel][:,::-1]
        test = test.flatten()
        psdsraw.append((np.abs(np.fft.rfft(test))**2*1./readout_freq * 1./test.size*2).astype('float32'))

    k = np.fft.rfftfreq(test.size, 1./readout_freq)
    return k , np.asarray(psdsraw)
    
    
# ----------------------
# ---- Classical NL ----
# ----------------------
    
class ClassicalNL : 
    
    """Estimate the classical non-linearity from a ramp.

    The mean (or median) ramp is fitted with a polynomial in time, from
    which the linear part is extracted. The fractional deviation from it is
    then fitted as a polynomial of the signal.

    Parameters
    ----------
    ramp : ndarray
        Ramp data, e.g. a datacube of shape ``(ngroups, ny, nx)``.
    times : ndarray
        Time of each group, of shape ``(ngroups,)``.
    degree : int, optional
        Degree of the polynomial fitted to the ramp. Default is 3.

    See Also
    --------
    mercupix.ramp.RawRamp.correct_nl
    """
    def __init__(self, ramp, times, degree=3):
        self.degree= degree
        self.ramp= ramp 
        self.times= times 
        
    @staticmethod
    def poly_model(x, *params, degree=3): 
        """Evaluate the non-linear ramp model.

        The model is ``params[0] + (a x) + sum_{i>=2} params[i] * (a x)**i``
        with ``a = params[1]``.

        Parameters
        ----------
        x : ndarray
            Times at which the model is evaluated.
        *params : float
            Model parameters: offset, linear rate, then the coefficients of
            orders 2 to ``degree``.
        degree : int, optional
            Degree of the model. Default is 3.

        Returns
        -------
        ndarray
            Model evaluated at ``x``.
        """
        model = 0 
        for i in range(degree+1):
            if i != 1 : 
                model += params[i]*(params[1]*x)**i 
            else : 
                model += (params[1]*x)**i 
                
        return model 
    
    def get_model(self, x, *params):
        """Evaluate `poly_model` with the degree of the instance.

        Parameters
        ----------
        x : ndarray
            Times at which the model is evaluated.
        *params : float
            Model parameters, see `poly_model`.

        Returns
        -------
        ndarray
            Model evaluated at ``x``.
        """
        return self.poly_model(x, *params, degree=self.degree)
        
    def get_nlcorr(self, on_mean=True, on_median=True,  init_val=None, corr_degree=4): 
        """Estimate the non-linearity correction coefficients.

        Parameters
        ----------
        on_mean : bool, optional
            Whether to fit the mean ramp. Takes precedence over
            ``on_median``. Default is True.
        on_median : bool, optional
            Whether to fit the median ramp. Default is True.
        init_val : list of float, optional
            Initial values of the model parameters. If None, all are 0
            except the linear rate. Default is None.
        corr_degree : int, optional
            Degree of the polynomial correction. Default is 4.

        Returns
        -------
        ndarray
            Correction coefficients, of length ``corr_degree + 1``, in
            increasing polynomial order.

        Raises
        ------
        NotImplementedError
            If both ``on_mean`` and ``on_median`` are False (per-pixel fit).
        """
        from scipy.optimize import curve_fit 

        if init_val is None : 
            init_val = [0]*(self.degree+1)
            init_val[1] = self.ramp[0].mean()/2 
             
        if self.ramp.ndim == 4:
            axis = (0,1,2)
        elif self.ramp.ndim == 3 :
            axis=(1,2)
        else :
            axis = -1

        if on_mean :
            data_vec = np.nanmean(self.ramp, axis=axis)
            err = self.ramp.std(axis=axis, ddof=1)
        elif on_median:
            data_vec = np.nanmedian(self.ramp, axis=axis)
            err = np.nanstd(self.ramp, axis=axis, ddof=1)

        else : 
            raise NotImplementedError('Fit per pixel not implemented yet')

        # Fit b+a*x + c*(a*x)**2 + .... 
        # Could probably minimize on degrees but unsure.
        # Legendre could be tested as well but unsure as well.
        p = curve_fit(self.get_model, self.times, data_vec,p0=init_val, sigma=err, maxfev = 15000)[0]

        linearized_data = np.poly1d(p[:2][::-1])(self.times)/data_vec  - 1      
        ps = np.polyfit(data_vec, linearized_data, corr_degree) 
        nl_coefs = ps[::-1] 
        
        return nl_coefs
