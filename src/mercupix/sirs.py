"""SIRS-like reference pixel correction in the Fourier domain.

The correlated noise of each readout channel is modeled from the left and
right reference columns, with frequency-dependent weights ``alpha`` and
``beta`` trained on dark ramps. The Fourier transforms of the reference
columns are computed with an incomplete FFT (Rauscher 2026).

Notes
-----
The training is currently tuned for IP2I data.
"""
import numpy as np
from scipy.stats import median_abs_deviation
from joblib import Parallel, delayed #Temporary accelaration before we change strategy.
from .fitter import LSQFit
from .carac import get_readout_sequence
from .ramp import RawRamp


# Utils func to add refs to data -> add to Ramp
def get_data_with_ref(data, ramp_with_refpix, ramp):
    """Add the reference pixels back around a processed active datacube.

    Parameters
    ----------
    data : ndarray
        Processed datacube of shape ``(ngroups, ny_act, nx_act)``.
    ramp_with_refpix : ndarray
        Datacube of shape ``(ngroups, ny, nx)`` providing the reference
        pixels.
    ramp : mercupix.baseramp.BaseRamp
        Ramp giving the detector geometry.

    Returns
    -------
    ndarray
        Datacube of shape ``(ngroups, ny, nx)``.

    Notes
    -----
    Assumes a reference pixel border of 4 pixels.
    """
    data = np.column_stack([ramp_with_refpix[:,:4,ramp.actslice],
                            data, ramp_with_refpix[:,-4:, ramp.actslice]])
    data = np.dstack([ramp_with_refpix[:,:,:4], data, ramp_with_refpix[:,:, -4:]])
    return data


# Training function -> will become a function of a class. (or a class method ?)
# BEWARE -- TUNED FOR IP2I FOR NOW.
def train_sirs(ramp_list, refpix=4,  nroh=12, detector_freq=200_000, njobs=5, debug=False,
               filenames_kwargs=dict(datatype='ip2i',
                                     time_arr_path='/run/seq_1/env_measures')):

    """Train the SIRS correction weights on a set of dark ramps.

    For each ramp, the signal is removed with a polynomial fit and the
    cross power spectra between the regular pixels and the reference
    columns of each channel are accumulated frame by frame. The weights are
    then obtained by solving the resulting linear system per frequency.

    Parameters
    ----------
    ramp_list : list of tuple
        ``(filename, frame_path)`` of each dark ramp.
    refpix : int, optional
        Width of the reference pixel border, in pixels. Default is 4.
    nroh : int, optional
        Number of overhead columns per row. Default is 12.
    detector_freq : float, optional
        Pixel readout frequency, in Hz. Default is 200000.
    njobs : int, optional
        Number of parallel jobs. If 0 or lower, frames are processed
        sequentially. Default is 5.
    debug : bool, optional
        Unused. Default is False.
    filenames_kwargs : dict, optional
        Passed to `mercupix.ramp.RawRamp.from_filename`. Default is
        ``dict(datatype='ip2i', time_arr_path='/run/seq_1/env_measures')``.

    Returns
    -------
    alpha : ndarray
        Weights of the left reference columns, per channel and frequency,
        as complex64.
    beta : ndarray
        Weights of the right reference columns, per channel and frequency,
        as complex64.
    fft_basis_ref : ndarray
        Pseudo-inverse of the incomplete FFT basis of the reference columns.
    results : ndarray
        Accumulated spectra, of shape ``(nchannels, 6, nfreq)``. See
        `get_fourier_corrections`.
    freqs : ndarray
        Frequencies of the full readout sequence, in Hz.

    See Also
    --------
    process_frame
    get_sirs_ref
    """
    results = 0
    counters = 0
    offsets = 0

    for i, (dark_data, frame_path) in enumerate(ramp_list) :
        ramp = RawRamp.from_filename(dark_data, frame_path=frame_path,  **filenames_kwargs, )

        # This part is standalone.
        rampdata, _, mask = ramp.preprocess_ramp(corr_refpix=False, mask=True,
                                                 mask_kwargs=dict(cosmics=False, saturated=True))
        #_ , rampdata = get_cosmics(rampdata, inpaint=True)

        rampdata = get_data_with_ref(rampdata, ramp.data, ramp) #Adding refpix left & right.
        #Fit procedure
        fitter = LSQFit(data=rampdata, times=ramp.times, degree=3)
        coefs = fitter.fit(inplace=False)
        residuals = fitter.get_residuals(coefs=coefs)
        #Get residuals.
        del coefs, rampdata

        # End of standalone.
        if i == 0 :  #could be SIRS class init.
            nrow_pix = ramp.frame_shape[0]
            ncol_pix = ramp.npix_per_channel+nroh
            n = nrow_pix*ncol_pix

            freqs = np.fft.rfftfreq(n, d=1./detector_freq)

            pix_index = np.arange(n).reshape(nrow_pix,ncol_pix)
            j_ref = pix_index[:,:4].reshape(-1)
            k_ref = np.concatenate([np.arange(len(freqs))[:nrow_pix//2+1] ,
                                    np.arange(len(freqs))[-nrow_pix//2:]])
            fft_basis_ref, _ = get_fft_basis(n,j_ref, k_ref)

            pixmask = np.zeros(ramp.frame_shape, dtype=bool)
            pixmask[ramp.actslice, ramp.actslice] = True

        print('Now processing frame per frame')
        if njobs > 0 :
            result = Parallel(n_jobs=njobs)(delayed(process_frame)(ramp, frame, good_,
                                                                 refpix, fft_basis_ref,
                                                                 pixmask, nroh=nroh)
                                          for frame, good_ in zip(residuals, mask))

            result_ , offset_, counter_ = zip(*result)

            del result

            results += np.asarray(result_).sum(axis=0)
            offsets += np.asarray(counter_).sum(axis=0)
            counters += np.asarray(offset_).sum(axis=0)

            del result_, counter_, offset_

        else :
            print('Not parallel')
            for i, (frame, good_) in enumerate(zip(residuals, mask)):
                result  = process_frame(ramp, frame, good_, refpix, fft_basis_ref, pixmask, nroh=nroh)
                results += result[0]
                offsets += result[1]
                counters += result[2]
                if i%10 == 0 :
                    print(result[0].shape)
                    print(f"{i/len(residuals)*100}%")

        del residuals, mask

    # "Solving here"
    denom_ = results[:,2]*results[:,1] - np.real(results[:,5]*results[:,5].conjugate())
    alpha = (results[:,2]*results[:,4] - results[:,3]*results[:,5]) / denom_ #(Zspec.real**2 + Zspec.imag**2))
    beta = (results[:,3]*results[:,1] - results[:,4]*np.conj(results[:,5])) / denom_ #(Zspec.real**2 + Zspec.imag**2))

    return alpha.astype(np.complex64), beta.astype(np.complex64), fft_basis_ref, results, freqs

def save_sirs(fft_basis_ref, alpha, beta, freqs, n, results,  filepath='sirs.h5', debug=True):
    # For now h5 because easy to mix diffent dimensions. But actually not sure it is an issue tbh.
    """Save the trained SIRS correction to an HDF5 file.

    Parameters
    ----------
    fft_basis_ref : ndarray
        Pseudo-inverse of the incomplete FFT basis of the reference columns.
    alpha : ndarray
        Weights of the left reference columns.
    beta : ndarray
        Weights of the right reference columns.
    freqs : ndarray
        Frequencies of the readout sequence, in Hz.
    n : int
        Number of pixels in the readout sequence of a channel.
    results : ndarray
        Accumulated spectra, as returned by `train_sirs`.
    filepath : str, optional
        Path of the output file. Default is ``'sirs.h5'``.
    debug : bool, optional
        Whether to also store the accumulated spectra. Default is True.

    See Also
    --------
    train_sirs
    """
    import h5py

    with h5py.File(filepath, 'w') as f:
        f.create_dataset("fft_basis", data = fft_basis_ref)
        f.create_dataset("alpha_corr", data = alpha)
        f.create_dataset("beta_corr", data = beta)
        f.create_dataset("freqs", data = freqs)
        f.create_dataset("npix_readout", data = n)
        if debug :
            f.create_dataset('Nspec', data=results[:, 0]) #still not sure what use is Nspec tbh.
            f.create_dataset('Lspec', data=results[:, 1])
            f.create_dataset('Rspec', data=results[:, 2])
            f.create_dataset('Xspec', data=results[:, 3])
            f.create_dataset('Yspec', data=results[:, 4])
            f.create_dataset('Zspec', data=results[:, 5])

    print(f"Saved : {filepath}")

def process_frame(ramp, frame, good_, refpix, fft_basis_ref, pixmask, nroh=12):

    """Compute the SIRS spectra of all channels for a single frame.

    Parameters
    ----------
    ramp : mercupix.baseramp.BaseRamp
        Ramp giving the detector geometry.
    frame : ndarray
        Residual frame of shape ``(ny, nx)``, including reference pixels.
    good_ : ndarray
        Bitmask of the active area, of shape ``(ny_act, nx_act)``. Nonzero
        values mark bad pixels.
    refpix : int
        Width of the reference pixel border, in pixels.
    fft_basis_ref : ndarray
        Pseudo-inverse of the incomplete FFT basis of the reference columns.
    pixmask : ndarray
        Boolean map of shape ``(ny, nx)``, True on the active area.
    nroh : int, optional
        Number of overhead columns per row. Default is 12.

    Returns
    -------
    coefs_spec : ndarray
        Spectra of each channel, of shape ``(nchannels, 6, nfreq)``. See
        `get_fourier_corrections`.
    offset : ndarray
        Offset between regular and reference pixels of the last channel.
    counter : ndarray
        Validity counter of the last channel.
    """
    coefs_spec = []
    offset_counter = []

    good = np.zeros(frame.shape, dtype=int)
    good[ramp.actslice, ramp.actslice] = good_[:,:]
    good = ~good.astype(bool)

    left_fft = inc_rfft(fft_basis_ref,frame[:, :refpix].reshape(-1), mask=None)
    right_fft = inc_rfft(fft_basis_ref,frame[:, -refpix:][:,::-1].reshape(-1), mask=None) #Reorder last pixels.

    for channel in range(ramp.nchannels):
        Nspec_, Lspec_, Rspec_, Xspec_, Yspec_, Zspec_, offset_, counter_ = get_fourier_corrections(ramp,
                                                                                frame,
                                                                                channel,
                                                                                pixmask,
                                                                                good,
                                                                                right_fft,
                                                                                left_fft,
                                                                                nroh=nroh,
                                                                                refpix=ramp.nrefpixels)

        if hasattr(Nspec_, "__len__"):
            coefs_spec.append([Nspec_,Lspec_, Rspec_, Xspec_, Yspec_, Zspec_])
            offset_counter.append([offset_, counter_])

        else :
            coefs_spec.append([np.zeros(left_fft.shape)]*6)
            offset_counter.append([offset_, counter_])

    offsets, counters = np.asarray(offset_counter).T
    return np.asarray(coefs_spec), offsets, counters #Testing, not ideal tbh.

# To go into the SIRS class
def get_fourier_corrections(ramp, frame, channel, pixmask, good, right_fft, left_fft,
                            nroh=12, refpix=4, GD_OP_SIG_MIN = 0, GD_OP_SIG_MAX = 30):

    """Compute the power and cross spectra of a channel and its references.

    The channel is unrolled in readout order, outliers and bad pixels are
    replaced by row means, and the overhead columns are filled by
    mirroring the last columns before computing the FFT.

    Parameters
    ----------
    ramp : mercupix.baseramp.BaseRamp
        Ramp giving the detector geometry.
    frame : ndarray
        Residual frame of shape ``(ny, nx)``.
    channel : int
        Index of the readout channel.
    pixmask : ndarray
        Boolean map of shape ``(ny, nx)``, True on the active area.
    good : ndarray
        Boolean map of shape ``(ny, nx)``, True for good pixels.
    right_fft : ndarray
        Incomplete FFT of the right reference columns.
    left_fft : ndarray
        Incomplete FFT of the left reference columns.
    nroh : int, optional
        Number of overhead columns per row. Default is 12.
    refpix : int, optional
        Width of the reference pixel border, in pixels. Default is 4.
    GD_OP_SIG_MIN : float, optional
        Minimum noise level for the channel to be used. Default is 0.
    GD_OP_SIG_MAX : float, optional
        Maximum noise level for the channel to be used. Default is 30.

    Returns
    -------
    Nspec : ndarray
        Power spectrum of the regular pixels.
    Lspec : ndarray
        Power spectrum of the left reference columns.
    Rspec : ndarray
        Power spectrum of the right reference columns.
    Xspec : ndarray
        Cross spectrum between regular pixels and right references.
    Yspec : ndarray
        Cross spectrum between regular pixels and left references.
    Zspec : ndarray
        Cross spectrum between right and left references.
    offset : float
        Mean offset between regular and reference pixels.
    counter : int
        1 if the channel is valid.

    Notes
    -----
    If the noise level of the channel is outside
    ``(GD_OP_SIG_MIN, GD_OP_SIG_MAX)``, eight zeros are returned instead.
    """
    d = get_readout_sequence(ramp,data=frame, channel=channel, corr_refpix=False, pad=nroh)

    # Transform mask to data.
    good = get_readout_sequence(ramp, data=good, channel=channel, corr_refpix=False, pad=nroh)
    good = good.astype(bool)

    # Get edge mask
    mask_edge = get_readout_sequence(ramp, data=pixmask, channel=channel, corr_refpix=False, pad=nroh)
    mask_edge = mask_edge.astype(bool)

    valid = mask_edge & good # Regular pixels
    minimum, maximum = np.percentile(d[valid].flatten(),[2.5, 97.5]) #Agressively remove outliers.
    valid[valid][d[valid] <= minimum] = False
    valid[valid][d[valid] >= maximum] = False

    # Add that to mask I guess.
    good = valid & good

    # Quality flags
    region = d[valid] #Overhead pixels are per default discarded.
    mu = np.mean(region)
    sigma = median_abs_deviation(region.ravel(), scale="normal") #np.std(region, ddof=1)
    #print(sigma, d[mask_edge][good[mask_edge]].std(ddof=1) )
    if not ((sigma > GD_OP_SIG_MIN) & (sigma < GD_OP_SIG_MAX)):
        return [0]*8

    ref = np.concatenate([d[:refpix, :-nroh], d[-refpix:, :-nroh]]).reshape(-1)
    rmin, rmax = np.percentile(ref.reshape(-1),[5//2,95//2]) #5% and 95% as min and max.
    mu_ref = np.mean(ref[(ref > rmin) & (ref < rmax)])

    # Remplacement des pixels invalides
    # Will see how to rewrite to remove loop if needed.
    flag_pix = np.all(good, axis=0)
    for i in range(good.shape[1]):
        if flag_pix[i]:
            continue

        if good[:, i].sum() > 0 :
            row_m =  np.mean(d[good[:, i]]) #Should be median ?

        else :
            row_m = mu

        d[:,i][~good[:, i]] = row_m

    # Colonnes d'overhead
    roi = d[:,-2*nroh:-nroh] #Get data before overhead.
    d[:,-nroh:] = roi[:,::-1] # Mirror data to fill in overhead.
    vector = d.reshape(-1)

    spectrum = np.fft.rfft(vector)

    keep = np.concatenate((spectrum[:ramp.frame_shape[0]//2+1],
                        spectrum[-ramp.frame_shape[0]//2:]))

    # This is a part to validate for example.
    Nspec = np.real(keep * np.conj(keep))
    Lspec = np.real(left_fft *np.conj(left_fft))
    Rspec = np.real(right_fft *np.conj(right_fft))
    Xspec = (keep *np.conj(right_fft))
    Yspec = (keep *np.conj(left_fft))
    Zspec = (right_fft *np.conj(left_fft))
    offset = (mu - mu_ref)

    return Nspec, Lspec, Rspec, Xspec, Yspec, Zspec, offset, 1

# Could be apply along axis ?
def get_sirs_ref(frame, fft_basis, alpha, beta , refpix=4, nroh=7, npix_per_channel=64):
    """Build the SIRS reference correction of a channel.

    The incomplete FFTs of the left and right reference columns are
    weighted by ``alpha`` and ``beta`` and transformed back to the readout
    sequence.

    Parameters
    ----------
    frame : ndarray
        Frame of shape ``(ny, nx)`` providing the reference columns.
    fft_basis : ndarray
        Pseudo-inverse of the incomplete FFT basis of the reference columns.
    alpha : ndarray
        Weights of the left reference columns for the channel.
    beta : ndarray
        Weights of the right reference columns for the channel.
    refpix : int, optional
        Width of the reference pixel border, in pixels. Default is 4.
    nroh : int, optional
        Number of overhead columns per row. Default is 7.
    npix_per_channel : int, optional
        Number of pixel columns per channel. Default is 64.

    Returns
    -------
    ndarray
        Reference correction of shape ``(ny, npix_per_channel)``, in readout
        order.
    """
    alpha = alpha.copy()
    beta = beta.copy()

    #alpha[0] = 0
    #beta[0] = 0 # Commented for now because it seems that it is better not to set to 0, "smoother PSD profile I guess".

    left_fftcor = inc_rfft(fft_basis,frame[:, :refpix].reshape(-1), mask=None)
    right_fftcor = inc_rfft(fft_basis,frame[:, -refpix:][:,::-1].reshape(-1), mask=None) #Reorder last pixels.

    n = frame.shape[0]*(npix_per_channel+nroh)

    ref = np.zeros(n, dtype=np.complex64)
    #ref = alpha*left_fftcor + right_fftcor*beta
    ref[:frame.shape[0]//2+1] = (alpha*left_fftcor + right_fftcor*beta)[:frame.shape[0]//2+1]
    ref[-frame.shape[0]//2:] = (alpha*left_fftcor + right_fftcor*beta)[-frame.shape[0]//2:]
    ref = np.fft.irfft(ref, n=n)
    ref = ref.reshape(frame.shape[0],npix_per_channel+nroh)[:, :npix_per_channel]

    return ref



# SIRS utils
def get_fft_basis(n, j_ref, k_ref,  mask=None):
    """Build the Fourier basis of an incomplete FFT (Rauscher 2026).

    Parameters
    ----------
    n : int
        Length of the full sequence.
    j_ref : ndarray
        Indices of the sampled positions in the sequence.
    k_ref : ndarray
        Indices of the frequencies to recover.
    mask : ndarray, optional
        Unused. Default is None.

    Returns
    -------
    pinvB : ndarray
        Moore-Penrose pseudo-inverse of ``B / n``, of shape
        ``(len(k_ref), len(j_ref))``.
    B : ndarray
        Fourier basis of shape ``(len(j_ref), len(k_ref))``.
    """
    # Will check how to implement mask I guess.
    B = np.exp(2*np.pi*j_ref[:,None]*k_ref[None,:]*1J/n)
    # Compute its Moore-Penrose inverse
    pinvB = np.linalg.pinv(B/n)
    return pinvB, B

def inc_rfft(fft_basis, data, mask=None):
    """Compute an incomplete FFT (Rauscher 2026).

    Parameters
    ----------
    fft_basis : ndarray
        Pseudo-inverse basis, as returned by `get_fft_basis`.
    data : ndarray
        Sampled values, of shape ``(len(j_ref),)``.
    mask : ndarray, optional
        Boolean mask of the values to ignore. Default is None.

    Returns
    -------
    ndarray
        Fourier coefficients at the frequencies ``k_ref``.
    """
    if mask is None :
        mask = np.zeros(data.shape, dtype='bool')
    return np.matmul(fft_basis, data[~mask])
