import numpy as np 
from .fitter import LSQFit
from .carac import get_channel_slices, get_readout_sequence
  
def train_sirs(dark_data, detector_freq=200_000):
      # Missing sutff: 

    ramp = RawRamp.from_filename(dark_data)
    rampdata, _, mask = ramp.preprocess_data(corr_refpix=False, mask=False)

    #Fit procedure
    fitter = LSQFit(data=rampdata, times=ramp.times)
    coefs = fitter.fit(inplace=False)
    residuals = fitter.get_residuals(coefs=coefs)
    #Get residuals.
    
    del coefs, rampdata
    
    n = np.fft.rfftfreq(n, d=1/detector_freq)
    
    #Beginning of procedure. 
    GD_OP_SIG_MIN = 0
    GD_OP_SIG_MAX = 28
    # There probably is something to do

    fft_basis_ref = get_fft_basis(n) # Should not have any impact for references ? 
    #fft_basis_dat = get_fft_basis(n, mask=mask) # For comparison.

    refpix=4 # Get from ramp
    nroh = 12 #or 7 ? Clarify here ?

    pixmask = np.zeros(ramp.frame_shape, dtype=bool)
    regular[ramp.actslice, ramp.actslice] = True

    # From mask
    for frame, good in (residuals, mask) : 

        frame = frame
        
        left_fft = inc_rfft(fft_basis_ref,frame[:refpix, :].reshape(-1))
        right_fft = inc_rfft(fft_basis_ref,frame[-refpix:, :].reshape(-1))

        for channel in range(ramp.channel):
            # need channel size
            slice_col, slice_row = get_channel_slices(ramp, channel=channel)

            d = get_readout_sequence(ramp,data=frame, channel=channel, corr_refpix=False, pad=nroh)

            # Unclear if zis step iz necezarry
            good = get_readout_sequence(ramp, data=mask, channel=channel, corr_refpix=False, pad=nroh)

            mask_edge = get_readout_sequence(ramp, data=pixmask, channel=channel, corr_refpix=False, pad=nroh)

            regular = d[mask_edge]
            valid = pixmask[mask_edge]

            # Clipping whacky data I guess. 5% is a lot though.
            vals = regular[valid] # Keep only OK data.
            vals = np.sort(vals) #not sure why
            minimum, maximum = np.percentile(vals,[5//2,100-5//2]) #5//2% and 95//2% as min and max.
            valid[regular <= minimum] = False
            valid[regular >= maximum] = False

            # Add that to mask I guess.
            good[mask_edge] = valid

            # Quality stuff. Skip bad frames. But weeeeird.
            region = d[ramp.actslice, :ramp.npix_per_channel]
            mu = np.mean(region[good[ramp.actslice,:ramp.npix_per_channel]])
            sigma = np.std(region[good[ramp.actslice,,:ramp.npix_per_channel], ddof=1)

            ref = d[mask_edge][-refpix:].reshape(-1)
            rmin, rmax = np.percentile(ref.reshape(-1),[5//2,95//2]) #5% and 95% as min and max.
            mu_ref = np.mean(ref[(ref > rmin) & (ref < rmax)])

            # Remplacement des pixels invalides
            # Will see how to rewrite to remove loop if needed.
            flag_pix = np.all(good, axis=0)
            for i in range(good.shape[1]):
                if flag_pix[i]: 
                    continue

                row_m =  np.mean(row[good[:, i]]) #Should be median ?
                d[:,i][~good[:, i]] = row

            # Colonnes d'overhead
            roi = d[-2*nroh:-nroh] # Mirror data before overhead I gues
            d[nroh:] = np.flipud(roi) #Restore order because why not
            vector = d.reshape(-1)
            
            spectrum = np.fft.rfft(vector) #

            keep = np.concatenate((spectrum[:ramp.frame_shape[0] // 2 + 1], 
                                spectrum[-(ramp.frame_shape[0] // 2 - 1):]))


            # This is a part to validate for example.
            Nspec[:, op] += np.real(keep * np.conj(keep))
            Lspec[:, op] += np.real(left_fft *np.conj(left_fft))
            Rspec[:, op] += np.real(right_fft *np.conj(right_fft))
            Xspec[:, op] += (keep *np.conj(right_fft))
            Yspec[:, op] += (keep *np.conj(left_fft))
            Zspec[:, op] += (right_fft *np.conj(left_fft))
            offset[op] += (mu - mu_ref)
            counter[op] += 1
    
    # Output basis here: 
    # Sum stuff & get alpha and beta.
    # Might be something worth to look into adding other ref pixels.
    
    alpha = 0
    beta = 0
    
    return alpha, beta
    
  # This is utiliy func for sirs.
def get_fft_basis(n, n_keep=8, mask=None):
    """
    Heavily inspired from SIRS julia implementation of Rauscher.
    """
    if mask == None : 
        mask = np.zeros(n, dtype='bool')    
    
    # Create vectors of j and k values for Eq. 2
    j = np.arange(n)[~mask]                         # J values for all the good samples
    k = np.arange(n//2+1)                          # All frequencies
    k = np.concatenate((k[:n_keep],k[-n_keep:]))   # Only those for the lowest and highest
    
    # An empty B matrix
    B = np.empty((np.sum(mask),2*n_keep), dtype=np.complex64)
    
    # Fill in B with loop because badpix and all
    for _j in np.arange(len(j)):
        for _k in np.arange(len(k)):
            B[_j,_k] = np.exp(2*np.pi*j[_j]*k[_k]*1J/n)
            
    # Compute its Moore-Penrose inverse
    pinvB = np.linalg.pinv(B)

    return pinvB

def inc_rfft(fft_basis, data, mask=None):
    return np.matmul(fft_basis, data[~mask])
