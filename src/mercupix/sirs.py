import numpy as np 
from scipy.stats import median_abs_deviation
from joblib import Parallel, delayed
from .fitter import LSQFit
from .carac import get_readout_sequence
from .ramp import RawRamp


# Utils func to add refs to data -> add to Ramp
def get_data_with_ref(data, ramp_with_refpix, ramp): 
    data = np.column_stack([ramp_with_refpix[:,:4,ramp.actslice], data, ramp_with_refpix[:,-4:, ramp.actslice]])
    data = np.dstack([ramp_with_refpix[:,:,:4], data, ramp_with_refpix[:,:, -4:]])
    return data


# Training function -> will become a function of a class. (or a class method ?)
# BEWARE -- TUNED FOR IP2I FOR NOW.
def train_sirs(ramp_list, refpix=4,  nroh=12, detector_freq=200_000, njobs=5, debug=False, filenames_kwargs=dict(datatype='ip2i', 
                                     time_arr_path='/run/seq_1/env_measures')):

    results = 0
    counters = 0
    offsets = 0
    if debug : 
        delta_mu = []
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

            results += np.asarray(result_).sum(axis=0) #[result[i][0] for i in range(len(result))]) # Unpacking 
            offsets += np.asarray(counter_).sum(axis=0) ##result[i][1] for i in range(len(result))]) # Unpacking 
            counters += np.asarray(offset_).sum(axis=0) #[result[i][2] for i in range(len(result))]) # Unpacking 

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
        
    #print(results.shape)
    #Nspec, Lspec, Rspec, Xspec, Yspec, Zspec = np.swapaxes(results, 0
        
    # "Solving here"
    denom_ = results[:,2]*results[:,1] - np.real(results[:,5]*results[:,5].conjugate())
    alpha = (results[:,2]*results[:,4] - results[:,3]*results[:,5]) / denom_ #(Zspec.real**2 + Zspec.imag**2))
    beta = (results[:,3]*results[:,1] - results[:,4]*np.conj(results[:,5])) / denom_ #(Zspec.real**2 + Zspec.imag**2)) 
    
    return alpha.astype(np.complex64), beta.astype(np.complex64), fft_basis_ref, results, freqs

def save_sirs(fft_basis_ref, alpha, beta, freqs, n, results,  filepath='sirs.h5', debug=True):
    # For now h5 because easy to mix diffent dimensions. But actually not sure it is an issue tbh.
    import h5py 

    with h5py.File(filepath, 'w') as f:  
        f.create_dataset("fft_basis", data = fft_basis_ref)
        f.create_dataset("alpha_corr", data = alpha)
        f.create_dataset("beta_corr", data = beta)
        f.create_dataset("freqs", data = freqs)
        f.create_dataset("npix_readout", data = n)
        if debug : 
            f.create_dataset('Nspec', data=results[0]) #still not sure what use is Nspec tbh.
            f.create_dataset('Lspec', data=results[1])
            f.create_dataset('Rspec', data=results[2])
            f.create_dataset('Xspec', data=results[3])
            f.create_dataset('Yspec', data=results[4])
            f.create_dataset('Zspec', data=results[5])
        
    print("Saved : {filepath}")
    #return None

def process_frame(ramp, frame, good_, refpix, fft_basis_ref, pixmask, nroh=12):

    coefs_spec = []
    counter_offset = []
    
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
            counter_offset.append([offset_, counter_])

        else : 
            coefs_spec.append([np.zeros(left_fft.shape)]*6)
            counter_offset.append([offset_, counter_])
        
    return np.asarray(coefs_spec) , np.asarray(offset_), np.asarray(counter_) #Testing, not ideal tbh.
    
# To go into the SIRS class
def get_fourier_corrections(ramp, frame, channel, pixmask, good, right_fft, left_fft, 
                            nroh=12, refpix=4, GD_OP_SIG_MIN = 0, GD_OP_SIG_MAX = 30):              

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
    """
    FFT basis for incomplete rrft based on Rauscher 2026
    """    
    # Will check how to implement mask I guess.
    B = np.exp(2*np.pi*j_ref[:,None]*k_ref[None,:]*1J/n)        
    # Compute its Moore-Penrose inverse
    pinvB = np.linalg.pinv(B/n)
    return pinvB, B

def inc_rfft(fft_basis, data, mask=None):
    """ Incomplete rfft based on Rauscher 2026"""
    if mask is None : 
        mask = np.zeros(data.shape, dtype='bool')    
    return np.matmul(fft_basis, data[~mask])



