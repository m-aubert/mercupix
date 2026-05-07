import numpy as np 

# Utils func
def get_ptc(data):
    return data.mean(axis=(1,2)) , data.std(axis=(1,2), ddof=1) 


def get_psd(ramp, data=None, channel=0, frame=0, corr_refpix=True, pad=12, readout_freq=203_000): 
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
    
# Classical NL carac.
class ClassicalNL : 
    
    def __init__(self, ramp, times, degree=3):
        self.degree= degree
        self.ramp= ramp 
        self.times= times 
        
    @staticmethod
    def poly_model(x, *params, degree=3): 
        model = 0 
        for i in range(degree+1):
            if i != 1 : 
                model += params[i]*(params[1]*x)**i 
            else : 
                model += (params[1]*x)**i 
                
        return model 
    
    def get_model(self, x, *params):
        return self.poly_model(x, *params, degree=self.degree)
        
    def get_nlcorr(self, on_mean=True, init_val=None, corr_degree=4): 
        from scipy.optimize import curve_fit 

        if init_val is None : 
            init_val = [1]*(self.degree+1)
            init_val[1] = self.ramp[0].mean()/2 

        if on_mean : 
            data_vec = self.ramp.mean(axis=(1,2))
            err = self.ramp.std(axis=(1,2), ddof=1)
        else : 
            raise NotImplementedError('Fit per pixel not implemented yet')

        # Fit b+a*x + c*(a*x)**2 + .... 
        # Could probably minimize on degrees but unsure.
        # Legendre could be tested as well but unsure as well.
        p = curve_fit(self.get_model, self.times, data_vec,p0=init_val, maxfev = 10000)[0]

        linearized_data = np.poly1d(p[:2][::-1])(self.times)/data_vec  - 1      
        ps = np.polyfit(data_vec, linearized_data, corr_degree) 
        nl_coefs = ps[::-1] 
        
        return nl_coefs