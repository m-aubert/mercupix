import numpy as np
import h5py
from astropy.io import fits
from scipy.signal import savgol_filter

class Cube : 
    """
    For now placeholder very large func. Will see how this goes. 
    -> Could be rawcube
    -> then sci cube 
    -> then something else. 
    To be discussed.

    Could be Euclid specific or Roman specific. 
    We keep best of both worlds ? 
    """
    def __init__(self, 
                 datacube=None, 
                 refpix=None, 
                 filename=None, 
                 header=None, 
                 detectype=None, 
                 scislice=slice(4,-4),
                 resolution=None, 
                 channel = None,
                grouped=False):
        
        self.datacube = datacube
        self.filename = filename
        self.detectype = detectype 
        self.scislice = scislice
        self.ismacc = False
        self.header = header
        self.refpix = refpix
        self.detectype = detectype
        self.resolution = resolution
        self.channel = channel
        self.grouped = grouped
        
    @classmethod
    def from_filename(cls,filename, debug=False):

        if 'fits' in filename : 
            with fits.open(filename) as hdul : 
                header = hdul[0].header
                cube = hdul[1].data[:,:,:-128] #Make sure this is systematically true.
                #resetframe = False if header['RESETFRM'] == 'F' else True

            detectype = 'H4'
            resolution = cube.shape[2]
            channel = 128
        elif 'h5'  in filename : 
            # Mostlikely Euclid
            detectype = 'H2'
            f = h5py.File(filename, 'r')
            list_of_names = [] #Get all tree of H5PY.
            f.visit(list_of_names.append) 
            print(list_of_names)
 
            # Match frame path
            try : 
                frame_path = [path for path in list_of_names if 'frames' in path][0]
            except Exception as e : 
                raise ValueError("frames does not exist in this file, is there something wrong ?")
            # Match time path (for time array)
            time_arr_path = [path for path in list_of_names if 'measures' in path][0]

            cube = f[frame_path][()][:,:,64:]
            tarr = f[time_arr_path]['time'] 
            header = dict(texp = tarr.max(), 
                          tframe=np.diff(tarr).mean())
            resolution = cube.shape[2]
            channel = 64

            
        else : 
            raise(NotImplementedError('Other type of HxRG are not yet implemented'))

        return cls(datacube=cube, 
                   refpix=None, 
                   filename=filename, 
                   header=header, 
                   detectype=detectype,
                   resolution=resolution, 
                   channel=channel) 

    def get_refpix(self, saverawref=True): 
        #Depends on detector type I guess (HxRG) - H2 or H4.
        # Ugly func. Will need to be clearer.
        if saverawref: 
            self.ud_refpix = np.concatenate([self.datacube[:,:4,:],
                                              self.datacube[:,-4:,:]], 
                                              axis=1).copy()
            
            self.lr_refpix = np.concatenate([self.datacube[:,:,:4], 
                                             self.datacube[:,:,-4:]], 
                                             axis=1).copy()

        UD = self.ud_refpix.copy()
        UD = np.median(UD.reshape(self.nframes, 8, self.resolution//self.channel,self.channel),
                       axis=(1,3)) #Hard-coded no good.
        UD = np.repeat(UD, self.channel, axis=1)
        
        LR = np.concatenate([(self.datacube-UD[:,None,:])[:,4:-4,:4], 
                             (self.datacube-UD[:,None,:])[:,4:-4,-4:]], 
                             axis=2).mean(axis=2)
        LR = np.asarray(list(map(lambda x : savgol_filter(x,
                                                           window_length=112, 
                                                           polyorder=4), LR)))
        self.refpix_map = UD[:,None,:]+LR[:,:,None]
        #There's probably a more efficient way to do the stuff I guess / less costly.

    def corr_refpix(self): 
        if not hasattr(self, 'refpix_map'):
            self.get_refpix()
        corr_data = self.get_scicube() - self.refpix_map[:,:,self.scislice]
        return corr_data
    
    def get_scicube(self):
        return self.datacube[:,self.scislice, self.scislice]
        
    @property
    def cube_shape(self):
        return (self.datacube.shape[1], self.datacube.shape[2])

    @property
    def sci_shape(self):
        return (self.datacube.shape[1], self.datacube.shape[2])

    @property
    def nframes(self):
        return self.ngroups * self.nfpgroups

    @property
    def texp(self):
        if not hasattr(self, '_texp'):
            self._texp = self.header['texp'] #Might not work for H4RG
        return self._texp
    
    @property
    def t_read(self):
        if not hasattr(self, '_t_read'):
            self._t_read = np.linspace(0,self.texp , num=self.nframes)
        return self._t_read

    @property
    def ngroups(self):
        return self.datacube.shape[0]

    @property
    def nfpgroups(self):
        if not self.grouped : 
            return 1
        else :
            return self.framespgp

    @property
    def ndropr(self):
        pass

    @property   
    def nrframes(self):
        pass
    
    
from scipy.special import legendre, hermitenorm


def poly_(x,deg=1, poly='mono'):
    if poly == "leg" : 
        pol = np.stack([legendre(i)(x) for i in range(deg+1)])
    elif poly == "mono" : 
        pol = np.stack([x**i for i in range(deg+1)])
    elif poly == "herm": 
        pol = np.stack([hermitenorm(i)(x) for i in range(deg+1)])
    else : 
        return None
    return pol.astype(np.float32)
    

class Rampfit: 
    """
    Should be a class dedicated to fitting stuff. Will see how this goes I guess. 
    """
    def __init__(self, cube, polytype, tarr):
        self.cube = cube
        self.polytype = polytype
        self.tarr = tarr
        if self.polytype == 'leg':
            self.tleg = tarr/tarr.max() * 2  - 1 #Pour legrendre Leander ver. -> assume t0 = 0 .

    def poly_(self, deg=1, leg=True):
        from scipy.special import legendre 
        if leg : 
            pol = np.stack([legendre(i)(self.tleg) for i in range(deg+1)])
        else : 
            pol = np.stack([self.tarr**i for i in range(deg+1)])
        return pol.astype(np.float32)

    def no_w_lstq(self, x, cube):
        cube = cube.reshape(cube.shape[0], -1).astype(np.float32)
        self.coeffs = np.linalg.lstsq(x.T, cube, rcond=None)[0]

    def fit(self, deg=1):
        x = self.poly_(deg=deg, leg=(self.polytype=='leg'))
        self.no_w_lstq(x, self.cube) 
        return self.coeffs
        
class SinglePixelRamp:
        def __init__(self, pixramp, tread):
             self.pixramp = pixramp
             self.tread = tread

        def linearize_single_pixel(self, coef, deg=3, poly='mono'): 
            lin_meas = 1 - self.pixramp / np.sum(poly_(self.tread, deg=1, poly=poly).T*coef[:2], axis=-1)
            coef_lin = np.linalg.lstsq(poly_(self.pixramp, deg=deg, poly=poly).T, lin_meas.astype(np.float32), rcond=None)[0]
            pixlin = self.pixramp * (1+(coef_lin[:,None]*poly_(self.pixramp, deg=deg, poly=poly)).sum(axis=0))
            return pixlin 
        
        def ramp_lstq(self, ramp, deg=1, poly='mono'):
            if poly == 'leg': 
                x= 2/self.tread[-1]*self.tread - 1.
            else: 
                x = self.tread
            x = poly_(x, deg=deg, poly=poly)
            coefs = np.linalg.lstsq(x.T, ramp, rcond=None)[0]
            return coefs
        
        def ramp_to_pixelflux(self, deg=1, poly='mono'):
            coefs = self.ramp_lstq(self.pixramp, deg=deg, poly=poly)
            pixlin = self.linearize_single_pixel(coefs, deg=deg, poly=poly)
            coef_lin = self.ramp_lstq(pixlin,deg=1, poly='mono')
            return coef_lin
        
class Kubik2015 : 

    def __init__(self,ramp, tpf, macc=None):
        self.diff = np.diff(ramp, axis=0)
        self.tpf = tpf
        if macc is not None : 
            self.ng, self.nf, self.nd =macc
        else :
            self.ng = ramp.shape[0]
            self.nf = 1
            self.nd = 0

    @property
    def alpha(self): 
        return (1-self.nf**2)/(3*self.nf+self.nd)
    
    def beta(self, sigma_r, fe):
        return 2*sigma_r**2 * fe / (self.nf * (1+self.alpha))
    
    def estim_f(self,fe, sigma_r):
        alpha_= self.alpha
        beta_ = self.beta(sigma_r, fe)

        term = (1+alpha_)/2*fe
        term *= np.sqrt(1 + 4*fe**2 * np.sum( (self.diff + beta_)**2) / 
                        ((self.ng+1)*(alpha_+1)**2 )) - 1
        term -= beta_
        return term
    
    def estim_sigf(self,sigma_r, fe):
        alpha_ = self.alpha
        beta_ = self.beta(sigma_r,fe)
        g_ = self.estim_f(fe, sigma_r)
        term  = 2*(g_+beta_)**2 /  ((self.ng-1)*
                                    (1+2*fe/(1+alpha_) * 
                                     (g_+beta_)))
        
        return term




class Brandt2024(Fitter):    
    """
        Method to fit flux based on Brandt 2024. Assumes data are grouped.
    """
    def __init__(self, data, times=None,  readnoise=11, timeframe=None,
                 add_reset=False, countrateguess=None, reset=False, nframes_list = []):
        super().__init__(data, times=times)
        
        self.readnoise = np.atleast_1d(readnoise)     
        self.ti = times
                
        # Add support for masked array   
        self.ismasked = True if isinstance(data, np.ma.MaskedArray) else False         
                
        self.diffti = np.diff(times)
        self.diffgp = np.diff(data,axis=0)      
                
        if len(nframes_list) == 0 : 
            self.Nis = np.ones(data.shape[0])
        else:     
            self.Nis = nframes_list
    
        self.tauis = np.sum( [ (2*ni - 2*i + 1)*tk for i, (ni,tk) in enumerate(zip(self.Nis, timeframe))])     
        self.tauis *=1/self.Nis**2
        self.vari = self.readnoise**2/self.Nis   
        
        self.reset = reset
        self.add_reset = add_reset if self.reset else False
        
        if self.add_reset : 
            append = np.ma.append if isinstance(data, np.ma.MaskedArray) else np.append        

            self.diffgp = append(([data[0]/self.ti[0]]), self.diffgp, axis=0)
            self.diffti = append(np.array([self.ti[0]]) , self.diffti, axis=0)

        self.diffd = (self.diffgp/self.diffti[:,None,None]).reshape(self.diffgp.shape[0], -1)
        
        self.ndiff = self.diffgp.shape[0]
        self.npix = self.diffd.shape[1:]
        self.sign = np.asarray((-1)**np.arange(1,self.ndiff+1))

        if countrateguess is None : 
            if self.add_reset: 
                countrateguess = self.diffd[1:].mean(axis=0)
            else : 
                countrateguess = self.diffd.mean(axis=0)
                
        self.alpha, self.beta = self.get_covar(countrateguess)
                
    def fit(self, return_qf=True, ktc_prior=None, inplace=True): 
        """Fit method.

        Returns:
            tuple of ndarray : slope and variance of the considered ramp.
        """
        coefs, cov_ab, qf = self._get_coefs(return_qf=return_qf, 
                                    ktc_prior=ktc_prior, 
                                    intercept=self.reset)
        
        coefs = [coi.reshape(self.diffgp.shape[1:]) for coi in coefs]
        
        if inplace :
            self.coefs = coefs
        
        if hasattr(qf, 'shape'):
            qf = qf.reshape(self.diffgp.shape[1:])
                        
        return coefs, cov_ab, qf
    
    def get_model(self, pixels=None, coefs=None, intercept=False, coefid=0):
        # Here linear model
        if coefs is None : 
            coefs = self.coefs
               
        if len(coefs) > 1 and intercept :    
            # Simple A*t + b model               
            model = coefs[0]*self.ti[:,None,None] + coefs[1]        
        else:
            # Simple A*t + b                
            model = coefs[coefid]*self.ti[:,None,None] 

        if pixels : 
            xpixel , ypixel = pixels
            model = model[:, xpixel, ypixel]
            
        return model
        
    def get_covar(self, a) : 
        """Define group difference covariance matrix.

        Returns:
            tuple of ndarray : the diagonal and off-diagonal components of the covariance.
        """
        ndim = a.ndim if hasattr(a , 'ndim') else 1

        lalpha_diag = (self.vari[1:]+self.vari[:-1])/self.diffti**2  
        ralpha_diag = 1/(self.diffti**2) * (self.tauis[1:]+self.tauis[:-1] - 2*self.ti[:-1])

        if lalpha_diag.ndim <= ndim : 
            lalpha_diag = np.expand_dims(lalpha_diag, list(range(1,ndim+1)))
            ralpha_diag = np.expand_dims(ralpha_diag, list(range(1,ndim+1)))
            
        alpha_diag = lalpha_diag + a*ralpha_diag
        
        lbeta_diag = (-self.vari[1:-1])/(self.diffti[1:]*self.diffti[:-1])
        rbeta_diag = 1/(self.diffti[1:]*self.diffti[:-1]) * ( self.ti[1:-1] -self.tauis[1:-1])

        if lbeta_diag.ndim <= ndim : 
            lbeta_diag = np.expand_dims(lbeta_diag, list(range(1,ndim+1)))
            rbeta_diag = np.expand_dims(rbeta_diag, list(range(1,ndim+1)))
            
        beta_diag = lbeta_diag + a*rbeta_diag
            
        if self.reset : 
            append = np.ma.append if self.ismasked else np.append

            a0 = np.expand_dims((self.vari[0] + a*self.tauis[0])/ self.ti[0]**2, list(range(1,ndim+1)))
            b0 = np.expand_dims((self.vari[0] + a*(self.ti[0] - self.tauis[0]))/(self.ti[0]*self.diffti[0]),list(range(1,ndim+1)))
            
            alpha_diag = append(a0, alpha_diag, axis=0)
            beta_diag = append(b0, beta_diag, axis=0)
            
        return alpha_diag, beta_diag
        
    def _get_coefs(self, return_qf = True, intercept=True, ktc_prior=None, jumps=False):
        """Estimator for the slope of the ramp.

        Args:
            return_qf (bool, optional): Whether to return a quality of the fit estimator. Defaults to True.

        Returns:
            tuple of ndarray : slope, slope variance, if return_qf is True, chi2.
        """
                 
        B = np.sum(self.diffd * self.sign[:,None] / self.theta[-1] * (self.phi[1:]*self.Theta+ self.theta[:-1]*self.Phi), axis=0)        
        C = np.sum(self.sign[:,None]/self.theta[-1] * (self.phi[1:]*self.Theta+self.theta[:-1]*self.Phi), axis=0)
        
        A = 0.
        chi2=0.
        if return_qf : 
            A = 2*np.sum(self.diffd * self.sign[:,None]/self.theta[-1] * self.phi[1:] * self.ThetaD[:-1], axis=0)
            A += np.sum((self.diffd)**2 * self.theta[:-1] * self.phi[1:] / self.theta[-1], axis=0)
            
        if not intercept: 
            coefs = [B/C] 
            cov_ab = [1./C]
            if return_qf :
                chi2 = A - B**2/C
                    
        else : 
            coefs, cov_ab, chi2 = self._get_slope_intercept(A, B,C, 
                                                            ktc_prior=ktc_prior, 
                                                            return_qf=return_qf)
                                                                                                      
        return coefs, cov_ab, chi2      
    
    def _get_slope_jumps(self, A, B,C, return_qf=True, jumps=False):
        
        if not jumps : 
            coefs = [B/C] 
            cov_ab = [1./C]
            chi2=0
            if return_qf :
                chi2 = A - B**2/C
            
        else : 
            pass
            # two steps : 
            
            # one-omit 
            
            
            # two-omit 
            
            
        return coefs, cov_ab, chi2  
        
    
    def _get_slope_intercept(self, A, B,C, ktc_prior=None, return_qf=True): 
        if ktc_prior is not None :  
            z, zerr = ktc_prior
            z_term1 = z/zerr**2
            z_term2 = 1/zerr**2
            
        else : 
            z_term1 = 0
            z_term2 = 0
    
        b = (self.Bp1 / self.ti[1] - self.Cp1*B / (C*self.ti[1]) + z_term1)
        b /= (self.C11/self.ti[1]**2 - self.Cp1**2/ (C*self.ti[1]) + z_term2) 
        
        a = B/C - b*(self.Cp1*B / (C*self.ti[1]))
                
        coefs = [a, b]
        array_ = np.ma.array if self.ismasked else np.array
        cov_ab = array_([[C , self.Cp1 / self.ti[1]], [self.Cp1 / self.ti[1], self.C11/self.ti[1]**2 + z_term2]])

        chi2 = 0
        if return_qf : 
            chi2 = A + a**2*C + b**2/self.ti[1]**2 * self.C11 - 2*b/self.ti[1] * self.Bp1 
            chi2 += -2*a*B + 2*a*b/self.ti[1]*self.Cp1 
            if z_term2 != 0:
                chi2 += (b-z)**2 / zerr
            
        return coefs, cov_ab, chi2     
    
                                
    @property
    def Cd(self):
        """ Diagonal component of the covariance"""
        return self.theta[:-1]/self.theta[self.ndiff] * self.phi[1:]
    
    @property
    def Cp1(self):
        #return self.sign/self.theta[self.ndiff] * (self.phi[2]*self.Theta[1] + self.theta[0]*self.Phi[1])
        return self.sign/self.theta[self.ndiff] * (self.phi[1:]*self.Theta + self.theta[:-1]*self.Phi)
    
    @property
    def Bp1(self):
        #return -1/self.theta[-1] * (self.phi[2]*self.ThetaD[1] + self.theta[0]*self.PhiD[1])
        return self.sign/self.theta[self.ndiff] * (self.phi[1:]*self.ThetaD[1:] + self.theta[:-1]*self.PhiD)

    @property
    def Cod(self):
        """ Off-diagonal component """
        return -self.beta * self.theta[:-2] * self.phi[2:] / self.theta[self.ndiff]

    @property
    def PhiD(self):
        if not hasattr(self, '_PhiD'):
            PhiD = np.zeros((self.ndiff, *self.npix))     
            for i in range(self.ndiff - 2, -1, -1):
                PhiD[i] = (PhiD[i+1] + self.sign[i+1] * self.diffd[i+1] * self.phi[i+2]) * self.beta[i] 
            self._PhiD = PhiD
        return self._PhiD

    @property
    def theta(self):
        if not hasattr(self, '_theta'): 
            theta = np.ones((self.ndiff+1, *self.npix))    
                 
            theta[1] = self.alpha[0]
            for i in range(2,self.ndiff+1):
                theta[i] = self.alpha[i-1] * theta[i-1] - self.beta[i-2]**2 * theta[i-2]
                
            self._theta = theta
        
        return self._theta
    
    @property
    def phi(self):
        if not hasattr(self, '_phi'):
            phi = np.ones((self.ndiff+1, *self.npix))     
            phi[self.ndiff-1] = self.alpha[self.ndiff-1] 
            for i in range(self.ndiff-2, -1, -1):
                phi[i] = self.alpha[i] * phi[i+1] - self.beta[i]**2 * phi[i+2]
         
            self._phi = phi
        
        return self._phi    
    
    @property
    def Phi(self):
        if not hasattr(self, '_Phi'):
            Phi = np.zeros((self.ndiff, *self.npix))     
            for i in range(self.ndiff-2, -1, -1):
                Phi[i] = self.beta[i]*Phi[i+1] + self.sign[i+1]*self.beta[i]*self.phi[i+2]
                
            self._Phi = Phi
            
        return self._Phi
    
    @property
    def Theta(self):
        if not hasattr(self, '_Theta'):
            Theta = np.zeros((self.ndiff, *self.npix))            
            Theta[0] = -self.theta[0]
            
            for i in range(1,self.ndiff):
                Theta[i] = self.beta[i-1]*Theta[i-1] + self.sign[i]*self.theta[i]
                
            self._Theta = Theta
            
        return self._Theta
            
    @property
    def ThetaD(self):
        if not hasattr(self, '_ThetaD'): 
            ThetaD = np.zeros((self.ndiff+1, *self.npix))
                             
            ThetaD[1] = -self.diffd[0] * self.theta[0]
            for i in range(1,self.ndiff):
                ThetaD[i+1] = self.beta[i-1]*ThetaD[i] + self.sign[i]*self.diffd[i]*self.theta[i]
                
            self._ThetaD = ThetaD
            
        return self._ThetaD
  
  
  
  
  
  
  
  
  ###### SAND BOX SIRS - refpix correction 
  
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

def irrf(fft_basis, data, mask=None):
    return np.matmul(pinvB, data[~mask])


# Missing sutff: 


ramp = RawRamp.from_filename(dark_data)
rampdata, _, mask = ramp.preprocess_data(**kwargs)


#Fit procedure

fitter = LSQFit(data=rampdata, times=ramp.times)
coefs = fitter.fit(inplace=False)
fitter.get_residuals(coefs=coefs)
#Get residuals.


#Estimate fft_basis

#Beginning of procedure. 
GD_OP_SIG_MIN = 0
GD_OP_SIG_MAX = 28

fft_basis_ref = get_fft_basis(n) # Should not have any impact for references ? 
fft_basis_dat = get_fft_basis(n, mask=mask) # For comparison.

refpix=4
nroh = 12 #or 7 ?

pixmask = np.zeros(ramp.frame_shape, dtype=bool)
regular[ramp.actslice, ramp.actslice] = True
# From mask
for frame, good in (residuals, mask) : 

    frame = frame
    
    #frame = delta[z]
    left_fft = inc_rfft(fft_basis_ref,frame[:refpix, :].reshape(-1))
    right_fft = inc_rfft(fft_basis_ref,frame[-refpix:, :].reshape(-1))

    for channel in range(nchannel):
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
        sc.Nspec[:, op] += np.real(keep * np.conj(keep))
        sc.Lspec[:, op] += np.real(left_fft *np.conj(left_fft))
        sc.Rspec[:, op] += np.real(right_fft *np.conj(right_fft))
        sc.Xspec[:, op] += (keep *np.conj(right_fft))
        sc.Yspec[:, op] += (keep *np.conj(left_fft))
        sc.Zspec[:, op] += (right_fft *np.conj(left_fft))
        sc.offset[op] += (mu - mu_ref)
        sc.counter[op] += 1