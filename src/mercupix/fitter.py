import numpy as np
import warnings

# High level function
def fit_ramp(rawramp,  method='LSQ', 
                       persistence=False, 
                       bfe=False,
                       crnl=False,
                       raw_ramp_kwargs={},
                       fit_args={},
                       **kwargs): 
    """
    High-level function to fit ramp

    Parameters
    ----------
    datacube : ndarray
        (N, nact, nact) datacube
    varcube : ndarray, optional
        Variance cube of datacube (same shape), by default None
    time_frames : ndarray-like, optional
        time per frames, by default None
    time_group : ndarray-like, optional
        time per groups if datacube is grouped, by default None
    method : str, optional
        Fitting method. Implemented are 'LSQ', 'Kubik', 'Brandt', by default 'LSQ'
    fit_args : dict, optional
        keyword-arguments of the fitter method, by default {}
    persistence : bool, optional
        Accounts for persistence effects in fit, by default False
    bfe : bool, optional
        Accounts for bright-fatter effect in fit, by default False
    crnl : bool, optional
        Accounts for count-rate non-linearity, by default False
    cosmics : bool, optional 

    Returns
    -------
    array-like or tuple of ndarray-like
        Output of the fitting procedure : fitted slope, variance and QF (chi2 or else).
    """    
    if time_group is None and time_frames is not None :
        time_group = time_frames
        
    try : 
        fitter = model_register['method'].from_rawramp(rawramp, 
                                                       ramp_kwargs=raw_ramp_kwargs,
                                                       **kwargs)
        
    except KeyError as ke :
        raise KeyError("Fitter {method} does not exist")
         
    fitted_slope = fitter.fit(**fit_args)
    
    if method == 'LSQ': 
       _ , fitted_slope, _ =  fitter.parse_coefs()
    
    elif method == 'Kubik' or method == 'Brandt' : 
        fitted_slope = fitted_slope[0][0]

    return fitted_slope 


## General Fitter level
class Fitter(): 
    """ 
    Baseline Fitter Class for models.
    """
    def __init__(self, data, times=None):
        """ """
        self._data = data
        if times is None:
            times = np.arange(data.shape[0])
                
    @classmethod
    def from_rawramp(cls, rawramp, ramp_kwargs=dict(), **kwargs):
        """ """
        datacube, varcube, mask = rawramp.preprocess_ramp(**ramp_kwargs)
        return cls(data=datacube, times=rawramp.times, **kwargs)
        
    def get_model(self, pixels=None, coefs=None):
        """ """
        return None
        
    def get_residuals(self, data=None, pixels=None, coefs=None, model=None):
        """ """
        if model is None:
            model = self.get_model(pixels=pixels, coefs=coefs)
            
        if data is None : 
            data = self.data 
            
        if pixels is None:
            residuals = data - model
        else:
            xpixel, ypixel = pixels
            residuals = data[:, xpixel, ypixel] - model
            
        return residuals
    
    def fit(self, data=None, inplace=False):
        """ """
        return None

    # ================ #
    #   properties     #
    # ================ #    
    @property
    def data(self):
        """ datacube to be fitted """
        return self._data

    @property
    def times(self):
        """ """
        return self._times

    @property
    def _cached_fitted_times(self):
        """ """
        if not hasattr(self, "_hcached_fitted_times") or self._hcached_fitted_times is None:
            self._hcached_fitted_times = self.get_times()
            
        return self._hcached_fitted_times


class LSQFit(Fitter): 
    """ Uses least-square methods to evaluate flux."""
    def __init__(self, data, times=None, 
                 degree=5, polytype='leg'):
        """ """        
        super().__init__(data, times=times)

        self.set_times(times)
        self.set_polynome(polytype=polytype, degree=degree)

    # ================ #
    #   method         #
    # ================ #    
    # ------- #
    # SETTER  # 
    # ------- #
    def set_polynome(self, degree, polytype="legendre", load=False):
        """ """
        
        self._polynome = {"type": polytype, 
                          "degree": degree, 
                          "array": None,
                          "coefs": None,
                         }            
        self._hcached_fitted_times = None
        
    def set_times(self, times):
        """ """
        self._times = np.asarray(times, dtype="float32")
        self._hcached_fitted_times = None

    # ------- #
    # GETTER  # 
    # ------- #
    def get_times(self, as_fitted=True, polytype=None):
        """ """
        if not as_fitted:
            return self.times

        if polytype is None:
            polytype = self.polynome_type

        return self._times_to_polynomex(times=self.times, polytype=polytype)
    
    def fit(self, data=None, inplace=False):
        """ """
        if data is None:
            data = self.data
            
        coefs = self._fit_polynome_on_data(polynome=self.polynome_basis,
                                            data=data)
        if inplace:
            self._polynome["coefs"] = coefs

        return coefs


    def scan_fit(self, data=None, progress_bar=False,
                     frame_range=[4, None],
                     x_range=None, y_range=None):
        """ """
        if progress_bar:
            from tqdm import tqdm
            
        if data is None:
            data = self.data

        if x_range is not None:
            print("x_range")
            data = data[:, :, x_range[0]:x_range[1]]

        if y_range is not None:
            print("y_range")
            data = data[:, y_range[0]:y_range[1], :]
            
        # parse input
        firstframe, lastframe = frame_range
        if firstframe is None:
            firstframe = self.polynome_degree+1
            
        elif firstframe < self.polynome_degree:
            warnings.warn("firstframe is lower than the number of degree. Issues are expected.")
        
        if lastframe is None:
            lastframe = data.shape[0]
        elif lastframe < 0:
            lastframe = data.shape[0] + lastframe +1
            
        # ---- #
        xframes = np.arange(*frame_range)
        coefs = []
        times = []
        for lastframe in xframes if not progress_bar else tqdm(xframes):
            this_data = data[:lastframe]
            this_times = self.times[:lastframe]
            
            # this polynome case:
            polynome = self._polynome_of_times(this_times,
                                                 degree=self.polynome_degree,
                                                 polytype=self.polynome_type)
    
            coefs.append( self._fit_polynome_on_data(polynome, data=this_data) )
            times.append( this_times.max() )
            
        return xframes, np.asarray(times), np.asarray(coefs)

    def get_model(self, pixels=None, coefs=None):
        """ """
        if coefs is None:
            coefs = self.polynome_coefs

        if pixels is None:
            model = np.einsum('ij,jkl->ikl', self.polynome_basis.T, coefs)
        else:
            xpixel, ypixel = pixels
            model = self.polynome_basis.T @ coefs[:, xpixel, ypixel]
            
        return model
        
    def parse_coefs(self, coefs=None, times=None, axis=None, in_adu=False):
        """ """
        if coefs is None:
            if self.polynome_coefs is None:
                raise ValueError("no coefs given, none stored.")
                
            coefs = self.polynome_coefs
            
        if axis is None:
            # fetch which axis corresponds that of the degree:
            index = np.argwhere(np.asarray(coefs.shape) == self.polynome_degree+1).squeeze()
            if index.ndim != 0: # mean int
                raise ValueError("No dimension of coefs matches that of the polynome degree. "+\
                                 "Please set the degree axis")
            else:
                axis = int(index)
    
        # split the coefficients per interest
        offset, slope, nlinearities = np.split(coefs, [1, 2], axis=axis)
        offset = offset.squeeze()
        slope = slope.squeeze()
    
        if in_adu:
            if self.polynome_type in ["leg", "legendre", "herm", "hermite"]:
                # 
                offset -= slope

                
                if times is None:
                    times = self.times
                    
                if slope.ndim > 2 and times.ndim != slope.ndim:
                    missing_dims = slope.ndim - times.ndim
                    times = times.reshape([times.shape[0] ]+ [1, ]*missing_dims)
                        
                elif slope.ndim == 2:
                    times = np.float32(self.times.max())
                else:
                    # let's assume user know what they are doing.
                    pass
                    
                # broadcast.
                slope = slope * 2/times
    
        return offset, slope, nlinearities
    
    # ------------- #
    #   internal    #
    # ------------- #
    @staticmethod
    def _times_to_polynomex(times, polytype):
        """ """
        # careful, -1 must corresponds to t=0 so tmin=0, *not* np.min(times)
        if polytype in ["leg", "legendre", "herm", "hermite"]:
            tmin, tmax = 0, np.max(times)
            times = ((times-tmin) / (tmax-tmin))*2 -1
            
        return times

    @staticmethod
    def _fit_polynome_on_data(polynome, data):
        """ """
        coefs = np.linalg.lstsq(polynome.T, 
                                data.reshape(data.shape[0], -1), 
                                rcond=None)[0]
        # reshaped back as data.
        return coefs.reshape(coefs.shape[0], *data.shape[1:])

    @staticmethod
    def _get_polynome(degree, polytype, xnormed):
        """ """
        if polytype in ["leg", "legendre"]: 
            from scipy.special import legendre
            pol = np.stack([legendre(i)(xnormed) for i in range(degree+1)])
                
        elif polytype in ["herm", "hermite"]: 
            from scipy.special import hermitenorm
            pol = np.stack([hermitenorm(i)(xnormed) for i in range(degree+1)])
            
        elif polytype in ["mono", "monomial"]:
            pol = np.stack([xnormed**i for i in range(degree+1)])

        else: 
            raise NotImplementedError("Unknown {polytype=} (legendre or hermite implemented)")

        return pol.astype("float32")
    
    @classmethod
    def _polynome_of_times(cls, times, degree, polytype):
        """ """
        times = cls._times_to_polynomex(times, polytype=polytype)
        return cls._get_polynome(degree=degree,
                                 polytype=polytype,
                                 xnormed=times)
        
    # ================ #
    #   properties     #
    # ================ #    
    @property
    def data(self):
        """ datacube to be fitted """
        return self._data

    @property
    def times(self):
        """ """
        return self._times

    @property
    def _cached_fitted_times(self):
        """ """
        if not hasattr(self, "_hcached_fitted_times") or self._hcached_fitted_times is None:
            self._hcached_fitted_times = self.get_times()
            
        return self._hcached_fitted_times

    #  polynome
    @property
    def polynome_type(self):
        """ kind of polynome """
        return self._polynome.get("type")

    @property
    def polynome_degree(self):
        """ kind of polynome """
        return self._polynome.get("degree")    

    @property
    def polynome_basis(self):
        """ kind of polynome """
        if self._polynome.get("array", None) is None:
            self._polynome["array"] = self._get_polynome(degree=self.polynome_degree,
                                                         polytype=self.polynome_type,
                                                         xnormed=self._cached_fitted_times)
        return self._polynome["array"]

    @property
    def polynome_coefs(self):
        """ """
        return self._polynome.get("coefs", None)
  
class Brandt2024(Fitter):    
    """
        Method to fit flux based on Brandt 2024. Assumes data are grouped.
    """
    def __init__(self, data, times=None,  readnoise=11, timeframe=None,
                 add_reset=False, countrateguess=None, reset=False, nframes_list = [], rescale=True):
        super().__init__(data, times=times)
        
        self.readnoise = np.atleast_1d(readnoise)     
        self.ti = times
                
        # Add support for masked array   
        self.ismasked = True if isinstance(data, np.ma.MaskedArray) else False         
                
        self.diffti = np.diff(times)
        self.diffgp = np.diff(data,axis=0)      
                
        if len(nframes_list) == 0 : 
            self.Nis = np.ones(data.shape[0])
            self.tauis = times #np.sum( [ (2*ni - 2*i + 1)*tk for i, (ni,tk) in enumerate(zip(self.Nis, timeframe))])     
            self.tauis *=1/self.Nis**2
            self.vari = self.readnoise**2/self.Nis   
            
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

        if rescale : 
            self.scale = self.alpha[0] 

            self.alpha = self.alpha/self.scale
            self.beta = self.beta/self.scale

        else : 
            self.scale = 1
                
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
        
        if return_qf : 
            # Follows equation 60 and not 61 because beta forgotten.
            A = 2*np.sum(self.diffd[1:] * self.sign[1:,None]/self.theta[-1] * self.phi[2:] * self.ThetaD[1:-1] * self.beta, axis=0) 
            A += 2*(self.diffd[0] * self.sign[0]/self.theta[-1] * self.phi[1] * self.ThetaD[0]) # beta0 = 1
            A += np.sum((self.diffd)**2 * self.theta[:-1] * self.phi[1:] / self.theta[-1], axis=0)
        
        else : 
            A = 0 
        
        if not intercept: 
            coefs = [(B/C).astype('float32')] 
            cov_ab = [(self.scale/C).astype('float32')]
            if return_qf :
                chi2 = ((A - B**2/C)/self.scale).astype('float32')
                    
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
  

class Kubik2015(Fitter): 
    """
    Method based on Kubik2016 to estimate flux from group-difference images
    """
    def __init__(self, data, times=None, macc=None):
        """Kubik 2015 flux estimator. Uses the group difference.

        Args:
            ramp (ndarray): Datacube to consider (N, M, M) 
            t_frame (float): The time per frame.
            macc (tuble, optional): MACC mode for data acquisition. Defaults to None.
        """
        self.diff = np.diff(data, axis=0)
        self.tpf = np.diff(times).mean() #Time/frame.
        if macc is not None : 
            self.ng, self.nf, self.nd = macc
        else :
            self.ng = data.shape[0]
            self.nf = 1
            self.nd = 0

        self.alpha = (1-self.nf**2)/(3*self.nf+self.nd)

        self.pixshape = data.shape[1:]
    
    def estim_f(self, sigma_r=10):
        
        xi = (1+self.alpha)/2.
        beta = 2*sigma_r**2 / (1+self.alpha)
        m2 = np.sum( (self.diff + beta)**2, axis=0) / (self.ng-1)
        
        return np.sqrt(xi**2 + m2 - xi - beta)
        
    def estim_sigf(self,sigma_r, fe):
        """Variance estimator.
    
        Args:
            fe (float or ndarray): Average gain or gain map
            sigma_r (float or ndarray): Average readnoise or readnoise map.
    
        Returns:
            Variance : Variance of the slope of the ramp in units of ADU.
        """
        alpha_ = self.alpha
        beta_ =  2*sigma_r**2 / (1+self.alpha)
        g_ = self.estim_f(fe, sigma_r)
        term  = 2*(g_+beta_)**2 /  ((self.ng-1)*
                                    (1+2*fe/(1+alpha_) * 
                                     (g_+beta_)))
        
        return term

    def fit(self, readnoise=10, fe=1):
        """Fit method.

        Args:
            fe (int, optional): Average gain or gain map. Defaults to 1.
            readnoise (int, optional):  Average readnoise or readnoise map. Defaults to 10.
        Returns:
            tuple of ndarray : slope and variance of the considered ramp.
        """
        
        self.coefs = [self.estim_f(sigma_r=readnoise).reshape(self.pixshape)]
        
        return self.coefs, self.estim_sigf(sigma_r=readnoise, fe=fe).reshape(self.pixshape) , 0
            
    def get_model(self, pixels=None, coefs=None):
        if coefs is None : 
            coefs = self.coefs

        # Simple A*t model               
        model = coefs[0]*self.ti[:,None,None] 

        if pixels : 
            xpixel , ypixel = pixels
            model = model[:, xpixel, ypixel]
            
        return model
    
# Keeping it temporarily to make sure everything alright.
class LSQFit_dep(): 
    """ Uses least-square methods to evaluate flux."""
    def __init__(self, data, times=None, 
                 degree=5, polytype='leg'):
        """ """
        self._data = data
        if times is None:
            times = np.arange(data.shape[0])
        
        self.set_times(times)
        self.set_polynome(polytype=polytype, degree=degree)

        
    @classmethod
    def from_rawramp(cls, rawramp, degree=5, polytype='leg', **kwargs):
        """ """
        datacube, varcube, mask = rawramp.preprocess_ramp(**kwargs)
        return cls(data=datacube, times=rawramp.times, degree=degree, polytype=polytype)
        
    # ================ #
    #   method         #
    # ================ #    
    # ------- #
    # SETTER  # 
    # ------- #
    def set_polynome(self, degree, polytype="legendre", load=False):
        """ """
        
        self._polynome = {"type": polytype, 
                          "degree": degree, 
                          "array": None,
                          "coefs": None,
                         }            
        self._hcached_fitted_times = None
        
    def set_times(self, times):
        """ """
        self._times = np.asarray(times, dtype="float")
        self._hcached_fitted_times = None

    # ------- #
    # GETTER  # 
    # ------- #
    def get_times(self, as_fitted=True, polytype=None):
        """ """
        if not as_fitted:
            return self.times

        if polytype is None:
            polytype = self.polynome_type

        return self._times_to_polynomex(times=self.times, polytype=polytype)
    
    def fit(self, data=None, inplace=False):
        """ """
        if data is None:
            data = self.data
            
        coefs = self._fit_polynome_on_data(polynome=self.polynome_basis,
                                            data=data)
        if inplace:
            self._polynome["coefs"] = coefs

        return coefs


    def scan_fit(self, data=None, progress_bar=False,
                     frame_range=[4, None],
                     x_range=None, y_range=None):
        """ """
        if progress_bar:
            from tqdm import tqdm
            
        if data is None:
            data = self.data

        if x_range is not None:
            print("x_range")
            data = data[:, :, x_range[0]:x_range[1]]

        if y_range is not None:
            print("y_range")
            data = data[:, y_range[0]:y_range[1], :]
            
        # parse input
        firstframe, lastframe = frame_range
        if firstframe is None:
            firstframe = self.polynome_degree+1
            
        elif firstframe < self.polynome_degree:
            warnings.warn("firstframe is lower than the number of degree. Issues are expected.")
        
        if lastframe is None:
            lastframe = data.shape[0]
        elif lastframe < 0:
            lastframe = data.shape[0] + lastframe +1
            
        # ---- #
        xframes = np.arange(*frame_range)
        coefs = []
        times = []
        for lastframe in xframes if not progress_bar else tqdm(xframes):
            this_data = data[:lastframe]
            this_times = self.times[:lastframe]
            
            # this polynome case:
            polynome = self._polynome_of_times(this_times,
                                                 degree=self.polynome_degree,
                                                 polytype=self.polynome_type)
    
            coefs.append( self._fit_polynome_on_data(polynome, data=this_data) )
            times.append( this_times.max() )
            
        return xframes, np.asarray(times), np.asarray(coefs)

    def get_model(self, pixels=None, coefs=None):
        """ """
        if coefs is None:
            coefs = self.polynome_coefs

        if pixels is None:
            model = np.einsum('ij,jkl->ikl', self.polynome_basis.T, coefs)
        else:
            xpixel, ypixel = pixels
            model = self.polynome_basis.T @ coefs[:, xpixel, ypixel]
            
        return model
        
    def get_residuals(self, pixels=None, coefs=None, model=None):
        """ """
        if model is None:
            model = self.get_model(pixels=pixels, coefs=coefs)
            
        if pixels is None:
            residuals = self.data - model
        else:
            xpixel, ypixel = pixels
            residuals = self.data[:, xpixel, ypixel] - model
            
        return residuals

    def parse_coefs(self, coefs=None, times=None, axis=None, in_adu=False):
        """ """
        if coefs is None:
            if self.polynome_coefs is None:
                raise ValueError("no coefs given, none stored.")
                
            coefs = self.polynome_coefs
            
        if axis is None:
            # fetch which axis corresponds that of the degree:
            index = np.argwhere(np.asarray(coefs.shape) == self.polynome_degree+1).squeeze()
            if index.ndim != 0: # mean int
                raise ValueError("No dimension of coefs matches that of the polynome degree. "+\
                                 "Please set the degree axis")
            else:
                axis = int(index)
    
        # split the coefficients per interest
        offset, slope, nlinearities = np.split(coefs, [1, 2], axis=axis)
        offset = offset.squeeze()
        slope = slope.squeeze()
    
        if in_adu:
            if self.polynome_type in ["leg", "legendre", "herm", "hermite"]:
                # 
                offset -= slope

                
                if times is None:
                    times = self.times
                    
                if slope.ndim > 2 and times.ndim != slope.ndim:
                    missing_dims = slope.ndim - times.ndim
                    times = times.reshape([times.shape[0] ]+ [1, ]*missing_dims)
                        
                elif slope.ndim == 2:
                    times = float(self.times.max())
                else:
                    # let's assume user know what they are doing.
                    pass
                    
                # broadcast.
                slope = slope * 2/times
    
        return offset, slope, nlinearities
    
    # ------------- #
    #   internal    #
    # ------------- #
    @staticmethod
    def _times_to_polynomex(times, polytype):
        """ """
        # careful, -1 must corresponds to t=0 so tmin=0, *not* np.min(times)
        if polytype in ["leg", "legendre", "herm", "hermite"]:
            tmin, tmax = 0, np.max(times)
            times = ((times-tmin) / (tmax-tmin))*2 -1
        else:
            raise NotImplementedError("Unknown {polytype=} (legendre or hermite implemented).")

        return times

    @staticmethod
    def _fit_polynome_on_data(polynome, data):
        """ """
        coefs = np.linalg.lstsq(polynome.T, 
                                data.reshape(data.shape[0], -1).astype(np.float32), 
                                rcond=None)[0]
        # reshaped back as data.
        return coefs.reshape(coefs.shape[0], *data.shape[1:])

    @staticmethod
    def _get_polynome(degree, polytype, xnormed):
        """ """
        if polytype in ["leg", "legendre"]: 
            from scipy.special import legendre
            pol = np.stack([legendre(i)(xnormed) for i in range(degree+1)])
                
        elif polytype in ["herm", "hermite"]: 
            from scipy.special import hermitenorm
            pol = np.stack([hermitenorm(i)(xnormed) for i in range(degree+1)])
            
        elif polytype in ["mono", "monomial"]:
            pol = np.stack([xnormed**2 for i in range(degree+1)])

        else: 
            raise NotImplementedError("Unknown {polytype=} (legendre or hermite implemented)")

        return pol
    
    @classmethod
    def _polynome_of_times(cls, times, degree, polytype):
        """ """
        times = cls._times_to_polynomex(times, polytype=polytype)
        return cls._get_polynome(degree=degree,
                                 polytype=polytype,
                                 xnormed=times)
        
    # ================ #
    #   properties     #
    # ================ #    
    @property
    def data(self):
        """ datacube to be fitted """
        return self._data

    @property
    def times(self):
        """ """
        return self._times

    @property
    def _cached_fitted_times(self):
        """ """
        if not hasattr(self, "_hcached_fitted_times") or self._hcached_fitted_times is None:
            self._hcached_fitted_times = self.get_times()
            
        return self._hcached_fitted_times

    #  polynome
    @property
    def polynome_type(self):
        """ kind of polynome """
        return self._polynome.get("type")

    @property
    def polynome_degree(self):
        """ kind of polynome """
        return self._polynome.get("degree")    

    @property
    def polynome_basis(self):
        """ kind of polynome """
        if self._polynome.get("array", None) is None:
            self._polynome["array"] = self._get_polynome(degree=self.polynome_degree,
                                                         polytype=self.polynome_type,
                                                         xnormed=self._cached_fitted_times)
        return self._polynome["array"]

    @property
    def polynome_coefs(self):
        """ """
        return self._polynome.get("coefs", None)
    
    
    
class IP2IBestFit(Fitter):
    pass

# Register models
model_register = {'LSQ' : LSQFit,
                  'Kubik': Kubik2015,
                  'Brandt': Brandt2024}
