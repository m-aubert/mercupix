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
    try : 
        fitter = model_register[method].from_rawramp(rawramp, 
                                                       ramp_kwargs=raw_ramp_kwargs,
                                                       **kwargs)
        
    except KeyError as ke :
        raise KeyError(f"Fitter {method} does not exist")
         
    result = fitter.fit(**fit_args)
    
    return result 


## General Fitter level
class Fitter(): 
    """ 
    Baseline Fitter Class for models.
    """
    def __init__(self, data, mask=None, times=None):
        """ """
        self._data = data
        self._mask = mask
        if times is None:
            self._times = np.arange(data.shape[0])
        else : 
            self._times = times
                
    @classmethod
    def from_rawramp(cls, rawramp, ramp_kwargs=dict(), **kwargs):
        """ """
        datacube, varcube, mask = rawramp.preprocess_ramp(**ramp_kwargs)
        return cls(data=datacube, times=rawramp.times, mask=mask, **kwargs)
        
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
    def __init__(self, data, times=None, mask=None, #Adding mask for now. Will see later how to use it.
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
    
    def _fit(self, data=None, inplace=False):
        """ """
        if data is None:
            data = self.data
            
        coefs = self._fit_polynome_on_data(polynome=self.polynome_basis,
                                            data=data)
        if inplace:
            self._polynome["coefs"] = coefs

        return coefs

    def fit(self, data=None, inplace=False):
        coefs = self._fit(data=data, inplace=inplace)
        offset, slope, nl = self.parse_coefs(coefs=coefs,in_adu=True)
        
        result = RampFitResults(slope=slope, 
                       offset=offset, 
                       nl=nl)
        return result

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
    def __init__(self, data, times=None, mask=None, readnoise=11, timeframe=None,
                 countrateguess=None, reset=False, nframes_list = [], rescale=True):
        
        
        self.frame_shape = data.shape[1:]
        self.readnoise = np.atleast_1d(readnoise)     
        self.reset = reset
        self.rescale = rescale
        
        diffgp = np.diff(data, axis=0)
        
        ### Frame in a function 
        if len(nframes_list) == 0 : 
            nframes = np.ones(data.shape[0])
            self.tauis = times.astype("float32")  
            self.ti = times.astype("float32") 
        else:     
            nframes = np.asarray(nframes_list)
            tau = []
            mean_t = []
            for i, ni in enumerate(nframes) : 
                if ni > 1 :
                    # Reverse engineer the time of each read.
                    # Assumes to be even.   
                    tk = np.linspace(times[i-1], times[i], num=ni+1)[1:]
                    mean_ti = np.linspace(times[i-1], times[i], num=ni+1)[1:].mean()
                    mean_t.append(mean_ti)
                else : 
                    # Some resultant can still be only 1 frame.
                    tk = times[i]
                    mean_t.append(times[i])

                taui = sum([(2 * ni - 2 * i + 1)*tki for i, tki in zip(range(1,ni+1), np.atleast_1d(tk))])/ni**2
                tau.append(taui)
            
                self.ti = np.asarray(mean_t, dtype='float32') # Making sure no risk of changing dtype.
                self.tauis = np.asarray(tau, dtype='float32') 
        ### End of to frame in functions.
    
        self.vari = self.readnoise**2/nframes
        
        self.diffti = np.diff(self.ti, axis=0)

        if reset : 
            diffgp = np.append(([data[0]/self.ti[0]]), diffgp, axis=0)
            self.diffti = np.append(np.array([self.ti[0]]) , self.diffti, axis=0)
        
        expand_dims = [None]*(diffgp.ndim-1)
        expand_dims = (slice(None), *expand_dims)
        self.diffd = (diffgp/self.diffti[expand_dims]).reshape(diffgp.shape[0], -1)

        self.countrateguess = countrateguess 
        self.ndiff = self.diffd.shape[0]
        self.npix = self.diffd.shape[1:]
        
        self.diffmask = np.ones_like(self.diffd, dtype='int8')
        if mask is not None : 
            mask = mask.reshape((self.ndiff+1, *self.npix)).astype(bool)
            self.diffmask[mask[1:,:]] = 0
            self.diffmask[mask[:-1,:]] = 0

        # Move to property.
        self.sign = np.asarray((-1)**np.arange(1,self.ndiff+1))#, dtype='float32')

    
    def get_alpha_beta_scale(self, mask=None, countrateguess=None):

        if countrateguess is None :
            # Ugly one liner
            countrateguess = self.get_countrateguess(mask=mask) if self.countrateguess is None else self.countrateguess 
    
        alpha, beta = self._get_covar(countrateguess)
        
        if self.rescale : 
            scale = alpha[0] 
            alpha = alpha/scale
            beta = beta/scale

        else : 
            scale = np.float32(1.)
            
        return alpha, beta, scale
                
    def get_countrateguess(self, diffd=None, mask=None): 
        if diffd is None : 
            diffd = self.diffd 
        if mask is not None : 
            diffd *= mask 
        if self.reset: 
            countrateguess = self.diffd[1:].mean(axis=0)
        else : 
            countrateguess = self.diffd.mean(axis=0)
            
        return countrateguess   
                
    def fit(self, inplace=True, jumps=False, jump_threshold=[20,30]): 
        
        result = self.get_slope(jumps=jumps, jump_threshold=jump_threshold)
        
        if inplace :
            self.result = result
                        
        return result
    
    def get_model(self, pixels=None, coefs=None, intercept=False, coefid=0):
        # Here linear model
        if coefs is None : 
            coefs = self.result.slope
               
        if len(coefs) > 1 and intercept :    
            # Simple A*t + b model               
            model = coefs[0]*self.ti[:,None,None] + coefs[1]        
        else:
            # Simple A*t + b                
            model = coefs*self.ti[:,None,None] 

        if pixels : 
            xpixel , ypixel = pixels
            model = model[:, xpixel, ypixel]
            
        return model
        
    def _get_covar(self, countrateguess) : 

        ndim = countrateguess.ndim if hasattr(countrateguess , 'ndim') else 1
        expand_dims = list(range(1,ndim+1))
        
        var = self.vari 
        t = self.ti 
        tau = self.tauis 
        diffti = self.diffti 
        
        lalpha_diag = (var[1:]+var[:-1])/diffti**2  
        ralpha_diag = 1/(diffti**2) * (tau[1:]+tau[:-1] - 2*t[:-1])
        
        if lalpha_diag.ndim <= ndim : 
            lalpha_diag = np.expand_dims(lalpha_diag, expand_dims)
            ralpha_diag = np.expand_dims(ralpha_diag, expand_dims)
        
        alpha_diag = lalpha_diag + countrateguess*ralpha_diag
        
        lbeta_diag = (-var[1:-1])/(diffti[1:]*diffti[:-1])
        rbeta_diag = 1/(diffti[1:]*diffti[:-1]) * (t[1:-1] - tau[1:-1])

        if lbeta_diag.ndim <= ndim : 
            lbeta_diag = np.expand_dims(lbeta_diag, expand_dims)
            rbeta_diag = np.expand_dims(rbeta_diag, expand_dims)
            
        beta_diag = lbeta_diag + countrateguess*rbeta_diag
        
        if self.reset : 
            a0 = np.expand_dims((var[0] + countrateguess*tau[0])/ t[0]**2, expand_dims)
            b0 = np.expand_dims((var[0] + countrateguess*(t[0] - tau[0]))/(t[0]*diffti[0]), expand_dims)
            
            alpha_diag = np.append(a0, alpha_diag, axis=0)
            beta_diag = np.append(b0, beta_diag, axis=0)
            
        return alpha_diag, beta_diag
             
    def get_slope(self,
                  jumps=False, 
                  intercept=False,
                  jump_threshold=[20,30], 
                  max_niter=5):
        
        ndrops = 1 
        slope = None
        diffmask = self.diffmask.copy() 
        diffd = self.diffd*diffmask 
        detected_jump = np.ones_like(diffmask)
        
        if jumps : 
            n=0
            if all([self.reset, jumps]): 
                raise ValueError('jumps and intercept cannot be done at the same time')

            while ndrops > 0 and n < max_niter: 
                result, newmask, ndrops = self._slope_iteration(diffd=diffd, 
                                                                countrateguess=slope, 
                                                                diffmask=diffmask, 
                                                                intercept=False,
                                                                jumps=jumps, 
                                                                jump_threshold=jump_threshold)
                
                detected_jump *=newmask #jump only
                diffmask *=newmask  #combine mask and jumps
                slope = result.slope
                diffd = self.diffd*diffmask 
                n+=1

            #if ndrops > 0:
            # Add Warning for cosmic detection ?    


        # Debiasing - last run without any interuption. Best countrateguess option.
        # Readnoise's role is TBD. Plays a lot in the jump_threshold.
        result, _, _ = self._slope_iteration(diffd=diffd*diffmask, 
                                             countrateguess=slope, 
                                             intercept=intercept, 
                                             diffmask=diffmask, 
                                             jumps=False)

        result.set_mask2d(detected_jump.sum(axis=0)) 
        result.reshape(self.frame_shape, reshape_only=['slope', 
                                                       'mask2d',
                                                       'qof', 
                                                       'offset',
                                                       'var'])
        # Mask 2D will flag pixels.
        # TBD if we want to know how many times the pixels was found to be whacky.
        return result 
        
    def _slope_iteration(self, diffd=None, diffmask=None, countrateguess=None, 
                        jumps=False, jump_threshold=[20, 30], intercept=False):

        if diffd is None : 
            diffd = self.diffd
        if countrateguess is None : 
            countrateguess = self.countrateguess
        if diffmask is None : 
            diffmask = np.ones_like(self.diffd, dtype='int8')

        diffd *= diffmask 

        alpha, beta, scale = self.get_alpha_beta_scale(countrateguess=countrateguess) 
        
        beta *=  diffmask[1:]*diffmask[:-1]
        
        theta = self._get_theta(alpha, beta)
        phi = self._get_phi(alpha,beta)
        
        Phi = self._get_Phi(beta,phi)        
        Theta = self._get_Theta(beta,theta)    
        
        ThetaD = self._get_ThetaD(diffd, beta,theta)    
        
        theta_term = self.sign[:,None] / theta[-1]
        
        dC = theta_term * (phi[1:]*Theta+theta[:-1]*Phi)
        dC *= diffmask
        
        B = np.sum(diffd*dC, axis=0)        
        C = np.sum(dC, axis=0)
        
        A = 2*np.sum(diffd[1:] * theta_term[1:] * phi[2:] * ThetaD[1:-1] * beta, axis=0) 
        A += 2*(diffd[0] * theta_term[0] * phi[1] * ThetaD[0]) # beta0 = 1
        A += np.sum((diffd)**2 * theta[:-1] * phi[1:] / theta[-1], axis=0)
                    
        slope = B/C
        uncert = scale/C 
        totchi2 = (A - B**2/C)/scale
        
        # Prepare output
        newmask = 1
        ndrops = 0    
        if jumps : 
            if all([jumps, intercept]):
                raise ValueError('Jumps and intercept cannot both be true')
            
            slope, newmask, ndrops = self._jump_iteration(diffd, beta, theta, phi,
                                ThetaD, theta_term, A, B, C, dC,
                                scale, totchi2, slope, jump_threshold=jump_threshold)
            
        result = RampFitResults(slope=slope, 
                        var=uncert,
                        qof=totchi2)
           
        if intercept :
            result = self._get_slope_intercept(diffd, beta, theta, phi, Phi, Theta,
                        ThetaD, A, B, C, scale, ktc_prior=None)

        return result, newmask, ndrops
                                 
                            
    def _jump_iteration(self, diffd, beta, theta, phi,
                        ThetaD, theta_term, A, B, C, dC, scale, 
                        totchi2, slope, jump_threshold=None):
         
        PhiD = self._get_PhiD(diffd, beta, phi)
    
        cd = self._get_cov_diag(theta, phi)
        dB =  theta_term * (phi[1:]*ThetaD[1:] + theta[:-1]*PhiD)
        
        oneomit_slope = (cd * B - dB * dC) / (C * cd - dC**2) # Newcountrate.
        b = (dB - oneomit_slope*dC) / cd

        oneomit_chi2 = (A + oneomit_slope**2 * C - 
                        2 * oneomit_slope * B + b**2 * cd - 
                        2 * b * dB + 2 * oneomit_slope * b * dC)/scale
                
        cod = self._get_cov_offdiag(beta, theta, phi)
            
        factornew = dC[:-1]**2 - C*cd[:-1]
        factornew2 = dC[1:]*dC[:-1] - C*cod
        factornew3 = B*dC[:-1] - dB[1:] * C

        c = factornew3 / factornew - (B * dC[1:] - dB[1:] * C)/factornew2    
        c /= factornew2/factornew - (dC[1:]**2 - C * cd[1:])/factornew2

        b2 = (factornew3 - c*factornew2)/factornew
        twoomit_slope = (B - b2 * dC[:-1] - c*dC[1:]) / C

        twoomit_chi2 = A + twoomit_slope**2 * C + b2**2 * cd[:-1] + c**2 * cd[1:]
        twoomit_chi2 -= 2*twoomit_slope* B + 2*b2*dB[:-1]+2*c*dB[1:]
        twoomit_chi2 += 2*twoomit_slope*b2*dC[:-1] + 2*twoomit_slope*c*dC[1:] + 2*b2*c*cod
        twoomit_chi2 /= scale

        dchi_one = totchi2 - oneomit_chi2  # Chi2 map
        dchi_two = totchi2 - twoomit_chi2 # Chi2 map

        best_dchi_one = dchi_one.max(axis=0)
        best_dchi_two = dchi_two.max(axis=0)
        
        whichdrop = dchi_one.max(axis=0) - jump_threshold[0] > dchi_two.max(axis=0) - jump_threshold[1]
        
        bestchi2 = best_dchi_one * (best_dchi_one> jump_threshold[0]) * whichdrop
        bestchi2 += best_dchi_two * (best_dchi_two > jump_threshold[1]) * (~whichdrop)
        bestchi2[bestchi2 == 0] = np.nan
        
        dropone = (dchi_one == bestchi2) & whichdrop
        droptwo = (dchi_two == bestchi2) & (~whichdrop)

        # Change sign 
        droptot = np.ones_like(dropone)
        droptot *= ~dropone
        droptot[1:] *= ~droptwo
        droptot[:-1] *= ~droptwo
            
        slope[dropone.sum(axis=0).astype(bool)] = oneomit_slope[dropone] 
        slope[droptwo.sum(axis=0).astype(bool)] = twoomit_slope[droptwo]  
        
        ndrops = np.any([dropone.sum(axis=0), droptwo.sum(axis=0)], axis=0).sum()
        
        return slope, droptot, ndrops
    
    def _get_slope_intercept(self, diffd, beta, theta, phi, Phi, Theta,
                        ThetaD, A, B, C, scale, ktc_prior=None):
        if ktc_prior is not None :  
            z, zerr = ktc_prior
            z_term1 = z/zerr**2
            z_term2 = 1/zerr**2
            
        else : 
            z_term1 = 0
            z_term2 = 0
        
        PhiD = self._get_PhiD(diffd, beta, phi)

        c11 = theta[0]/theta[-1] * phi[2]
        cp1 = -1/theta[-1] * (phi[2]*Theta[1] + theta[0]*Phi[1])
        bp1 = 1/theta[-1] * (phi[2]*ThetaD[1] + theta[0]*PhiD[1])

        b = (bp1 / self.ti[1] - cp1*B / (C*self.ti[1]) + z_term1)
        b /= (c11/self.ti[1]**2 - cp1**2/ (C*self.ti[1]) + z_term2) 
        
        a = B/C - b*(cp1*B / (C*self.ti[1]))
                
        cov_ab = np.array([C , cp1 / self.ti[1], [cp1 / self.ti[1], c11/self.ti[1]**2 + z_term2]])
    
        chi2 = A + a**2*C + b**2/self.ti[1]**2 * c11 - 2*b/self.ti[1] * bp1 
        chi2 += -2*a*B + 2*a*b/self.ti[1]*cp1 
        if z_term2 != 0:
            chi2 += (b-z)**2 / zerr

        result = RampFitResults(slope=a,
                                offset=b,
                                var=(cov_ab),
                                qof=chi2/scale)
        
        return result 
    
    def _get_cov_diag(self, theta, phi):
        return theta[:-1]/theta[self.ndiff] * phi[1:]
    
    def _get_cp1_term(self, phi, theta, Theta, Phi):
        return self.sign/theta[self.ndiff] * (phi[1:]*Theta + theta[:-1]*Phi)

    def _get_bp1_term(self, beta,theta, phi, ThetaD, PhiD):
        return self.sign/theta[self.ndiff] * (phi[1:]*ThetaD[1:] + theta[:-1]*PhiD)

    def _get_cov_offdiag(self, beta,theta, phi):
        return -beta * theta[:-2] * phi[2:] / theta[self.ndiff]

    def _get_PhiD(self, diffd, beta, phi): 
        PhiD = np.zeros((self.ndiff, *self.npix))       
        for i in range(self.ndiff - 2, -1, -1):
                PhiD[i] = (PhiD[i+1] + self.sign[i+1] * diffd[i+1] * phi[i+2]) * beta[i] 
        
        return PhiD
        
    def _get_theta(self, alpha, beta):
        theta = np.ones((self.ndiff+1, *self.npix))         
        theta[1] = alpha[0]
        for i in range(2,self.ndiff+1):
            theta[i] = alpha[i-1] * theta[i-1] - beta[i-2]**2 * theta[i-2]
                    
        return theta
    
    def _get_phi(self, alpha, beta):
        phi = np.ones((self.ndiff+1, *self.npix))        
        phi[self.ndiff-1] = alpha[self.ndiff-1] 
        for i in range(self.ndiff-2, -1, -1):
            phi[i] = alpha[i] * phi[i+1] 
            phi[i] -= beta[i]**2 * phi[i+2]
            
        return phi   
    
    def _get_Phi(self, beta, phi):
        Phi = np.zeros((self.ndiff, *self.npix))    
        for i in range(self.ndiff-2, -1, -1):
            Phi[i] = beta[i]*Phi[i+1] 
            Phi[i] += self.sign[i+1]*beta[i]*phi[i+2]
                        
        return Phi    
    
    def _get_Theta(self, beta, theta):
        Theta = np.zeros((self.ndiff, *self.npix))          
        Theta[0] = -theta[0]
        for i in range(1,self.ndiff):
            Theta[i] = beta[i-1]*Theta[i-1] 
            Theta[i] += self.sign[i]*theta[i]
                    
        return Theta
            
    def _get_ThetaD(self, diffd, beta, theta): 
        
        ThetaD = np.zeros((self.ndiff+1, *self.npix))            
        ThetaD[1] = -diffd[0] * theta[0]
        for i in range(1,self.ndiff):
            ThetaD[i+1] =  beta[i-1]*ThetaD[i] 
            ThetaD[i+1] += self.sign[i]*diffd[i]*theta[i]
            
        return ThetaD
      
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
        g_ = self.estim_f(sigma_r)
        term  = 2*(g_+beta_)**2 /  ((self.ng-1)*
                                    (1+2*fe/(1+alpha_) * 
                                     (g_+beta_)))
        
        return term

    def fit(self, readnoise=10, fe=1, inplace=True):
        """Fit method.

        Args:
            fe (int, optional): Average gain or gain map. Defaults to 1.
            readnoise (int, optional):  Average readnoise or readnoise map. Defaults to 10.
        Returns:
            tuple of ndarray : slope and variance of the considered ramp.
        """        
        slope = self.estim_f(sigma_r=readnoise)
        err = self.estim_sigf(readnoise, fe)
        
        self.result = RampFitResults(slope=slope, var=err)
        
        return self.result 
            
    def get_model(self, pixels=None, coefs=None):
        if coefs is None : 
            coefs = self.result.slope 

        # Simple A*t model               
        model = coefs*self.times[:,None,None] 

        if pixels : 
            xpixel , ypixel = pixels
            model = model[:, xpixel, ypixel]
            
        return model


class RampFitResults():
    
    _reshape_ok = ['slope', 
                    'mask2d',
                    'mask3d',
                    'qof',
                    'offset',
                    'nl',
                    'var']
    
    def __init__(self, slope=None, mask2d=None, mask3d=None, var=None,
                        qof=None, offset=None, nl=None, model='LSQFit'):
        
        self._slope = slope
        self._mask2d  = mask2d
        self._mask3d = mask3d
        self._qof = qof
        self._offset = offset
        self._nl =  nl
        self._model = model 
        self._var = var
        
    @property
    def slope(self): 
        return self._slope
    
    @property
    def mask2d(self): 
        return self._mask2d
    
    @property
    def mask3d(self): 
        return self._mask3d
    
    @property
    def qof(self): 
        return self._qof
    
    @property
    def offset(self): 
        return self._offset
    
    @property
    def nl(self): 
        return self._nl
    
    @property
    def model(self): 
        return self._model

    @property
    def var(self): 
        return self._var
    
    def set_slope(self, slope): 
       self._slope = slope
    
    def set_mask2d(self, mask2d): 
        self._mask2d = mask2d
    
    def set_mask3d(self, mask3d): 
        self._mask3d = mask3d

    def reshape(self, *shape, reshape_only=None):
        if reshape_only is None : 
            reshape_only = self._reshape_ok
        
        else: 
            reshape_only = np.atleast_1d(reshape_only)
            valid = all(val in self._reshape_ok for val in reshape_only)  
            if not valid : 
                raise ValueError(f'Only {self._reshape_ok} attributes can be reshaped.')
    
        for name in reshape_only: 
            arr = getattr(self, '_'+name)
            if arr is None : 
                continue
            arr = np.asanyarray(arr)
            if arr.size != np.prod(shape):
                shape_ = (arr.shape[0], *shape) 
                # Assumes first dimension should be always kept.
            else : 
                shape_ = shape
                
            setattr(self, '_'+name, arr.reshape(*shape_))

    
    
class IP2IBestFit(Fitter):
    pass
     #likelihood based method ?
     # Placeholder for now : 
     # Ambition : 
     # Accounts for additional NL.
     # Accounts for BFE.
     # Accounts (correct) for persistence.

# Register models
model_register = {'LSQ' : LSQFit,
                  'Kubik': Kubik2015,
                  'Brandt': Brandt2024}
