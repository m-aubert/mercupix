import numpy as np
from scipy.special import legendre, hermitenorm
import warnings 
from .calib import PersistenceModel
from scipy.integrate import quad

# Rampfitting module. Implemented : 
#- LSQfit (for now the general least square)
#- Kubik2016 - compare with new paper.
#- Brandt2024
# To do : 
# Casertano2022 ( WFIRST and JWST models)
# GeneralLikelihood (with jax) 

def fit_ramp(datacube, varcube=None, 
                       mask=None,
                       time_frames=None, 
                       time_group=None, 
                       method='LSQ', 
                       fit_args={}, 
                       persistence=False, 
                       bfe=False,
                       crnl=False,
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
        
    if method == 'LSQ': 
        if persistence or bfe or crnl : 
            warnings.warn(f'Persistence, BFE and CRNL are not implemented in {method}')
        
        # Add warning persistence / bfe, crnl.
        Fitter = LSQFit(datacube, time_group,  **kwargs)
    elif method == 'Kubik': 
        if persistence or bfe or crnl : 
            warnings.warn(f'Persistence, BFE and CRNL are not implemented in {method}')
        
        Fitter = Kubik2015(datacube, np.diff(time_frames)[0], **kwargs)
    elif method == 'Brandt': 
        if persistence or bfe or crnl : 
            warnings.warn(f'Persistence, BFE and CRNL are not implemented in {method}')
        
        Fitter = Brandt2024(datacube, timeframe=time_frames, timegroup=time_group, **kwargs)
    elif method == 'IP2I':
        if persistence or bfe or crnl : 
            warnings.warn(f'Persistence, BFE and CRNL are not implemented YET in {method}')
            
        Fitter = IP2IBestFit(datacube=datacube, varcube=varcube)
        
    fitted_slope = Fitter.fit(**fit_args)

    return fitted_slope 

class LSQFit: 
    """
    Uses least-square methods to evaluate flux.
    """
    def __init__(self, ramp, tarr, polytype='leg'):
        self.cube = ramp
        self.polytype = polytype
        self.tarr = tarr
        if self.polytype == 'leg':
            if tarr.min() == 0: 
                self.tleg = tarr/tarr.max() * 2  - 1 #Pour legrendre Leander ver. -> assume t0 = 0 .
            else : 
                self.tleg = 2.*(tarr - tarr.min())/np.ptp(tarr)-1

    def poly_(self, deg=3, poly='mono'):
        if poly == "leg" : 
            pol = np.stack([legendre(i)(self.tleg) for i in range(deg+1)])
        elif poly == "mono" : 
            pol = np.stack([self.tarr**i for i in range(deg+1)])
        elif poly == "herm": 
            pol = np.stack([hermitenorm(i)(self.tleg) for i in range(deg+1)])
        else : 
            return None
        return pol.astype(np.float32)

    def no_w_lstq(self, x, cube):
        cube = cube.reshape(cube.shape[0], -1).astype(np.float32)
        self.coeffs = np.linalg.lstsq(x.T, cube, rcond=None)[0]

    def fit(self, deg=1, return_slope=True):
        x = self.poly_(deg=deg, poly=self.polytype)
        self.no_w_lstq(x, self.cube) 
        
        if self.polytype == 'leg': 
            self.coeffs[1]*=2./self.tarr.max()
            
        if return_slope:  
            return self.coeffs[1], None
        else:
            return self.coeffs, None 
        
class Kubik2015: 
    """
    Method based on Kubik2016 to estimate flux from group-difference images
    """
    def __init__(self,ramp, t_frame, macc=None):
        """Kubik 2015 flux estimator. Uses the group difference.

        Args:
            ramp (ndarray): Datacube to consider (N, M, M) 
            t_frame (float): The time per frame.
            macc (tuble, optional): MACC mode for data acquisition. Defaults to None.
        """
        self.diff = np.diff(ramp, axis=0)
        self.tpf = t_frame
        if macc is not None : 
            self.ng, self.nf, self.nd = macc
        else :
            self.ng = ramp.shape[0]
            self.nf = 1
            self.nd = 0

        self.alpha = (1-self.nf**2)/(3*self.nf+self.nd)
    
    def estim_f(self, sigma_r=10):
        
        xi = (1+self.alpha)/2.
        beta = 2*sigma_r**2 / (1+self.alpha)
        m2 = np.sum( (self.diff + beta)**2, axis=0) / (self.ng-1)
        
        return np.sqrt(xi**2 + m2 - xi - beta)
        
    #def estim_sigf(self,fe,sigma_r):
    #    """Variance estimator.
    #
    #    Args:
    #        fe (float or ndarray): Average gain or gain map
    #        sigma_r (float or ndarray): Average readnoise or readnoise map.
    #
    #    Returns:
    #        Variance : Variance of the slope of the ramp in units of ADU.
    #    """
    #    alpha_ = self.alpha
    #    beta_ = self.beta(fe, sigma_r)
    #    g_ = self.estim_f(fe, sigma_r)
    #    term  = 2*(g_+beta_)**2 /  ((self.ng-1)*
    #                                (1+2*fe/(1+alpha_) * 
    #                                 (g_+beta_)))
    #    
    #    return term

    def fit(self, readnoise=10):
        """Fit method.

        Args:
            fe (int, optional): Average gain or gain map. Defaults to 1.
            readnoise (int, optional):  Average readnoise or readnoise map. Defaults to 10.
        Returns:
            tuple of ndarray : slope and variance of the considered ramp.
        """
        return self.estim_f(sigma_r=readnoise), None, None
 
class Brandt2024:    
    """
        Method to fit flux based on Brandt 2024. Assumes data are grouped.
    """
    def __init__(self, data, readnoise=11, timeframe=None, timegroup=None, 
                 add_reset=False, countrateguess=None, reset=False, nframes_list = []):
        """Brandt 2024 method to estimate flux of a ramp from group differences.
        
        Notes: 
            Assumes data are gouped.

        Args:
            data (ndarray): Datacube of (N, M, M) where N is the number of groups and M is the number of pixels.
            readnoise (float, optional): A readnoise estimate for estimating the variance. Defaults to 11.
            timeframe (ndarray, optional): The time for each frames that where used to create the grouped datacube. 
                                           Defaults to None.
            timegroup (ndarray, optional): The time for each group in the datacube (N,). Defaults to None.
            add_reset (bool, optional): If True, estimate the intercept of the slope. Defaults to False.
            countrateguess (float or ndarray of floats, optional): An initial guess for the counterate (gain). 
                                                                   Defaults to None.
            group_info (dict, optional): Information concerning the macc mode sotring the number of frame constituting each group. 
                                         Defaults to None.
        """
        
        self.readnoise = np.atleast_1d(readnoise)     
        self.datai = np.diff(data,axis=0)
        self.ti = timegroup
        if len(nframes_list) == 0 : 
            self.Nis = np.ones(data.shape[0])
        else:     
            self.Nis = nframes_list
    
        self.tauis = np.sum( [ (2*ni- 2*i + 1)*tk for i, (ni,tk) in enumerate(zip(self.Nis, timeframe))])     
        self.tauis *=1/self.Nis**2
        self.vari = self.readnoise**2/self.Nis   
        self.diffti = np.diff(self.ti)
        self.diffgp = np.diff(data,axis=0)      
             
        self.reset = reset
        self.add_reset = add_reset if self.reset else False
        
        if self.add_reset : 
            self.diffgp = np.append(np.array([self.datai[0]/self.ti[0]]), self.diffgp, axis=0)
            self.diffti = np.append(np.array([self.ti[0]]) , self.diffti, axis=0)

        self.diffd = (self.diffgp/self.diffti[:,None,None]).reshape(self.diffgp.shape[0], -1)
        
        self.ndiff = self.diffgp.shape[0]
        self.npix = self.diffd.shape[1:]
        self.sign = np.asarray((-1)**np.arange(1,self.ndiff+1))

        if countrateguess is None : 
            if self.add_reset: 
                countrateguess = self.diffd[1:].sum(axis=0)
            else : 
                countrateguess = self.diffd.sum(axis=0)
                
        self.alpha, self.beta = self.get_covar(countrateguess)
                
    def fit(self, **kwargs): 
        """Fit method.

        Returns:
            tuple of ndarray : slope and variance of the considered ramp.
        """
        if self.reset :
            a, _, a_err, qf = self.get_countrate_intercept(return_qf=True,**kwargs)
            return a, a_err[0,0], qf
        else :
            a, a_err, qf = self.get_countrate_nointercept(return_qf=True, **kwargs)
            return a, a_err, qf

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
            a0 = (self.vari[0] + a*self.tauis[0])/ self.ti[0]**2
            b0 = (self.vari[0] + a*(self.ti[0] - self.tauis[0]))/(self.ti[0]*self.diffti[0])
            alpha_diag = np.append(np.array([a0]), alpha_diag, axis=0)
            beta_diag = np.append(np.array([b0]), beta_diag, axis=0)
            
        return alpha_diag, beta_diag
        
    def get_countrate_nointercept(self, return_qf = True):
        """Estimator for the slope of the ramp.

        Args:
            return_qf (bool, optional): Whether to return a quality of the fit estimator. Defaults to True.

        Returns:
            tuple of ndarray : slope, slope variance, if return_qf is True, chi2.
        """
                 
        B = np.sum(self.diffd * self.sign[:,None] / self.theta[-1] * (self.phi[1:]*self.Theta+ self.theta[:-1]*self.Phi), axis=0)        
        C = np.sum(self.sign[:,None]/self.theta[-1] * (self.phi[1:]*self.Theta+self.theta[:-1]*self.Phi), axis=0)

        a = B/C
        a_err = 1./C
        
        if return_qf:       
            A = 2*np.sum(self.diffd * self.sign[:,None]/self.theta[-1] * self.phi[1:] * self.ThetaD[:-1], axis=0)
            A += np.sum((self.diffd)**2 * self.theta[:-1] * self.phi[1:] / self.theta[-1], axis=0)
            chi2 = A - B**2/C
            
            return a, a_err, chi2        
        
        
                                    
        return a,a_err
        
    def get_countrate_intercept(self, return_qf = True, ktc_prior=None):
        """Estimator for the slope of the ramp and its intercept.

        Args:
            return_qf (bool, optional): Whether to return a quality of the fit estimator. Defaults to True.
            ktc_prior (tuple of floats or ndarray, optional): Priors for the ktc value and errors. Defaults to None.

        Returns:
            tuple of ndarray : slope, intercept, covariance, and if return_qf is True, chi2.
        """
        
        if ktc_prior is not None : 
            z, zerr = ktc_prior
            z_term1 = z/zerr**2
            z_term2 = 1/zerr**2
            
        else : 
            z_term1 = 0
            z_term2 = 0
        # Still in dev, for now merely copypaste of function above 
         
        B = np.sum(self.diffd * self.sign[:,None] / self.theta[-1] * (self.phi[1:]*self.Theta+ self.theta[:-1]*self.Phi), axis=0)        
        C = np.sum( self.sign[:,None]/self.theta[-1] * (self.phi[1:]*self.Theta+self.theta[:-1]*self.Phi), axis=0)
        
        b = (self.Bp1 / self.ti[1] - self.Cp1*B / (C*self.ti[1]) + z_term1)
        b /= (self.C11/self.ti[1]**2 - self.Cp1**2/ (C*self.ti[1]) + z_term2) 
        
        a = B/C - b*( self.Cp1*B / (C*self.ti[1]))
                
        cov_ab = np.array([[C , self.Cp1 / self.ti[1]], [self.Cp1 / self.ti[1], self.C11/self.ti[1]**2 + z_term2]])
        #cov_ab = np.linalg.inv(cov_ab)
        if return_qf:       
            A = 2*np.sum(self.diffd * self.sign[:,None]/self.theta[-1] * self.phi[1:] * self.ThetaD[:-1], axis=0)
            A += np.sum((self.diffd)**2 * self.theta[:-1] * self.phi[1:] / self.theta[-1], axis=0)
                
            chi2 = A + a**2*C + b**2/self.ti[1]**2 * self.C11 - 2*b/self.ti[1] * self.Bp1 
            chi2 += -2*a*B + 2*a*b/self.ti[1]*self.Cp1 
            if z_term2 != 0:
                chi2 += (b-z)**2 / zerr
            
            return a, b, cov_ab, chi2
                               
        return a,b, cov_ab      
    
    @property
    def C11(self):
        return self.theta[0]/self.theta[-1] * self.phi[2]
    
    @property
    def Cp1(self):
        return -1/self.theta[-1] * (self.phi[2]*self.Theta[1] + self.theta[0]*self.Phi[1])
    
    @property
    def Bp1(self):
        return -1/self.theta[-1] * (self.phi[2]*self.ThetaD[1] + self.theta[0]*self.PhiD[1])
    
    @property
    def PhiD(self):
        if not hasattr(self, '_PhiD'):
            PhiD = np.ones((self.ndiff, *self.npix)) 
            PhiD[-1] = 0
            PhiD[:-1] *= self.beta[:,None] 
            for i in range(self.ndiff - 2, -1, -1):
                PhiD[i] *= (PhiD[i+1] + self.sign[i+1] * self.diffd[i+1] * self.phi[i+2])
            self._PhiD = PhiD
        return self._PhiD

    @property
    def theta(self):
        if not hasattr(self, '_theta'):
            theta = np.ones((self.ndiff+1, *self.npix))
            theta[1] = self.alpha[1]
            for i in range(2,self.ndiff+1):
                theta[i] = self.alpha[i-1] * theta[i-1] - self.beta[i-2]**2 * theta[i-2]
                
            self._theta = theta
        
        return self._theta
    
    @property
    def phi(self):
        if not hasattr(self, '_phi'):
            phi = np.ones((self.ndiff+1, *self.npix))

            phi[-2] = self.alpha[-1]
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
            Theta = np.zeros((self.ndiff,*self.npix))
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
 
 
class IP2IBestFit:
    """
        In case we want to do some jax/other minimizations for linear
    """
    def __init__(self, datacube=None, varcube=None):
        self.datacube = datacube
        self.varcube = varcube
        
    def full_flux_model(self, t, I, a_pers, dt_obs, gamma, gain=1):        
        # First draft of this. 
        # Will defo change most-likely to account for:
        # - previous ramp impact
        # - the fact that this is not a single ramp but bazillions pixels.
        pers = np.fromiter(map(lambda x : quad(PersistenceModel.power_law_persistence, 0, x, 
                                               args=(a_pers, dt_obs, gamma,))[0], t.tolist()))
        return I*t + gain*pers
 
    def fit(self):
        # Will need to differentiate Legendre method from this one : 
        # With N-legendre fit compute a quality-factor (residuals) with same number of free parameters.
        # If not -> have fun.
        # Loop over N. Look at numba solutions / jit / minuit and lowlevel calls.
        # Might need to update python & modules for proper integration between : 
        # Numba, Low-level calls for quad and other stuff. 
        # Fun times ahead.
        pass
        
    
class Casertano2022:
    """
        Method based on Casertano 2022 report for flux estimate.
    """
    def __init__(self):
        pass
        
### Below : to delete after tests

class GroupImg : 
    """
        Temporary function to define a grouped image and its informations.
    """
    def __init__(self, data, time, readnoise): 
        self.readnoise = np.atleast_1d(readnoise)
        self.datai = data.mean(axis=0) if data.ndim == 3 else data #Assume (Nframes, Npix, Npix)
        self.ti = np.atleast_1d(time).mean(axis=0) #Assumes (Nframes, ti)
        self.Ni = data.shape[0] if data.ndim == 3 else 1 
        self.taui  = np.sum( [ (2*self.Ni - 2*i + 1)*tk for i, tk in enumerate(np.atleast_1d(time))])        
        self.taui *=1/self.Ni**2
        self.vari = self.readnoise**2/self.Ni    
    
class GroupDiff:
    """
        Temporary function to create a group difference array.
    """
    def __init__(self, gp_list, add_reset=False): #Assume ordered gp_list
        #gp_list should be list of gp image.
        
        self.diffgp = np.diff(list(map(lambda x : x.datai, gp_list)), axis=0)
        self.ti = np.asarray(list(map(lambda x : x.ti, gp_list)))
        self.diffti = np.diff(self.ti)
        self.Nis = np.asarray(list(map(lambda x : x.Ni, gp_list)))
        self.tauis = np.asarray(list(map(lambda x : x.taui, gp_list)))
        self.vari = np.asarray(list(map(lambda x : x.vari, gp_list))).ravel()

        self.reset = add_reset
        
        if add_reset : 
            self.diffgp = np.append(np.array([gp_list[0].datai/self.ti[0]]), self.diffgp, axis=0)
            
    def get_covar(self, a) : 
        # Only returns diag value for now. No need to use mem for no reason.
        # Plus sparse matrix here.
        alpha_diag = (self.vari[1:]+self.vari[:-1])/self.diffti**2  
        alpha_diag += a/self.diffti**2 * (self.tauis[1:]+self.tauis[:-1] - 2*self.ti[:-1])

        beta_diag = (-self.vari[1:-1])/(self.diffti[1:]*self.diffti[:-1])
        beta_diag += a/(self.diffti[1:]*self.diffti[:-1]) * ( self.ti[1:-1] -self.tauis[1:-1])
        
        if self.add_reset : 
            a0 = (self.vari[0] + a*self.tauis[0])/ self.ti[0]**2
            b0 = (self.vari[0] + a*(self.ti[0] - self.tauis[0]))/(self.ti[0]*self.diffti[0])
            alpha_diag = np.append(np.array([a0]), alpha_diag)
            beta_diag = np.append(np.array([b0]), beta_diag)
            
        return alpha_diag, beta_diag
    
    
        