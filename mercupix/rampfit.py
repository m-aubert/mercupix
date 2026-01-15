"""
This document will probably list all estimator usable for flux.

- Kubik2016
- Brandt2024
- Casertano2022 (to do)
- GeneralLikelihood (with jax for fun why not, lets see)
- Rampfit (for now the general least square)

Shall now create high-level class that should call the estimator of interest. 

"""
import numpy as np
from scipy.special import legendre, hermitenorm


def get_ramp_fit(Ramp, fit_procedure='LSQ', fit_args={},  **kwargs): 
    """High-level method to estimate fit.

    Args:
        Ramp (class): _description_
        fit_procedure (str, optional): Refers to the method used for fitting. \ 
                                        Can be 'LSQ', 'Kubik', 'Brandt'. Defaults to 'LSQ'.

    Returns:
        ndarray : fitted slope
    """        
    if fit_procedure == 'LSQ': 
        Fitter = LSQFit(Ramp.raw_to_calib(), Ramp.t_read,  **kwargs)
    elif fit_procedure == 'Kubik': 
        Fitter = Kubik2015(Ramp.raw_to_calib(), np.diff(Ramp.t_read)[0], **kwargs)
    elif fit_procedure == 'Brandt': 
        Fitter = Brandt2024(Ramp.raw_to_calib(), timeframe=Ramp.t_read, timegroup=Ramp.t_gread, **kwargs)
        
    slope = Fitter.fit(**fit_args)
    return slope 

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

    def fit(self, deg=1, return_slope=True):
        x = self.poly_(deg=deg, leg=(self.polytype=='leg'))
        self.no_w_lstq(x, self.cube) 
        
        if self.poly_ == 'leg': 
            self.coeffs[1]*=2/self.tarr.max()
            
        if return_slope:  
            return self.coeffs[1]
        else:
            return self.coeffs 
        
class Kubik2015 : 
    """
    Method based on Kubik2016 to estimate flux from group-difference images
    """
    def __init__(self,ramp, t_frame, macc=None):
        self.diff = np.diff(ramp, axis=0)
        self.tpf = t_frame
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
        term *= np.sqrt(1 + 4*fe**2 * np.sum( (self.diff + beta_)**2, axis=0) / 
                        ((self.ng+1)*(alpha_+1)**2 )) - 1
        term -= beta_
        return term
    
    def estim_sigf(self,sigma_r, fe):
        alpha_ = self.alpha
        beta_ = self.beta(sigma_r,fe)
        g_ = self.estim_f(fe, sigma_r)
        term  = 2*(g_+beta_)**2 /  ((self.ng-1)*
                                    (1+2*fe/(1+alpha_) * 
                                     (g+beta_)))
        
        return term

    def fit(self, fe=1, readnoise=10):
        return self.estim_f(fe, readnoise) 
            
class Brandt2024:    
    """
        Method to fit flux based on Brandt 2024. Assumes data are grouped.
    """
    def __init__(self, data, readnoise=11, timeframe=None, timegroup=None, 
                 add_reset=False, countrateguess=None, group_info=None):
        
        self.readnoise = np.atleast_1d(readnoise)     
        self.datai = data
        self.ti = timegroup
        self.Nis = group_info['nframe'] # Array of Nframes / group
        self.tauis = np.sum( [ (2*ni- 2*i + 1)*tk for i, (ni,tk) in enumerate(zip(self.Nis, timeframe))])     
        self.tauis *=1/self.Nis**2
        self.vari = self.readnoise**2/self.Nis   
        self.diffti = np.diff(self.ti)
        self.diffgp = np.diff(data,axis=0)
        
        self.reset = add_reset                            
                    
        if self.reset : 
            self.diffgp = np.append(np.array([self.datai[0]/self.ti[0]]), self.diffgp, axis=0)
            self.diffti = np.append(np.array([self.ti[0]]) , self.diffti, axis=0)

        self.diffd = (self.diffgp/self.diffti[:,None,None]).reshape(self.diffgp.shape[0], -1)
        
        self.ndiff = self.diffgp.shape[0]
        self.npix = self.diffd.shape[-1]
        self.sign = ((-1)**np.arange(1,self.ndiff+1))

        if countrateguess is None : 
            if self.reset: 
                countrateguess = self.diffd[1:].sum(axis=0)
            else : 
                countrateguess = self.diffd.sum(axis=0)
                
        self.alpha, self.beta = self.get_covar(countrateguess)
                
    def fit(self, **kwargs): 
        if self.reset :
            return self.get_countrate_intercept(self, return_qf =True,**kwargs)[0]
        else :
            return self.get_countrate_nointercept(return_qf=False, **kwargs)[0]

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
        
    def get_countrate_nointercept(self, return_qf = True):
                 
        B = np.sum(self.diffd * self.sign[:,None] / self.theta[-1] * (self.phi[1:]*self.Theta+ self.theta[:-1]*self.Phi), axis=0)        
        C = np.sum( self.sign[:,None]/self.theta[-1] * (self.phi[1:]*self.Theta+self.theta[:-1]*self.Phi), axis=0)

        a = B/C
        a_err = 1./C
        
        if return_qf:       
            A = 2*np.sum(self.diffd * self.sign[:,None]/self.theta[-1] * self.phi[1:] * self.ThetaD[:-1], axis=0)
            A += np.sum((self.diffd)**2 * self.theta[:-1] * self.phi[1:] / self.theta[-1], axis=0)
            chi2 = A - B**2/C
            
            return a, a_err, chi2                                    
        return a,a_err
        
    def get_countrate_intercept(self, return_qf = True, ktc_prior=None):
        
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
            
            return a, b, chi2, cov_ab
                               
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
            PhiD = np.ones((self.ndiff, self.npix)) #Trying to save mem.
            PhiD[-1] = 0
            PhiD[:-1] *= self.beta[:,None] 
            for i in range(self.ndiff - 2, -1, -1):
                PhiD[i] *= (PhiD[i+1] + self.sign[i+1] * self.diffd[i+1] * self.phi[i+2])
            self._PhiD = PhiD
        return self._PhiD

    @property
    def theta(self):
        if not hasattr(self, '_theta'):
            theta = np.ones((self.ndiff+1, self.npix))
            theta[1] = self.alpha[1]
            for i in range(2,self.ndiff+1):
                theta[i] = self.alpha[i-1] * theta[i-1] - self.beta[i-2]**2 * theta[i-2]
                
            self._theta = theta
        
        return self._theta
    
    @property
    def phi(self):
        if not hasattr(self, '_phi'):
            phi = np.ones((self.ndiff+1, self.npix))

            phi[-2] = self.alpha[-1]
            for i in range(self.ndiff-2, -1, -1):
                phi[i] = self.alpha[i] * phi[i+1] - self.beta[i]**2 * phi[i+2]
        
            self._phi = phi
        
        return self._phi    
    
    @property
    def Phi(self):
        if not hasattr(self, '_Phi'):
            Phi = np.zeros((self.ndiff,self.npix))
            for i in range(self.ndiff-2, -1, -1):
                Phi[i] = self.beta[i]*Phi[i+1] + self.sign[i+1]*self.beta[i]*self.phi[i+2]
                
            self._Phi = Phi
            
        return self._Phi
    
    @property
    def Theta(self):
        if not hasattr(self, '_Theta'):
            Theta = np.zeros((self.ndiff,self.npix))
            Theta[0] = -self.theta[0]
            
            for i in range(1,self.ndiff):
                Theta[i] = self.beta[i-1]*Theta[i-1] + self.sign[i]*self.theta[i]
                
            self._Theta = Theta
            
        return self._Theta
            
    @property
    def ThetaD(self):
        if not hasattr(self, '_ThetaD'):
            ThetaD = np.zeros((self.ndiff+1, self.npix))
            ThetaD[1] = -self.diffd[0] * self.theta[0]
            
            for i in range(1,self.ndiff):
                ThetaD[i+1] = self.beta[i-1]*ThetaD[i] + self.sign[i]*self.diffd[i]*self.theta[i]
                
            self._ThetaD = ThetaD
            
        return self._ThetaD

class Casertano2022:
    """
        Method based on Casertano 2022 report for flux estimate.
    """
    def __init__(self):
        pass
    

class GenericLikelihoodBestfit :
    """
        In case we want to do some jax/other minimizations for linear
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
    
    
    
class SinglePixelRamp:
    """Dedicated to single pix processing.
    Should include linearization procedure but this should probably be experimented on more.
    Should also be included Ramp linearization part tbh.
    """
    def __init__(self, pixramp, tread):
            self.pixramp = pixramp
            self.tread = tread

    def linearize_single_pixel(self, coef, deg=3, poly='mono'): 
        lin_meas = 1 - self.pixramp / np.sum(poly_(self.t_read, deg=1, poly='mono').T*coef[:2], axis=1)
        coef_lin = np.linalg.lstsq(poly_(self.pixramp, deg=deg, poly='mono').T, lin_meas.astype(np.float32), rcond=None)[0]
        pixlin = self.pixramp * (1+(coef_lin[:,None]*poly_(self.pixramp, deg=deg, poly='mono')).sum(axis=0))
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
        coefs = self.ramp_lstq(deg=deg, poly=poly)
        pixlin = self.linearize_single_pixel(coefs, deg=deg, poly=poly)
        coef_lin = self.ramp_lstq(pixlin,deg=1, poly='mono')
        return coef_lin
    
        