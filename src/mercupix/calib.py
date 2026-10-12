"""
    This file will basically list all effects that should be used to get a nice 2D image ready to be detrended.
    Could be carac.py. To be discussed.
    
    List of effects that might bias the per pixel flux estimate : 
    
    - Classical NL 
    - Brighter-Fatter
    - Persistence  
    - Burnin 
    - IPC correction 
    - NL-IPC correction
    - Charge diffusion
    
    Other / unclear : 
    - Gain estimation (PTC) 
    - Dark current (shouldn't it be after flux estimate ?)
    
    This file will either : 
    - Measure these effects in ramps (if possible)
    - Provide method using "external" data to correct for these effects.
    Will use loaded data from extio.py 
"""

import numpy as np

def bfe_kernel(values) : 
    """BFE kernel definition estimated from caracterization campaign.
    
    Args:
        values (ndarray-like): List or array of kernel coefficients for Brighter-Fatter
    Returns:
        ndarray : BFE kernel to be applied. 
    """
    return None

def ipc_kernel(values, kernel=5,  nl=False) : 
    """IPC kernel definition

    Args:
        values (ndarray-like): List or array of kernel coefficients for IPC
        kernel : Kernel shape (3x3) , (5x5). Proably depends on nl kwargs. 
        nl (bool, optional): Whether to include non-linear IPC correction. Defaults to False.
    Returns:
        ndarray : NxN kernel to be applied. 
    """
    kernel = np.zeros((kernel, kernel))
    np.fill_diagonal(kernel, values)
    
    return kernel


class PersistenceModel: 
    def __init__(self, model=None, metadata={}):
        self.model = model
        self.metadata = metadata    
            
    def get_persistence_correction_adu(self, t):
        gain = self.get_gain()
        dt = self.get_last_obs_dt()  # x would be last recorded fluence
        x, exptime = self.get_last_fluence()
        x  = x * gain * exptime #Need to think about units.
        # from tabulated data : 
        a_pers, x0, alpha, dx = self.get_tabulated_values()
        # BEWARE.
        a_pers = self.time_independent_fermi(a_pers, x, x0, alpha, dx)
        return a_pers*gain
        
    def time_independent_fermi(self, a_pers, x, x0, alpha, dx):
        return a_pers*(x/x0)**alpha/(1.+np.exp(-(x - x0)/dx))
    
    @staticmethod
    def power_law_persistence(t, a_pers, dt_obs, gamma):
        return a_pers * ((t+dt_obs)/1000)**(-gamma)
        
    def get_last_fluence(self): 
        from .io import get_previous_exposure
        exp = get_previous_exposure(self.metadata)
        if exp is not None : 
            dt = exp.obstime 
            x = exp.slope
        else : 
            x = 1
            dt = 1
            
        return x, dt 
    
    def get_tabulated_values(self):
        return  1., 1., 1., 1.
        
    
    
