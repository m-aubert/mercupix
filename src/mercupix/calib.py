"""Detector effects that bias the per-pixel flux estimate.

This module lists the effects to correct to obtain a 2D image ready to be
detrended, and either measures them in ramps (when possible) or provides
methods to correct them using external data loaded with `mercupix.io`.

Effects considered:

- classical non-linearity,
- brighter-fatter effect (BFE),
- persistence,
- burn-in,
- inter-pixel capacitance (IPC),
- non-linear IPC,
- charge diffusion.

Other effects, still to be placed in the pipeline:

- gain estimation (photon transfer curve),
- dark current (possibly after the flux estimate).

Notes
-----
This module may be merged into `mercupix.carac`.
"""

import numpy as np

def bfe_kernel(values) : 
    """Build the brighter-fatter kernel from characterization coefficients.

    Parameters
    ----------
    values : array_like
        Brighter-fatter kernel coefficients.

    Returns
    -------
    ndarray or None
        BFE kernel to be applied.

    Notes
    -----
    Not implemented yet: currently returns None.
    """
    return None

def ipc_kernel(values, kernel=5,  nl=False) : 
    """Build the IPC kernel from characterization coefficients.

    Parameters
    ----------
    values : array_like
        IPC kernel coefficients, placed on the kernel diagonal.
    kernel : int, optional
        Size of the square kernel (3 or 5). May depend on ``nl``.
        Default is 5.
    nl : bool, optional
        Whether to include the non-linear IPC correction. Default is False.

    Returns
    -------
    ndarray
        ``(kernel, kernel)`` IPC kernel to be applied.
    """
    kernel = np.zeros((kernel, kernel))
    np.fill_diagonal(kernel, values)
    
    return kernel


class PersistenceModel: 
    """Persistence model based on the fluence of the previous exposure.

    Parameters
    ----------
    model : object, optional
        Persistence model description. Default is None.
    metadata : dict, optional
        Metadata giving access to the previous exposure and the tabulated
        persistence parameters. Default is ``{}``.

    Notes
    -----
    Work in progress: gain and timing accessors are not implemented yet.
    """
    def __init__(self, model=None, metadata={}):
        self.model = model
        self.metadata = metadata    
            
    def get_persistence_correction_adu(self, t):
        """Compute the persistence signal to subtract, in ADU.

        Parameters
        ----------
        t : float or ndarray
            Time since the start of the current exposure.

        Returns
        -------
        float or ndarray
            Persistence amplitude in ADU.
        """
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
        """Evaluate the time-independent Fermi-like persistence amplitude.

        The model is ``a_pers * (x / x0)**alpha / (1 + exp(-(x - x0) / dx))``.

        Parameters
        ----------
        a_pers : float or ndarray
            Persistence amplitude normalization.
        x : float or ndarray
            Fluence of the previous exposure.
        x0 : float or ndarray
            Fluence at the transition of the Fermi function.
        alpha : float or ndarray
            Power-law index of the fluence dependence.
        dx : float or ndarray
            Width of the Fermi transition.

        Returns
        -------
        float or ndarray
            Persistence amplitude.
        """
        return a_pers*(x/x0)**alpha/(1.+np.exp(-(x - x0)/dx))
    
    @staticmethod
    def power_law_persistence(t, a_pers, dt_obs, gamma):
        """Evaluate the power-law time decay of persistence.

        The model is ``a_pers * ((t + dt_obs) / 1000)**(-gamma)``.

        Parameters
        ----------
        t : float or ndarray
            Time since the start of the current exposure.
        a_pers : float or ndarray
            Persistence amplitude.
        dt_obs : float
            Time elapsed between the previous exposure and the current one.
        gamma : float or ndarray
            Power-law decay index.

        Returns
        -------
        float or ndarray
            Persistence signal at time ``t``.
        """
        return a_pers * ((t+dt_obs)/1000)**(-gamma)
        
    def get_last_fluence(self): 
        """Get the fluence and time of the previous exposure.

        Returns
        -------
        x : float or ndarray
            Slope of the previous exposure, or 1 if none is found.
        dt : float
            Observation time of the previous exposure, or 1 if none is
            found.
        """
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
        """Get the tabulated persistence parameters.

        Returns
        -------
        a_pers, x0, alpha, dx : float
            Parameters of `time_independent_fermi`.

        Notes
        -----
        Placeholder: currently returns ``(1., 1., 1., 1.)``.
        """
        return  1., 1., 1., 1.
        
    
    
