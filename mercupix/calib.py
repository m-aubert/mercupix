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

def bfe_kernel(values) : 
    """BFE kernel definition estimated from caracterization campaign.
    
    Args:
        values (ndarray-like): List or array of kernel coefficients for IPC
    Returns:
        ndarray : BFE kernel to be applied. 
    """
    return None

def ipc_kernel(values, kernel,  nl=False) : 
    """IPC kernel definition

    Args:
        values (ndarray-like): List or array of kernel coefficients for IPC
        kernel : Kernel shape (3x3) , (5x5). Proably depends on nl kwargs. 
        nl (bool, optional): Whether to include non-linear IPC correction. Defaults to False.
    Returns:
        ndarray : NxN kernel to be applied. 
    """
    return None

