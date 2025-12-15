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
    
"""