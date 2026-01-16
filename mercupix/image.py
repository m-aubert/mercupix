"""
This script will define all the processing applied to exposures.
These store the results from the ramp-fitting.
"""
import numpy as np
import matplotlib.pyplot as plt

class Exposure: 
    """ Low-level class defining exposure (2D image extracted from Ramp).
    """
    def __init__(self, data=None, mask=None): 
        self.data = data
        self.mask = mask
        
    def to_fits(self):
        pass
    
    def _header(self):
        pass 
    
    def show(self,data=None, fig=None, ax=None, **kwargs):
        if fig is None and ax is None : 
            fig, ax = plt.subplots()
        if ax is None and fig is not None : 
            ax = fig.add_subplot()
            
        if data is None : 
            data = self.data 

        vmin, vmax = np.percentile(data.ravel() , [1,99])
        
        dp = ax.imshow(data, vmin = vmin, vmax=vmax, **kwargs)
        fig.colorbar(dp, ax=ax, shrink=0.4)
        return fig
        
class SciExposure(Exposure):
    """
    Sci-exposure with detrending.
    Args:
        Exposure (_type_): _description_
    """
    def __init__(self, data=None, mask=None):
        super().__init__(data, mask)
        
    def corr_flatfield(): 
        pass
    
    