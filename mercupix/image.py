"""
This script will define all the processing applied to exposures.
These store the results from the ramp-fitting.

"""

class Exposure: 
    """ Will store the product of RampFit as 2D Exposure
    """
    def __init__(self, data=None, mask=None): 
        pass
    
    def to_fits():
        pass
    
    def _header():
        pass 
        
    
class SciExposure(Exposure):
    """
    Exposure with detrended data like flat-fielding.
    Args:
        Exposure (_type_): _description_
    """
    def __init__(self, data=None, mask=None):
        super().__init__(data, mask)
        
    def corr_flatfield(): 
        pass
    
    