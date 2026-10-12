"""
This script will define all the processing applied to exposures.
These store the results from the ramp-fitting.
"""
import numpy as np
import matplotlib.pyplot as plt


def fluxmap_to_scimg(data, error, mask, **meta): 
    """
    High-level function to post-process the processed ramp and ramp fitting outputs product 

    Parameters
    ----------
    data : ndarray
        (Nact, Nact) ndarray issued from the ramp fit
    error : ndarray
        Variance of the estimated slope
    mask : ndarray
        Mask of the initial datacube 

    Returns
    -------
    path
        File path in which the image and its metadata were stored.
    """
    cleanimg = MetaFrame(data=data, variance=error, mask=mask, metadata=meta)
    
    newdata = cleanimg.correct_flatfield()
    mask = cleanimg.get_mask()
    header = cleanimg.generate_header()
    cleanimg.to_fits(data=newdata, mask=mask, header=header)
    
    return cleanimg.fileout 
    
class Exposure: 
    """ Low-level class defining exposure (2D image extracted from Ramp).
    """
    def __init__(self, data=None, mask=None, var=None,  metadata=None): 
        """
        Exposure class

        Parameters
        ----------
        data : ndarray, optional
            ndarray of (Nactpix, Nactpix), slope of the linear fit returned by the ramp fitting procedure, by default None
        mask : ndarray, optional
            ndarray of ints, (Nactpix, Nactpix), bitmask of the image, by default None
        variance : ndarray , optional
            Variance of the data, by default None
        metadata : dict, optional
            Metadata from fetch_metadata, by default None
        """
        self.data = data
        self.mask = mask
        self.metadata = metadata
        self.var= var
        
    @classmethod
    def from_filename(cls,filename):
        from astropy.io import fits 
        
        with fits.open(filename) as hdul : 
            image = hdul[1].data
            dq = hdul[2].data
            var = hdul[3].data  
            
        #
        # Add some metadata processing with header or something else.
        #
        
        return cls(data=image, mask=dq, var=var, metadata={})
    
    
    @classmethod
    def from_fitresult(cls, result, metadata=None):
        # Some redundancy to note.                
        return cls(data=result.slope, 
                   mask=result.mask2d, 
                   var=result.var, 
                   metadata=metadata)
                
    def to_fits(self, data=None, mask=None, variance=None):
        """
        Output function that stores image in a fits file.

        Parameters
        ----------
        data : ndarray, optional
            Data to store, by default None
        mask : ndarray of int, optional
            Final mask of the , by default None
        """
        # 3 extensions file : 
        # Image [float]
        # Variance [float]
        # Mask  [int]
        pass
    
    def generate_header(self):
        """
        Function that should generate a header 
        """
        #Should generate header.
        # Total exptime 
        # Npixel off / bad
        # General QA flags of the image ? 
        pass 
    
    def show(self,data=None, fig=None, ax=None, **kwargs):
        """
        Utility function to display a 2D image.

        Parameters
        ----------
        data : ndarray, optional
            Data to display if different from self.data, by default None
        fig : plt.Figure, optional
            A maplotlib Figure instance, by default None
        ax : plt.Axes, optional
            A matplotlib Axes instance by default None

        Returns
        -------
        plt.Figure
            The displayed figure.
        """
        if fig is None and ax is None : 
            fig, ax = plt.subplots()
        if ax is None and fig is not None : 
            ax = fig.add_subplot()
            
        if data is None : 
            data = self.data 

        vmin, vmax = np.percentile(data.ravel() , [1,99])
        
        dp = ax.imshow(data, vmin = vmin, vmax=vmax, **kwargs)
        plt.colorbar(dp, ax=ax, shrink=0.4)
        return fig
        
class MetaFrame(Exposure):
    """
    Sci-exposure with detrending.
    Args:
        Exposure (_type_): _description_
    """
    
    def correct_flatfield(self): 
        """
        Apply flat-fielding to the 2D image.

        Returns
        -------
        ndarray
            The flat-fielded image.
        """
        from .io import get_flatfield
        
        flat = get_flatfield(self.metadata)
        
        return self.data/flat    
    
    def get_mask(self):
        # Generate a 2D mask from 3D mask ? 
        mask = self.mask
        return mask 
    

class MultiRampExposure:
    def __init__(self, images=[], dq=[], var=[], 
                 corr_flatfield=False): 
        
        # Should we use similar collection of as ztfimg ?
        # Or something lighter ?
        
        pass

    @classmethod
    def from_filename(cls, list_of_files):
        from astropy.io import fits
        
        list_of_images = []
        dq = []
        var = []
        for file_ in list_of_files:
            with fits.open(file_) as hdul : 
               list_of_images.append(hdul[1].data)
               dq.append(hdul[2].dq)
               var.append(hdul[3].data)    
                
        return cls(images=list_of_images, dq=dq, var=var)

    @classmethod
    def from_images(cls,list_of_images):
        return cls(images=list_of_images)
        
    def stack_images():
        pass 

    def combine_var():
        pass 

    def combine_dq():
        pass 


    def write():
        pass
    
    def make_header():
        pass 