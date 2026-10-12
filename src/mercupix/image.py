"""2D exposures built from the ramp fitting results.

Exposures store the outputs of the ramp fitting (flux, variance and mask)
and provide the post-processing applied to them.
"""
import numpy as np
import matplotlib.pyplot as plt


def fluxmap_to_scimg(data, error, mask, **meta): 
    """Post-process the ramp fitting outputs into a science image.

    Applies the flat-field, builds the final mask and header, and writes
    the image to a FITS file.

    Parameters
    ----------
    data : ndarray
        Flux map of shape ``(ny_act, nx_act)`` from the ramp fit.
    error : ndarray
        Variance of the flux map.
    mask : ndarray
        Data quality bitmask of the input datacube.
    **meta
        Metadata of the exposure.

    Returns
    -------
    str
        Path of the file in which the image and its metadata are stored.
    """
    cleanimg = MetaFrame(data=data, variance=error, mask=mask, metadata=meta)
    
    newdata = cleanimg.correct_flatfield()
    mask = cleanimg.get_mask()
    header = cleanimg.generate_header()
    cleanimg.to_fits(data=newdata, mask=mask, header=header)
    
    return cleanimg.fileout 
    
class Exposure: 
    """2D image extracted from a ramp.

    Parameters
    ----------
    data : ndarray, optional
        Flux map (slope) of shape ``(ny_act, nx_act)`` returned by the ramp
        fitting. Default is None.
    mask : ndarray, optional
        Integer bitmask of shape ``(ny_act, nx_act)``. Default is None.
    variance : ndarray, optional
        Variance of ``data``. Default is None.
    metadata : dict, optional
        Metadata, as returned by `mercupix.io.fetch_metadata`. Default is
        None.
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
        """Load an exposure from a FITS file.

        The file is expected to store the image, the bitmask and the
        variance in extensions 1, 2 and 3, respectively.

        Parameters
        ----------
        filename : str
            Path to the FITS file.

        Returns
        -------
        Exposure
            Instance of ``cls`` holding the loaded exposure.
        """
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
        """Build an exposure from a ramp fitting result.

        Parameters
        ----------
        result : mercupix.fitter.RampFitResults
            Output of a ramp fitter.
        metadata : dict, optional
            Metadata of the exposure. Default is None.

        Returns
        -------
        Exposure
            Instance of ``cls`` holding the slope, 2D mask and variance of
            ``result``.
        """
        return cls(data=result.slope, 
                   mask=result.mask2d, 
                   var=result.var, 
                   metadata=metadata)
                
    def to_fits(self, data=None, mask=None, variance=None):
        """Write the exposure to a FITS file.

        The file has three extensions: image (float), variance (float) and
        mask (int).

        Parameters
        ----------
        data : ndarray, optional
            Image to store. Default is None.
        mask : ndarray, optional
            Integer bitmask to store. Default is None.
        variance : ndarray, optional
            Variance to store. Default is None.

        Notes
        -----
        Not implemented yet.
        """
        # 3 extensions file : 
        # Image [float]
        # Variance [float]
        # Mask  [int]
        pass
    
    def generate_header(self):
        """Generate the FITS header of the exposure.

        The header should include the total exposure time, the number of
        bad pixels and general quality flags.

        Returns
        -------
        astropy.io.fits.Header
            Header of the exposure.

        Notes
        -----
        Not implemented yet: currently returns None.
        """
        #Should generate header.
        # Total exptime 
        # Npixel off / bad
        # General QA flags of the image ? 
        pass 
    
    def show(self,data=None, fig=None, ax=None, **kwargs):
        """Display the 2D image with percentile-based color limits.

        Parameters
        ----------
        data : ndarray, optional
            Image to display. If None, ``data`` is used. Default is None.
        fig : matplotlib.figure.Figure, optional
            Figure in which a new axes is created if ``ax`` is None.
            Default is None.
        ax : matplotlib.axes.Axes, optional
            Axes on which the image is drawn. If both ``fig`` and ``ax`` are
            None, a new figure and axes are created. Default is None.
        **kwargs
            Passed to `matplotlib.axes.Axes.imshow`.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot.
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
    """Science exposure with detrending.

    See `Exposure` for the constructor parameters.
    """
    
    def correct_flatfield(self): 
        """Apply the flat-field to the image.

        Returns
        -------
        ndarray
            Flat-fielded image.

        See Also
        --------
        mercupix.io.get_flatfield
        """
        from .io import get_flatfield
        
        flat = get_flatfield(self.metadata)
        
        return self.data/flat    
    
    def get_mask(self):
        # Generate a 2D mask from 3D mask ? 
        """Get the 2D bitmask of the exposure.

        Returns
        -------
        ndarray
            Bitmask of the exposure.

        Notes
        -----
        Currently returns ``mask`` unchanged. Collapsing a 3D mask to 2D is
        still to be decided.
        """
        mask = self.mask
        return mask 
    

class MultiRampExposure:
    """Combination of exposures from several ramps.

    Parameters
    ----------
    images : list of ndarray, optional
        Images to combine. Default is ``[]``.
    dq : list of ndarray, optional
        Bitmasks of the images. Default is ``[]``.
    var : list of ndarray, optional
        Variances of the images. Default is ``[]``.
    corr_flatfield : bool, optional
        Whether to flat-field the images. Default is False.

    Notes
    -----
    Not implemented yet.
    """
    def __init__(self, images=[], dq=[], var=[], 
                 corr_flatfield=False): 
        
        # Should we use similar collection of as ztfimg ?
        # Or something lighter ?
        
        pass

    @classmethod
    def from_filename(cls, list_of_files):
        """Load the exposures from a list of FITS files.

        Each file is expected to store the image, the bitmask and the
        variance in extensions 1, 2 and 3, respectively.

        Parameters
        ----------
        list_of_files : list of str
            Paths to the FITS files.

        Returns
        -------
        MultiRampExposure
            Instance of ``cls`` holding the loaded exposures.
        """
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
    def from_images(cls, list_of_images):
        """Build the combination from a list of images.

        Parameters
        ----------
        list_of_images : list of ndarray
            Images to combine.

        Returns
        -------
        MultiRampExposure
            Instance of ``cls`` holding the images.
        """
        return cls(images=list_of_images)
        
    def stack_images():
        """Stack the images. Not implemented yet."""
        pass 

    def combine_var():
        """Combine the variances. Not implemented yet."""
        pass 

    def combine_dq():
        """Combine the bitmasks. Not implemented yet."""
        pass 


    def write():
        """Write the combined exposure to file. Not implemented yet."""
        pass
    
    def make_header():
        """Generate the header of the combined exposure. Not implemented yet."""
        pass 