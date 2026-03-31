
import numpy as np
from .refpixels import get_refpix_ud, get_refpix_lr

class BaseRamp( ): 
    """
    General function to define a ramp.
    Will probably be declined depending what kind of data it is : 
    (e.g flat, dark etc)
    
    """
    def __init__(self, data, times, 
                 tgroup=None, 
                 nrefpixels=4,
                 nchannels=32,
                 # 
                 # refpix=None, filename=None, 
                 # header=None, 
                 # active_pixels=slice(4, -4),
                 # npix_channel = None,
                 # include_reset=False
                ):
        """Ramp class to read and process a given ramp.

        Parameters:
        -----------
        data: ndarray
            data cube of the ramp (ngroups, *img_shape).
        times: ndarray
            time stamp of group of the data cube. 
            (a group can have a single frame).
        tgroup: array, float, None
            time of a group. Could vary. 
            If None, the data cube is expected to have no-drop and tgroup is derived 
            by using np.diff on times.
        nrefpixels: int
            number of pixel around the active data cube that serve as reference pixels.
            Hence, width and height are nrefpixels*2 larger for the data cube that 
            for the active cube.
        nchannels: int
            number of channel in the detector. 
            This automatically sets npix_per_channel.
        """
        self._data = np.asarray(data)
        self._times = np.asarray(times)
        self._tgroup = tgroup
        self._nrefpixels = int(nrefpixels)
        self._nchannels = int(nchannels)
     
    def get_refpix_map(self, data=None, saverawref=False, npix_ref=8, 
                       active_shape=True, **kwargs):         
        """Estimate reference pixel correction

        Parameters
        ----------
        data : ndarray, optional
            Data to consider when estimating the reference pixels, by default None
        saverawref : bool, optional
            Store the references pixels into attributes, by default False
        npix_ref : int, optional
            number of reference pixels in a single row/column, by default 8

        Returns
        -------
        ndarray
            The reference pixel corrections of shape (N, Npix, Npix)
        """
        if data is None : 
            data = self.data
        
        ud_refpix = np.concatenate([data[:, :self.nrefpixels, :], 
                                    data[:, -self.nrefpixels:, :]], 
                                    axis=1).copy()
        
        ud_model = get_refpix_ud(ud_refpix, 
                                  self.ngroups, 
                                  self.nchannels, 
                                  self.npix_per_channel, 
                                  npix_ref=self.nrefpixels*2)

        lr_model = get_refpix_lr( data-ud_model[:, None, :], 
                                  self.npix_per_channel, 
                                  npix_ref=self.nrefpixels,
                                  **kwargs)    
        if active_shape:
            ud_model = ud_model[:, self.nrefpixels:-self.nrefpixels]
            
        return ud_model[:, None, :] + lr_model[:, :, None]
        
    # ================ #
    #   properties     #
    # ================ #    
    @property
    def data(self):
        """ datacube to be fitted """
        return self._data

    @property
    def times(self):
        """ """
        return self._times

    @property
    def tgroup(self):
        """ """
        return self._tgroup

    @property
    def ngroups(self):
        """ """
        return self._data.shape[0]

    @property
    def nchannels(self):
        """ """
        return self._nchannels

    @property
    def npix_per_channel(self):
        """ """
        return self.data.shape[2]//self.nchannels
        
    @property
    def nrefpixels(self):
        """ """
        return self._nrefpixels
        
