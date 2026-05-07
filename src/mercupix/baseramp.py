
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
                 tframe=None,
                 nrefpixels=4,
                 nchannels=32):
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
        self._tframe = tframe
        self._nrefpixels = int(nrefpixels)
        self._nchannels = int(nchannels)
        self._actslice = slice(self._nrefpixels,-self._nrefpixels)
        self._frame_shape = self._data.shape[1:] # Check
        self._act_shape = np.asarray(self._frame_shape)-self._nrefpixels*2
        
    def get_refpix_map(self, data=None, active_shape=True, **kwargs):         
        """Estimate reference pixel correction

        Parameters
        ----------
        data : ndarray, optional
            Data to consider when estimating the reference pixels, by default None

        active_shape : bool, optional
            Remap the reference pixel map to the active-shape of the detector,
            by default True
        
        **kwargs:
            Keyword arguments passed to the left-right pixel correction
        
        See also
        --------
        `mercupix.refpixels.get_refpix_lr`

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
                                  npix_ref=self.nrefpixels)

        lr_model = get_refpix_lr( data-ud_model[:, None, :], 
                                  self.npix_per_channel, 
                                  npix_ref=self.nrefpixels,
                                  **kwargs)    
        if active_shape:
            ud_model = ud_model[:, self._actslice]
            
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
    
    @property
    def active_shape(self):
        return self._act_shape
    
    @property
    def frame_shape(self):
        return self._frame_shape
    
    @property
    def actslice(self):
        """ """
        return self._actslice
    
    @property
    def cube_shape(self):
        return (self.ngroups, self.frameshape[0], self.frameshape[1])
        
    # ------- #
    #   I/O   #
    # ------- #
        
    @classmethod
    def from_filename(cls,filename, 
                      datatype='pyxim',
                      nrefpixels=4,
                      nchannels=32,
                      **kwargs):
        """Create Ramp from filename.

        Parameters
        ----------
        filename : str
            Path to the file to process. Should be fits file or h5 file.
            
        datatype : str, optional
            - 'pyxsim' for pyxim format data [fits]
            - 'roman' for roman flight-like format data [fits]
            - 'ip2i' for ip2i format data [hdf5]
            - None for anyfits format data. times will be set to empty. 
        
        **kwargs : 
            keyword arguments passed to from_ip2i.
        
        Returns
        -------
        cls
            Ramp instance        
        """
        if datatype == 'roman' : 
            from .io import from_roman 
            data, tframe, tgroup, nchannels, times = from_roman(filename)
        
        if datatype == 'pyxim' : 
            from .io import from_pyxim 
            data, tframe, tgroup, nchannels, times = from_pyxim(filename)
            
        if datatype == 'ip2i' : 
            from .io import from_ip2i
            data, tframe, tgroup, nchannels, times = from_ip2i(filename, **kwargs)
        if datatype is None :
            from .io import from_anyfits
            
            data = from_anyfits(filename)
            tframe = None 
            tgroup = None 
            times = []

        return cls(data, times, 
                 tgroup=tgroup, 
                 nrefpixels=nrefpixels,
                 nchannels=nchannels,
                 tframe=tframe)
        
    # ==================== #  
    #   Plotting utility   #
    # ==================== #  
        
    def show_frame(self, frame=None,  frame_num=0, corr_refpix=False, **kwargs):
        """Utility function to display a single frame in a cube.

        Parameters
        ----------
        frame : ndarray, optional
            A frame to display, by default None
        frame_num : int, optional
            A frame index to select, by default 0.
        corr_refpix : bool, optional
             Whether to correct the reference pixel on selected frame, only if frame is None, by default False
        **kwargs
            Passed to utils.plot_frame
            
        Returns
        -------
        plt.Figure, plt.Axes
            Figure and Axes instance on which axes and plots where drawn.
        """
        from .utils import plot_frame
                
        if frame is None : 
            if not corr_refpix : 
                frame = self.data[frame_num,:,:]
            else : 
                frame = self.correct_refpix()[frame_num,:,:]
                
        fig, ax = plot_frame(frame, **kwargs) 
        
        return (fig, ax)

    def show_pixramp(self, pixramp=None, 
                           pixel_ids=[0,0], 
                           corr_refpix=False, 
                           x_time = True, 
                           **kwargs):
        """Utility function to display pixel ramps in a cube.

        Parameters
        ----------
        pixramp : ndarray, optional
            A ramp of pixel to display of shape (Ngroup,).
            If None will consider a pixel ramp from the self.data instance, by default None
        pixel_ids : list of ints, slices, lists, optional
            Pixel indexes of the ramp, by default [0,0].
        corr_refpix : bool, optional
             Whether to correct the reference pixel on selected frame, only if frame is None, by default False
        **kwargs
            Passed to utils.plot_ramp
             
        Returns
        -------
        plt.Figure, plt.Axes
            Figure and Axes instance on which axes and plots where drawn.
        """  
        from .utils import plot_ramp 
        
        if pixramp is None :        
            if not corr_refpix : 
                pixramp = self.data[:, pixel_ids[0], pixel_ids[1]]
            else : 
                pixramp = self.correct_refpix()[:, pixel_ids[0], pixel_ids[1]]
                
        pixramp = pixramp.reshape(pixramp.shape[0], -1) if pixramp.ndim > 2 else pixramp
        
        if x_time : 
            x_time = self.times
        else : 
            x_time = None
        
        fig, ax = plot_ramp(pixramp, x=x_time, **kwargs)

        return (fig , ax)