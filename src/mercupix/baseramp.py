"""Base class to store and access an up-the-ramp HxRG datacube."""

import numpy as np
from .refpixels import get_refpix_ud, get_refpix_lr

class BaseRamp( ): 
    """Generic up-the-ramp datacube with reference pixels.

    Subclasses specialize the processing depending on the kind of data
    (e.g. raw science, flat, dark).

    Parameters
    ----------
    data : ndarray
        Ramp datacube of shape ``(ngroups, ny, nx)``, including the
        reference pixels.
    times : ndarray
        Time stamp of each group, of shape ``(ngroups,)``. A group can
        contain a single frame.
    tgroup : float or ndarray, optional
        Time of a group, which may vary. If None, the datacube is expected
        to have no dropped frames. Default is None.
    tframe : float, optional
        Time of a single frame. Default is None.
    nrefpixels : int, optional
        Width, in pixels, of the reference pixel border around the active
        area. The frame is ``2 * nrefpixels`` larger than the active area
        in each dimension. Default is 4.
    nchannels : int, optional
        Number of readout channels of the detector. Sets
        `npix_per_channel`. Default is 32.
    """
    def __init__(self, data, times, 
                 tgroup=None, 
                 tframe=None,
                 nrefpixels=4,
                 nchannels=32):
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
        """Estimate the reference pixel correction.

        Parameters
        ----------
        data : ndarray, optional
            Datacube of shape ``(ngroups, ny, nx)`` from which the reference
            pixels are read. If None, ``data`` is used. Default is None.
        active_shape : bool, optional
            Whether to crop the correction to the active area of the
            detector. Default is True.
        **kwargs
            Passed to `mercupix.refpixels.get_refpix_lr`.

        Returns
        -------
        ndarray
            Reference pixel correction of shape
            ``(ngroups, ny_act, nx_act)``.

        See Also
        --------
        mercupix.refpixels.get_refpix_ud
        mercupix.refpixels.get_refpix_lr
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
        """Ramp datacube of shape ``(ngroups, ny, nx)``."""
        return self._data

    @property
    def times(self):
        """Time stamp of each group, of shape ``(ngroups,)``."""
        return self._times

    @property
    def tgroup(self):
        """Time of a group."""
        return self._tgroup

    @property
    def ngroups(self):
        """Number of groups in the ramp."""
        return self._data.shape[0]

    @property
    def nchannels(self):
        """Number of readout channels."""
        return self._nchannels

    @property
    def npix_per_channel(self):
        """Number of pixel columns per readout channel."""
        return self.data.shape[2]//self.nchannels
        
    @property
    def nrefpixels(self):
        """Width of the reference pixel border, in pixels."""
        return self._nrefpixels
    
    @property
    def active_shape(self):
        """Shape ``(ny_act, nx_act)`` of the active area."""
        return self._act_shape
    
    @property
    def frame_shape(self):
        """Shape ``(ny, nx)`` of a full frame."""
        return self._frame_shape
    
    @property
    def actslice(self):
        """Slice selecting the active area along each spatial axis."""
        return self._actslice
    
    @property
    def cube_shape(self):
        """Shape ``(ngroups, ny, nx)`` of the datacube."""
        return (self.ngroups, self.frame_shape[0], self.frame_shape[1])
        
    # ------- #
    #   I/O   #
    # ------- #
        
    @classmethod
    def from_filename(cls,filename, 
                      datatype='pyxim',
                      nrefpixels=4,
                      nchannels=32,
                      **kwargs):
        """Load a ramp from a file.

        Parameters
        ----------
        filename : str
            Path to the file to load (FITS or HDF5).
        datatype : str or None, optional
            Format of the file:

            - ``'pyxim'``: pyxim simulation format (FITS),
            - ``'roman'``: Roman flight-like format (FITS),
            - ``'ip2i'``: IP2I test bench format (HDF5),
            - None: any FITS file; `times` is left empty.

            Default is ``'pyxim'``.
        nrefpixels : int, optional
            Width of the reference pixel border, in pixels. Default is 4.
        nchannels : int, optional
            Number of readout channels. Overridden by the value read from
            the file when ``datatype`` is not None. Default is 32.
        **kwargs
            Passed to `mercupix.io.from_ip2i` when ``datatype='ip2i'``.

        Returns
        -------
        mercupix.baseramp.BaseRamp
            Instance of ``cls`` holding the loaded ramp.

        See Also
        --------
        mercupix.io.from_pyxim
        mercupix.io.from_roman
        mercupix.io.from_ip2i
        mercupix.io.from_anyfits
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
        """Display a single frame of the datacube.

        Parameters
        ----------
        frame : ndarray, optional
            2D frame to display. If None, the frame is taken from ``data``.
            Default is None.
        frame_num : int, optional
            Index of the group to display when ``frame`` is None.
            Default is 0.
        corr_refpix : bool, optional
            Whether to apply the reference pixel correction to the selected
            frame. Only used when ``frame`` is None. Default is False.
        **kwargs
            Passed to `mercupix.utils.plot_frame`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the plot.
        ax : matplotlib.axes.Axes
            Axes on which the frame is drawn.
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
        """Display the ramp of one or several pixels.

        Parameters
        ----------
        pixramp : ndarray, optional
            Pixel ramp(s) to display, of shape ``(ngroups, ...)``. If None,
            the ramps are taken from ``data``. Default is None.
        pixel_ids : list, optional
            Indices ``[i, j]`` along the two spatial axes of the pixels to
            display. Each index can be an int, a slice or a list.
            Default is ``[0, 0]``.
        corr_refpix : bool, optional
            Whether to apply the reference pixel correction to the selected
            ramps. Only used when ``pixramp`` is None. Default is False.
        x_time : bool, optional
            Whether to use `times` as abscissa instead of the group index.
            Default is True.
        **kwargs
            Passed to `mercupix.utils.plot_ramp`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the plot.
        ax : matplotlib.axes.Axes
            Axes on which the ramps are drawn.
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