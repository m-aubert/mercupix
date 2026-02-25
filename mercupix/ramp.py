import numpy as np
import h5py
from astropy.io import fits
from .utils import get_refpix_ud, get_refpix_lr
import matplotlib.pyplot as plt
import skimage as ski

  
class Ramp : 
    """
    General function to define a ramp.
    Will probably be declined depending what kind of data it is : 
    (e.g flat, dark etc)
    
    """
    def __init__(self, 
                 datacube=None, 
                 refpix=None, 
                 filename=None, 
                 header=None, 
                 detectype=None, 
                 actslice=slice(4,-4),
                 resolution=None, 
                 npix_channel = None,
                 grouped=False,
                 include_reset=False):
        """Ramp class to read and process a given ramp.

        Args:
            datacube (ndarray, optional): Datacube to consider (N, Npix, Npix). Defaults to None.
            refpix (ndarray, optional): Reference pixels. Defaults to None.
            filename (str, optional): Filename. Defaults to None.
            header (dict, optional): Dictionnary storing header of datacube. Defaults to None.
            detectype (str, optional): Type of detector. H4 or H2. Defaults to None.
            actslice (slice, optional): Slice defining active pixels. Defaults to slice(4,-4).
            resolution (int, optional): Number of total pixels on a single axis of datacube. Defaults to None.
            npix_channel (int, optional): Number of pixel per channel. Defaults to None.
            grouped (bool, optional): Whether datacube is storing grouped data. Defaults to False.
        """
        self.datacube = datacube
        self.filename = filename
        self.detectype = detectype 
        self.actslice = actslice
        self.ismacc = False
        self.header = header
        self.refpix = refpix
        self.detectype = detectype
        self.resolution = resolution
        self.grouped = grouped
        self.npix_channel = npix_channel
        self.channel = self.resolution//self.npix_channel
        self.include_reset= include_reset
    
    @classmethod
    def from_filename(cls,filename, frame_path=None, time_arr_path=None):
        """Class method to instantiate Cube from filename

        Parameters
        ----------
        filename : str
            Path to the file to process. Should be fits file or h5 file.
            
        frame_path : str, optional
            If hdf5 like file, frame_path in the file, by default None
        time_arr_path : str, optional
            If hdf5 like file, path to time per group or frame in the file, by default None

        Returns
        -------
        cls
            Ramp instance

        Raises
        ------
        ValueError
            If frames cannot be found in hdf5 files.
            
            
        Notes 
        -----
        Only accepts 'fits' 
        """
        if 'fits' in filename : 
            with fits.open(filename) as hdul : 
                header = hdul[0].header
                cube = np.squeeze(hdul[1].data)[:,:,:-128] # Read-in cube and discard guide pixels.
                
            detectype = 'H4'
            npix_channel = 128
            
        elif 'h5'  in filename : 
            # Mostlikely Euclid
            detectype = 'H2'
            f = h5py.File(filename, 'r')

            if frame_path is None:
                # Match frame path
                try : 
                    list_of_names = [] #Get all tree of H5PY.
                    f.visit(list_of_names.append) 
                    frame_path = [path for path in list_of_names if 'frames' in path] 
                    frame_path.sort(key=lambda x : int(x.split('/')[1].split('_')[-1]))    
                    frame_path = frame_path[0]
                except Exception as e : 
                    raise ValueError("frames does not exist in this file, is there something wrong ?")
                
                time_arr_path = [path for path in list_of_names if 'measures' in path]
                time_arr_path.sort(key=lambda x : int(x.split('/')[1].split('_')[-1]))    
                time_arr_path = time_arr_path[0]
                
            
            cube = f[frame_path][()][:,:,64:] # Read-in cube and discard guide pixels.
            tarr = f[time_arr_path]['time'] 

            header = dict(EXPTIME=(tarr-tarr.min()).max(), 
                          TFRAME=np.diff(tarr).mean())
            npix_channel = 64
            
        else : 
            raise(NotImplementedError('Other type of HxRG are not yet implemented'))

        resolution = cube.shape[2]
        return cls(datacube=cube, 
                   refpix=None, 
                   filename=filename, 
                   header=header, 
                   detectype=detectype,
                   resolution=resolution, 
                   npix_channel=npix_channel) 


    def get_refpix(self, data=None, saverawref=False, npix_ref=8):         
        """Estimate reference pixel correction of the datacube

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
            data = self.datacube
        
        ud_refpix = np.concatenate([data[:,:4,:],
                                    data[:,-4:,:]], 
                                    axis=1).copy()
        
        if saverawref: 
            self.ud_refpix = ud_refpix
            
            lr_refpix = np.concatenate([data[:,:,:4], 
                            data[:,:,-4:]], 
                            axis=2).copy()
            self.lr_refpix = lr_refpix 
            
        ud_refpix = get_refpix_ud(ud_refpix,self.nframes, self.channel, self.npix_channel, 
                                  npix_ref=npix_ref)    

        lr_refpix = get_refpix_lr(data-ud_refpix[:,None,:], self.npix_channel, 
                                  polyorder=3, red_fac=0.875)    
            
        return ud_refpix[:,None,:]+lr_refpix[:,:,None]

    def correct_refpix(self, data=None, **kwargs):
        """Apply reference pixel correction

        Parameters
        ----------
        data : ndarray, optional
            Datacube to use for refpix estimation and correction. 
            If None, uses the stored datacube passed at initialization, by default None.
            
        Returns
        -------
        ndarray
            Reference pixel subtracted datacube of shape (N, Nactpix, Nactpix),
            where Nactpix is the total number of pixels in the cube subtracted from the reference pixels.
        """        
        if data is None : 
            return  self.get_activecube() - self.get_refpix(**kwargs)[:,:,self.actslice]
        else : 
            return data - self.get_refpix(data=data, **kwargs)[:,:,self.actslice]   
    
    def get_activecube(self):
        """Retrieve the active pixel cube, the cube without reference pixels.

        Returns
        -------
        ndarray
            datacube of shape (N, Nactpix, Nactpix)
        """        
        return self.datacube[:,self.actslice, self.actslice]
        
    @property
    def cube_shape(self):        
        """Provides the shape of a single frame in the ramp.
        
        Returns
        -------
        tuple of ints 
            shape of a single frame.
        """
        return (self.datacube.shape[1], self.datacube.shape[2])

    @property
    def act_shape(self):
        """Provides the shape of active pixels of a single frame.
        
        Returns
        -------
        tuple of ints 
            shape of active pixels of a single frame.
        """
        _ , shape1, shape2 = self.get_activecube().shape
        return (shape1, shape2)

    @property
    def nframes(self):
        """Number of frames defining the datacube.

        Returns
        -------
        int
            Number of frames
            
        Notes
        -----
        If data is grouped, gives the total number of frames such that nframes > ngroups.
        If data is not grouped, ngroups and nframes provide the same information.
        """        
        return self.ngroups * self.nfpgroups

    @property
    def exptime(self):
        """Exposure time for the ramp.
        
        Returns
        -------
        float
            Exposure time
        """  
        return self.header['EXPTIME'] #Might not work for H4RG
    
    @property
    def t_read(self):
        """Time-array recording time per frame.

        Returns
        -------
        ndarray 
            Time for each frame of shape nframes.
        """
        if self.include_reset : 
            return np.linspace(0,self.exptime, num=self.nframes)
        else : 
            return np.linspace(0,self.exptime, num=self.nframes+1)[1:]
   
         
    @property
    def t_gread(self):
        """Time-array recording time per group.

        Returns
        -------
        ndarray 
            Time for each group.
            
        Notes
        -----
        If the ramp is not grouped, t_read and t_gread are equal.
        """
        if self.grouped : 
                # Doing nothing for now but should integrate macc modes info here.
                pass
        else : 
            self._t_gread = self.t_read
        return self._t_gread

    @property
    def ngroups(self):
        """Number of group in datacube.

        Returns
        -------
        int 
            Number of group in datacube.
        
        Notes  
        -----
        If data is not grouped, equals number of frames.
        """
        return self.datacube.shape[0]

    @property
    def nfpgroups(self):
        """Number of frames per group(s)

        Returns
        -------
        int or tuple of ints 
            The number of frame per groups. If groups are unevenly set up, tuple of lenght Ngroups.
        """
        if not self.grouped : 
            return 1
        else :
            if not hasattr(self, '_nfpgroups'):
                # Should integrate macc mode specs here.
                pass
                #do something
        return self._nfpgroups

    @property
    def ndropr(self):
        """Number of dropped frames in the ramp sampling.
        """
        pass

    @property   
    def nrframes(self):
        """Total number of frames in the ramp samping.
        """
        pass
    
    def show_frame(self, frame=None,  frame_num=0,  fig=None, ax=None, corr_refpix=False, **kwargs):
        """Utility function to display a single frame in a cube.

        Parameters
        ----------
        frame : ndarray, optional
            A frame to display, by default None
        frame_num : int, optional
            A frame index to select, by default 0.
        fig : plt.Figure, optional
            A matplotlib Figure instance, by default None
        ax : plt.Axes, optional
            A matplotlib Axes instance, by default None
        corr_refpix : bool, optional
             Whether to correct the reference pixel on selected frame, only if frame is None, by default False
        kwargs : 
            Keyword arguments passed to imshow.
        
        Returns
        -------
        plt.Figure
            A figure instance on which axes and plots where drawn.
        """        
        if fig is None and ax is None : 
            fig, ax = plt.subplots()
        if ax is None and fig is not None : 
            ax = fig.add_subplot()

        if frame is None : 
            if not corr_refpix : 
                frame = self.datacube[frame_num,:,:]
            else : 
                frame = self.correct_refpix()[frame_num,:,:]
        
        vmin, vmax = np.percentile(frame.ravel() , [1,99])
                
        dp = ax.imshow(frame, vmin = vmin, vmax=vmax,**kwargs)
        plt.colorbar(dp, ax=ax, shrink=0.4)
        return fig

    def show_pixramp(self, pixramp=None ,pixel_ids=[0,0],  fig=None, ax=None, corr_refpix=False, **kwargs):
        """Utility function to display pixel ramps in a cube.

        Parameters
        ----------
        pixramp : ndarray, optional
            A ramp of pixel to display of shape (Ngroup,).
            If None will consider a pixel ramp from the self.datacube instance, by default None
        pixel_ids : list of ints, slices, lists, optional
            Pixel indexes of the ramp, by default [0,0].
        fig : plt.Figure, optional
            A matplotlib Figure instance, by default None
        ax : plt.Axes, optional
            A matplotlib Axes instance, by default None
        corr_refpix : bool, optional
             Whether to correct the reference pixel on selected frame, only if frame is None, by default False
        **kwargs
            Passed to plt.plot
             
        Returns
        -------
        plt.Figure
            A figure instance on which axes and plots where drawn.
        """  
        if fig is None and ax is None : 
            fig, ax = plt.subplots()
        if ax is None and fig is not None : 
            ax = fig.add_subplot()
                    
        if pixramp is None :        
            if not corr_refpix : 
                pixramp = self.datacube[:, pixel_ids[0], pixel_ids[1]]
            else : 
                pixramp = self.correct_refpix()[:, pixel_ids[0], pixel_ids[1]]
                
        pixramp = pixramp.reshape(pixramp.shape[0], -1) if pixramp.ndim > 2 else pixramp

        if pixramp.shape[0] == self.ngroups :  
            ax.plot(self.t_gread, pixramp, **kwargs)
        else : 
            ax.plot(pixramp, **kwargs)

        return fig 
    
class RawRamp(Ramp):   
    """
    Ramp class that process Raw data.
    """    
    def subtract_offset(self,
                        data=None, 
                        metadata={},
                        offset=0):
        """Subtract an offset [or metadark] frame or cube to data. 

        Parameters
        ----------
        data : _type_, optional
            _description_, by default None
        metadata : dict, optional
            Metadata storing external data, here dark, by default {}
        offset : ndarray, optional
            offset to subtract. ! For now set to 0 for debuging stage , by default 0!
            If None and metadata is empty, the first frame of the datacube will automatically be subtracted.

        Returns
        -------
        ndarray
            The offset subtracted data.
        """        
        if data is None : 
            data = self.correct_refpix()

        if offset is None: 
            if not metadata: 
                offset = data[0,:,:]
            else : 
                from .io import get_dark
                offset = get_dark(metadata['dark'])
                
        return data - offset
        
        
    def get_mask(self, 
                 disconnected=True,
                 cosmics=False, 
                 saturated=False,
                 noisy=False, 
                 estimate_mask=False): 
        """Create a bitmask map of the ramp

        Parameters
        ----------
        disconnected : bool, optional
            Track disconnected or dark pixels, by default True
        cosmics : bool, optional
            Track cosmic tracks, by default False
        saturated : bool, optional
            Track saturated pixels, by default False
        noisy : bool, optional
            Track noisy or hot pixels, by default False
        estimate_mask : bool, optional
            Whether to estimate mask, by default False.
            Overrides all keyword parameters. 
            
        Returns
        -------
        ndarray
            A shape (N, Nactpix, Nactpix) bitmask.
        """        
        # Initialize mask
        mask = np.zeros(self.sci_shape, dtype='int8')

        if estimate_mask : 
            if disconnected : 
                mask = self.get_downpix()
            
            if noisy : 
                mask += self.get_noisy()
                
            if saturated : 
                mask += self.get_saturated()
            
            if cosmics : 
                mask += self.identify_cosmics()

            if mask.sum() == 0 : 
                from astropy.stats import sigma_clip
                
                maskdata = sigma_clip(self.datacube, sigma=17, axis=(1,2))
                mask = maskdata.mask.astype('int8')
            #if persistence: 
            #    mask &= self.get_persistence_pixels()
            
        return mask
        
    def get_noisy(self, data=None):
        """ Get identified noisy pixel in data.

        Parameters
        ----------
        data : ndarray, optional
            Data cube (N, Npix, Npix). by default None

        Returns
        -------
        ndarray
            Boolean (N, Npix, Npix) array where True means noisy pixels.
        """
        # Boolean
        # Get noisy pixels as bitmap or as boolean ?
        from astropy.stats import sigma_clip
            
        maskdata = sigma_clip(self.datacube, sigma=17, axis=(1,2))
        mask = maskdata.mask
        # For now this identifies noisy pixels in terms of delta. 
        # Sigma_clipping along time axis might help in detecting whacky pixels as well ?
        # See how this goes. 
        return mask
    
    def get_saturated(self, data=None):
        """Identify saturated pixel in data.

        Parameters
        ----------
        data : ndarray, optional
            Data cube (N, Npix, Npix). by default None
            
        Returns:
            ndarray : Boolean (N, Npix, Npix) array where True means saturated pixels.

        """
        # In the future will probably apply percentage to this value.
        # cause this one is the maximum allowed.
        
        return self.datacube == 65535
            
    def identify_cosmics(self, data=None): 
        """Identify cosmic track pixels.

        Parameters
        ----------
        data : ndarray, optional
            Data cube (N, Npix, Npix). by default None
            
        Returns:
            ndarray : Boolean (N, Npix, Npix) array where True means cosmic.
        """
        # Boolean
        # Brandt provides something in the fit.
        # Look at how it is implemented.
        # Despite jumps.
        # Flag likely cosmic -> might help in detecting the jumps.
        # Additionnaly -> levels & endpoints in terms of ADU might help in flagging persistence
        # Snowball events & all could lead to high high fluxes.
        return True
    
    def get_downpix(self, path=None):
        """Recover pixels markes as dark/disconnected.

        Parameters
        ----------
        path (str, optional) : Path to disconnected pixel mask from metadata.
            Data cube (N, Npix, Npix). by default None
            
        Returns
        -------
            ndarray : Boolean (N, Npix, Npix) array where True means disconnected.
        """
        from .io import load_downpix
        # path tbd
        mask = load_downpix(path=path)
        return mask
     
    
    def correct_ipc(self, data=None, 
                          nl=False, 
                          metadata={}, 
                          ipckernel=np.ones((5,5))):
        """Method to correct for Inter-pixel capacitance.

        Parameters
        ----------
        data : (ndarray, optional), optional
            Array to which correction is applied, by default None.
        nl : bool, optional
            If True, also consider non-linear IPC correction., by default False
        metadata : dict, optional
            Metadata to consider for loading ipc_kernel, by default empty dictionnary.
        ipckernel : ndarray, optional
            Kernel to convolve to correct IPC 
        Returns
        -------
            ndarray : corrected data cube (N,pix,pix)
        """
        from scipy.ndimage import convolve
        if ipckernel is None : 
            from .io import load_ipc
            from .calib import ipc_kernel 
            
            # Will probably define a kernel model (3x3) or (5x5).
            k_ipc = ipc_kernel(load_ipc(metadata['ipc'], nl=nl))
                    
        else : 
            k_ipc = ipckernel # OR DECONVOLVE ?
            
        return ski.restoration.richardson_lucy(data, k_ipc, num_iter=1)
        
            
    def correct_nl(self, data=None, 
                         metadata={},
                         nl_coefficients=None):
        """Method to correct for classical non-linearity.

        Parameters
        ----------
        data : (ndarray, optionnal), optional
            Array to which correction is applied, by default None.
        metadata : dict, optional
            Metadata to consider for loading corrections coefficients, by default empty dictionnary.
        nl_coefficients : list, optional
            Coefficients to apply for correction
        Returns
        -------
            ndarray : Linearized data cube (N, Npix, Npix).
        """
        
        if nl_coefficients is None : 
            from .io import load_nl_coefficients
            
            nl_coefs = load_nl_coefficients(path=metadata['nl'])
            
        correction_term = 1
        for i, coef in enumerate(nl_coefs): 
            correction_term += coef*data**i 
        
        return data*correction_term
        
    
    def get_variance(self,data=None):  
        """Estimate variance in the datacube

        Parameters
        ----------
        data : (ndarray, optionnal), optional
            Datacube to consider.

        Returns
        -------
            ndarray : Variance of the datacube, same shape as datacube. 
                Think on how to do that tbh.
        """
        
        #Clarifier ce qu'on veut. Variance temporelle ou variance par pixel ?
        var = 0
        if data is not None : 
            var = np.diff(data, axis=0) 
            var /=np.diff(self.t_read)[:,None,None]
            var = np.var(var,ddof=1)
        
        return 0
        
    
    def preprocess_ramp(self,           
                        datacube= None, 
                        corr_refpix=True,
                        corr_ipc=False,
                        corr_ipcnl=False, 
                        corr_dark=False, 
                        corr_nl=False, 
                        mask=False,
                        mask_kwargs={},
                        metadata={},
                        corr_bfe=False,
                        corr_crnl=False,
                        persistence=False,
                        corr_burnin=False,
                        nl_coefficients=None,
                        ipckernel=np.ones((5,5))):
        """General function that successively apply correction *prior* to fitting the ramp.

        Parameters
        ----------
        datacube : ndarray, optional
            datacube to process. If none, will use the loaded one, by default None
        corr_refpix : bool, optional
            Apply reference pixel correction, by default True
        corr_ipc : bool, optional
            Apply ipc correction, by default False
        corr_ipcnl : bool, optional
            Apply non-linear ipc correction, by default False
        corr_dark : bool, optional
            Subtract dark, if true apply self.subtract_offset, by default False
        corr_nl : bool, optional
            If True, apply non-linearity correction to ramp, by default False
        mask : bool, optional
            If True, compute a (N, npix,npix) bitmask, by default False
        mask_kwargs : dict, optional
            Dictionnary of keyword arguments to pass to self.get_mask, see get_mask of kwargs, by default {}
        metadata : dict, optional
            Metadata dictionnary with paths to external files, should be loaded with io.fetch_metadata, by default {}

        Returns
        -------
        datacube : ndarray
            Processed and corrected datacube of shape (N, Npix, Npix)
        varcube : ndarray
            Variance of datacube, shape (N, Npix, Npix)
        mask : ndarray
            Bitmask of datacube.shape (N,Npix,Npix) if mask is True. 
            If mask is False, bitmask is 0 everywhere.
        """     
        
        if datacube is None : 
            datacube = self.get_activecube()
            
        if corr_refpix : 
            datacube = self.correct_refpix()
                    
        if corr_ipc : 
            datacube = self.correct_ipc(data=datacube, 
                                        nl=corr_ipcnl, # C-NL (adu not flux)
                                        metadata=metadata,
                                        ipckernel=ipckernel)        
        if corr_dark :             
            datacube = self.subtract_offset(data=datacube,
                                            metadata=metadata)
        
        #if corr_burnin : 
        #    datacube = self.correct_burnin(data=datacube)
            
        #if persistence:  # CalibRamp / model -> to_fit ramp
        #    datacube = self.correct_persistence(data=datacube,
        #                                        metadata=metadata)
            
        #if corr_bfe:    # CalibRamp / model -> to_fit ramp
        #    datacube = self.correct_bfe(data=datacube,
        #                                metadata=metadata)
        
        if corr_nl : 
            datacube = self.correct_nl(data=datacube,
                                       metadata=metadata,
                                       nl_coefficients=nl_coefficients)
        
        varcube = self.get_variance(data=datacube) # Unclear when and what
        
        if mask: 
            mask = self.get_mask(**mask_kwargs)
        else : 
            mask = np.zeros(datacube.shape, dtype=np.int8) 

        return (datacube, varcube, mask)
        
        
class LinRamp: 
    """Should store result of corrections. Easier to store additional info such as number of groups etc.
        TBD
    """
    def __init__(self, 
                 datacube=None, 
                 varcube=None,
                 mask=None,
                 header=None, 
                 grouped=False):
        
        self.datacube = datacube
        self.ismacc = False
        self.header = header
        self.grouped = grouped
        self.varcube = varcube
        self.mask = mask 
        
        
        
    