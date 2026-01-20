import numpy as np
import h5py
from astropy.io import fits
from .utils import get_refpix_ud, get_refpix_lr
import matplotlib.pyplot as plt

  
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
                 grouped=False):
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
    
    @classmethod
    def from_filename(cls,filename, frame_path=None, time_arr_path=None):
        if 'fits' in filename : 
            with fits.open(filename) as hdul : 
                header = hdul[0].header
                cube = hdul[1].data[:,:,:-128] # Read-in cube and discard guide pixels.
                
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


    def get_refpix(self, saverawref=True, npix_ref=8): 
        """Method to estimate reference pixel correction map per frame.

        Args:
            saverawref (bool, optional): Whether to store raw reference in attributes. Defaults to True.
            npix_ref (int, optional): number of reference pixels in a single row/column. Defaults to 8.

        Returns:
            ndarray : Reference map of shape (N, Npix, Npix)
        """
        if saverawref: 
            self.ud_refpix = np.concatenate([self.datacube[:,:4,:],
                                              self.datacube[:,-4:,:]], 
                                              axis=1).copy()
            
            self.lr_refpix = np.concatenate([self.datacube[:,:,:4], 
                                             self.datacube[:,:,-4:]], 
                                             axis=1).copy()

        ud_refpix = self.ud_refpix.copy()
        ud_refpix = get_refpix_ud(ud_refpix,self.nframes, self.channel, self.npix_channel, 
                                  npix_ref=npix_ref)    

        lr_refpix = get_refpix_lr(self.datacube-ud_refpix[:,None,:], self.npix_channel, 
                                  polyorder=3, red_fac=0.875)    
            
        return ud_refpix[:,None,:]+lr_refpix[:,:,None]

    def correct_refpix(self):
        """Apply reference pixel correction

        Returns:
            ndarray : Reference pixel corrected of shape (Ngroup, Nactpix, Nactpix)
        """
        return  self.get_activecube() - self.get_refpix()[:,:,self.actslice]   
    
    def get_activecube(self):
        """Get active pixel cube, discards reference pixels.

        Returns:
            ndarray : datacube of shape (N, Nactpix, Nactpix).
        """
        return self.datacube[:,self.actslice, self.actslice]
        
    @property
    def cube_shape(self):
        """Provides the shape of a single frame in the ramp.
        Returns:
            tuple of ints : shape of a single frame.
        """
        return (self.datacube.shape[1], self.datacube.shape[2])

    @property
    def act_shape(self):
        """Provides the shape of a single frame in the ramp, excluding reference pixels.
        
        Returns:
            tuple of ints : shape of active pixels in a single frame.
        """
        _ , shape1, shape2 = self.get_activecube().shape
        return (shape1, shape2)

    @property
    def nframes(self):
        """Number of frames defining the datacube.
        
        Notes :   
            If data is grouped, gives the total number of frames : nframes > ngroups.
            If data is not grouped, ngroups and nframes provide the same information.

        Returns:
            int : Number of frames.
        """
        return self.ngroups * self.nfpgroups

    @property
    def exptime(self):
        """Exposure time for the ramp.
        
        Returns:
            float : Exposure time
        """  
        return self.header['EXPTIME'] #Might not work for H4RG
    
    @property
    def t_read(self):
        """Time-array recording time per frame.

        Returns:
            ndarray : Time for each frame.
        """
        return np.linspace(0,self.exptime, num=self.nframes)
    
    @property
    def t_gread(self):
        """Time-array recording time per group.

        Returns:
            ndarray : Time for each group.
            
        Notes : 
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

        Returns:
            int 
        
        Notes : 
            If data is not grouped, equals number of frames.
        """
        return self.datacube.shape[0]

    @property
    def nfpgroups(self):
        """Number of frames per group

        Returns:
            int or ndarray : int or ndarray depending on macc mode.
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

        Args:
            frame (ndarray, optional): A frame to display. Defaults to None.
            frame_num (int, optional): A frame number to select. Defaults to 0.
            fig (plt.Figure, optional): A figure instance. Defaults to None.
            ax (plt.Axe, optional): An axe. Defaults to None.
            corr_refpix (bool, optional): Whether to correct the reference pixel on selected frame if frame is None. 
                                          Defaults to False.

        Returns:
            fig : plt.Figure on which plot was plotted.
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

    def show_pixramp(self, pixramp=None ,pixel_num=0,  fig=None, ax=None, corr_refpix=False, **kwargs):
        """Utility function to display pixel ramps in a cube.

        Args:
            pixramp (ndarray, optional): A ramp of pixel to display of shape (Ngroup,). Defaults to None.
            pixel_num (int or list of ints or slice, optional): Pixel(s) for which to show the ramp. Can be a slice.
                                                                Defaults to 0.
            fig (plt.Figure, optional): A figure instance. Defaults to None.
            ax (plt.Axe, optional): An axe. Defaults to None.
            corr_refpix (bool, optional): Whether to correct the reference pixel on selected frame if frame is None. 
                                          Defaults to False.
                                          
        Returns:
            fig : plt.Figure on which plot was plotted.
        """
        
        if fig is None and ax is None : 
            fig, ax = plt.subplots()
        if ax is None and fig is not None : 
            ax = fig.add_subplot()
                
        pixel_num = np.atleast_1d(pixel_num) if (type(pixel_num) != slice) else pixel_num
        
        if pixramp is None :        
            if not corr_refpix : 
                pixramp = self.datacube.reshape(self.ngroups, -1)[:, pixel_num]
            else : 
                pixramp = self.correct_refpix().reshape(self.ngroups, -1)[:, pixel_num]

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
        """Subtract an offset frame to data

        Args:
            data (array, optional): Data to correct. Defaults to None.
            metadata (dict, optional): Metadata storing external data, here dark. 
            offset (array, optional): Offset to subtract. 
                                      ! For now set to 0 for debuging stage !

        ..Note : Inplace modification of the time array to account for dimension changes. 
        Will probably yield new class in the future. 

        Returns:
            ndarray : corrected data.
        """
        if data is None : 
            data = self.correct_refpix()

        if offset is not None: 
            data -= offset
            
        else :
            if not metadata: 
                data = data[1:,:,:] - data[0,:,:]
                self._t_read = self._tread[1:]
            else : 
                from .io import get_dark
                offset = get_dark(metadata['dark'])
                
        return data
        
        
    def get_mask(self, 
                 disconnected=True,
                 cosmics=False, 
                 persistence=False, 
                 saturated=False,
                 noisy=False): 
        """Routine to identifiy pixel to mask in ramp.

        Args:
            disconnected (bool, optional): Remove disconnected/dark pixels. Defaults to True.
            cosmics (bool, optional): Mask cosmic tracks. Defaults to False.
            persistence (bool, optional): Mask persistence. Defaults to False.
            saturated (bool, optional): Mask saturated pixel. Defaults to False.
            noisy (bool, optional): Mask noisy/hot pixels. Defaults to False.

        Returns:
            ndarray : Boolean (N, Npix, Npix) mask array.
        """
        
        # Initialize mask
        mask = np.zeros(self.sci_shape, dtype=bool)
        if disconnected : 
            mask &= self.get_downpix()
        
        if noisy : 
            mask &= self.get_noisy()
            
        if saturated : 
            mask &= self.get_saturated()
        
        if cosmics : 
            mask &= self.identify_cosmics()

        if persistence: 
            mask &= self.get_persistence_pixels()
            
        return mask
        
    def get_noisy(self, data=None):
        """Identify noisy pixel in data.
        Args:
            data (_type_, optional): Data cube (N, Npix, Npix). Defaults to None.
        Returns:
            ndarray : Boolean (N, Npix, Npix) array where True means noisy pixels.
        """
        # Boolean
        return True
    
    def get_saturated(self, data=None):
        """Identify saturated pixel in data.
        Args:
            data (_type_, optional): Data cube (N, Npix, Npix). Defaults to None.
        Returns:
            ndarray : Boolean (N, Npix, Npix) array where True means saturated pixels.
        """
        pass
    
    def get_persistence_pixels(self, data=None):
        """Identify persistent pixels
        Args:
            data (_type_, optional): Data cube (N, Npix, Npix). Defaults to None.
        Returns:
            ndarray : Boolean (N, Npix, Npix) array where True means persistent pixels.
        """
        pass
    
    def identify_cosmics(self, data=None): 
        """Identify cosmic track pixels
        Args:
            data (_type_, optional): Data cube (N, Npix, Npix). Defaults to None.
        Returns:
            ndarray : Boolean (N, Npix, Npix) array where True means cosmics.
        """
        # Boolean
        return True
    
    def get_downpix(self, path=None):
        """ Recover dark/disconnected pixels
        Args:
            path (_type_, optional): Path to disconnected pixel mask from metadata.
        Returns:
            ndarray : Boolean (N, Npix, Npix) array where True means disconnected.
        """
        from .io import load_downpix
        # path tbd
        mask = load_downpix(path=path)
        return mask
     
    def correct_persistence(self,data=None,
                                 metadata=None):
        """ Method to correct for persistence effect in datacube. 
        
        Args:
            data (ndarray, optionnal) : Array to which correction is applied
            path (_type_, optional): Metadata to consider for recovering persistence corrections.
        Returns:
            ndarray : Boolean (N, Npix, Npix) array where True means disconnected.
            
        Notes: 
            Not to be confused with persistence mask. Different way of handling it.
        """
        # some stuff to detect persistence in data and predict flux.
        pass
        
    def correct_bfe(self, data=None,
                          metadata={}):
        """Method to correct for Brighter-Fatter effect in ramp.

        Args:
            data (ndarray, optionnal) : Array to which correction is applied
            metadata (dict, optional): Metadata to consider for loading bfe_kernel.
        """
        # Not sure if should applied at this stage
        from .io import load_bfe
        from .calib import bfe_kernel 
                
        k_bfe = bfe_kernel(load_bfe(metadata['bfe']))
        
        pass
    
    def correct_ipc(self, data=None, 
                          nl=False, 
                          metadata={}):
        """Method to correct for Inter-pixel capacitance.

        Args:
            data (ndarray, optionnal) : Array to which correction is applied.
            nl (bool, optionnal) : If True, also consider non-linear IPC correction.
            metadata (dict, optional): Metadata to consider for loading ipc_kernel.
        """
        from .io import load_ipc
        from .calib import ipc_kernel 
        
        # Will probably define a kernel model (3x3) or (5x5).
        k_ipc = ipc_kernel(load_ipc(metadata['ipc'], nl=nl))
                
        pass
            
    def correct_nl(self, data=None, 
                         metadata={}):
        """Method to correct for classical non-linearity.

        Args:
            data (ndarray, optionnal) : Array to which correction is applied.
            metadata (dict, optional): Metadata to consider for loading corrections coefficients.
        """
        from .io import load_nl_coefficients
        
        a , b, c = load_nl_coefficients(path=metadata['nl'])
        # Do something        
        pass
        
    def correct_burnin(self):
        """Method to correct for burnin effect in ramp 
            
        """
        pass
    
    def correct_crnl(self):
        """Method to correct for count-rate (cr) non-linearity
            
        """
        pass
    
    def get_variance(self,data=None):
        """Estimate variance in the datacube.

        Args:
            data (ndarray, optional): Datacube. Defaults to None.

        Returns:
            ndarray : variance
        """
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
                        corr_burnin=False,
                        corr_nl=False, 
                        corr_crnl=False,
                        corr_bfe=False,
                        persistence=False,
                        mask=False,
                        mask_kwargs={},
                        metadata={}):
        """Function to pre-process ramp up before fitting flux.

        Args:
            ramp (_type_): _description_
            corr_refpix (bool, optional): _description_. Defaults to False.
            corr_ipc (bool, optional): _description_. Defaults to False.
            corr_ipcnl (bool, optional): _description_. Defaults to False.
            corr_dark (bool, optional): _description_. Defaults to False.
            corr_burnin (bool, optional): _description_. Defaults to False.
            corr_nl (bool, optional): _description_. Defaults to False.
            corr_crnl (bool, optional): _description_. Defaults to False.
            corr_bfe (bool, optional): _description_. Defaults to False.
            persistence (bool, optional): _description_. Defaults to False.
            mask (bool, optional): _description_. Defaults to False.
            mask_kwargs (dict, optional): _description_. Defaults to {}.

        Returns:
            _type_: _description_
        """
        
        if datacube is None : 
            datacube = self.get_activecube()
            
        if corr_refpix : 
            datacube = self.correct_refpix()
        
        if corr_ipc : 
            datacube = self.correct_ipc(data=datacube, 
                                        nl=corr_ipcnl,
                                        metadata=metadata)
        if corr_dark :             
            datacube = self.subtract_offset(data=datacube,
                                            metadata=metadata)
        
        if corr_burnin : 
            datacube = self.correct_burnin(data=datacube)
            
        if persistence: 
            datacube = self.correct_persistence(data=datacube,
                                                metadata=metadata)
            
        if corr_nl : 
            datacube = self.correct_nl(data=datacube,
                                       metadata=metadata)
        
        if corr_bfe:    
            datacube = self.correct_bfe(data=datacube,
                                        metadata=metadata)
            
        if corr_crnl:
            datacube = self.correct_crnl(data=datacube,
                                         metadata=metadata)
        
        varcube = self.get_variance(data=datacube) # Unclear when
        
        if mask: 
            mask = self.get_mask(**mask_kwargs)
            
            return datacube, varcube, mask

        else : 
            return datacube, varcube
        
        
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
        
        
        
    