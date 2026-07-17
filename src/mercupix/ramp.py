import numpy as np
import skimage as ski
from .baseramp import BaseRamp
  
class RawRamp(BaseRamp):   
    """
    Ramp class that process Raw data.
    """    

    def _from_header_to_metadata(header): 
        """
        To be filled function. 
        Processes data header to extract metadata.
        
        Parameters
        ----------
        header : fits.Header object.
        
        Returns
        -------
        
        dict 
            Dict-like object with stored metadata.
            Path to reference files: 
                e.g : caracterisation files and external data needed for processing
        """
        pass 

    @property
    def metadata(self): 
        return None
        
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
            return  self.data[:, self.actslice, self.actslice] - self.get_refpix_map(**kwargs)
        else : 
            return data - self.get_refpix_map(data=data, **kwargs)[:,:,self.actslice]   
        
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
        
    def correct_dark(self, 
                     data=None, 
                     metadata={}, 
                     darkdata=None):
        
        if data is None : 
            data = self.correct_refpix()        
            
        if darkdata is None:
            darkdata = 0
            
        if bool(metadata): 
           darkdata = get_dark(metadata['dark'])
        
        return data - darkdata*self.times  
        
    def get_mask(self, data=None,
                 disconnected=True,
                 cosmics=False, 
                 saturated=False,
                 hot=False,
                 charge_migration=False,
                 warm_threshold=0.15, 
                 hot_threshold=0.5,
                 sat_threshold=1.,
                 inpaint=False,
                 cr_threshold_mad=9): 
        """Create a bitmask map of the ramp

        Parameters
        ----------
        disconnected : bool, optional
            Track disconnected or dark pixels, by default True
        cosmics : bool, optional
            Track cosmic tracks, by default False
        saturated : bool, optional
            Track saturated pixels, by default False
        hot : bool, optional
            Track noisy or hot pixels, by default False
            
        Returns
        -------
        ndarray
            A shape (N, Nactpix, Nactpix) bitmask.
        """        
        # Initialize mask 
        mask = np.zeros(self.active_shape, dtype='uint8')
        
        if disconnected : 
            from .mask import get_downpix
            path = None if self.metadata is None else self.metadata['path']['DIS'] 
            mask += get_downpix(path=path)
        
        if hot : 
            from .mask import get_hotpix 
            # flag hot and warm 
            path = None if self.metadata is None else self.metadata['path']['HOT'] 
            mask += get_hotpix(path=path, 
                               warm_threshold=warm_threshold,
                               hot_threshold=hot_threshold)
                        
        if saturated: 
            from .mask import get_saturated
            # raw uint16 input, expands mask dims here.
            # Explicit sum cause inplace add does not broadcast.
            sat_mask = get_saturated(self.data, #Specifically use int raw data for saturation.
                                  sat_threshold=sat_threshold)[:, self.actslice, self.actslice] 
        
            mask = mask + sat_mask
            
        if charge_migration : 
            from .mask import get_charge_migration
            mask_cm = get_charge_migration(sat_mask)
            
            mask = mask + mask_cm

        if cosmics : 
            from .mask import get_cosmics
            maski , data = get_cosmics(data=data, 
                                cr_threshold_mad=cr_threshold_mad,
                                inpaint=inpaint)
            
            mask = mask + maski #For some dtype, inplace sum does not work. Document for next time.
                        
        return mask, data
        
    def correct_ipc(self, data=None, 
                          nl=False, 
                          metadata={}, 
                          ipckernel=np.ones((5,5))):
        """Method to correct for Inter-pixel capacitance frame per frame.

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
        
        if ipckernel is None : 
            from .io import load_ipc
            from .calib import ipc_kernel 
            
            # Will probably define a kernel model (3x3) or (5x5).
            ipckernel = ipc_kernel(load_ipc(metadata['ipc'], nl=nl))
                    
                
        return ski.restoration.richardson_lucy(data, ipckernel, num_iter=5, clip=False)
        
            
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
            
            nl_coefficients = load_nl_coefficients(path=metadata['nl'])
            
        correction_term = 1
        for i, coef in enumerate(nl_coefficients): 
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
            var /=np.diff(self.times)[:,None,None]
            var = np.var(var,ddof=1, axis=0)
        
        return var
        
    
    def preprocess_ramp(self,           
                        data= None, 
                        corr_refpix=True,
                        corr_ipc=False,
                        corr_ipcnl=False,
                        corr_offset=False,
                        corr_dark=False, 
                        corr_nl=False, 
                        corr_cosmics=False,
                        mask=False,
                        mask_kwargs={},
                        metadata={},
                        nl_coefficients=None,
                        ipckernel=np.ones((5,5)),
                        darkdata=None,
                        offset=0):
        """General function that successively apply correction *prior* to fitting the ramp.

        Parameters
        ----------
        data : ndarray, optional
            data to process. If none, will use the loaded one, by default None
        corr_refpix : bool, optional
            Apply reference pixel correction, by default True
        corr_ipc : bool, optional
            Apply ipc correction, by default False
        corr_ipcnl : bool, optional
            Apply non-linear ipc correction, by default False
        corr_offset : bool, optional
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
        if corr_refpix : 
            data = self.correct_refpix(data=data)

        if data is None : 
            data = self.data[:, self.actslice, self.actslice].astype("float32")
                                
        if corr_offset :             
            data= self.subtract_offset(data=data,
                                        metadata=metadata,
                                        offset=offset) #Basically bias here.
        if corr_ipc : 
            data= self.correct_ipc(data=data, 
                                        nl=corr_ipcnl, 
                                        metadata=metadata,
                                        ipckernel=ipckernel)    
        
        if corr_nl : 
            data= self.correct_nl(data=data,
                                       metadata=metadata,
                                       nl_coefficients=nl_coefficients)

        if corr_dark : 
            data= self.correct_dark(data=data,
                                    metadata=metadata,
                                    darkdata=darkdata)
        
        varcube = self.get_variance(data=data) # Unclear when and what
        
        if mask: 
            mask, data = self.get_mask(data=data, inpaint=corr_cosmics, **mask_kwargs)
            
        else : 
            if corr_cosmics : 
                from .mask import get_cosmics
                _, data = get_cosmics(data=data, inpaint=corr_cosmics, 
                                      cr_threshold_mad=mask_kwargs.pop('cr_threshold_mad', 7))
                
            mask = np.zeros(data.shape, dtype=np.int8) 

        return (data, varcube, mask)
        
    