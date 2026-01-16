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
                 scislice=slice(4,-4),
                 resolution=None, 
                 npix_channel = None,
                 grouped=False):
        
        self.datacube = datacube
        self.filename = filename
        self.detectype = detectype 
        self.scislice = scislice
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
                cube = hdul[1].data[:,:,:-128] 
                
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
                # Match time path (for time array)
                
                time_arr_path = [path for path in list_of_names if 'measures' in path]
                #print(time_arr_path)
                time_arr_path.sort(key=lambda x : int(x.split('/')[1].split('_')[-1]))    
                time_arr_path = time_arr_path[0]
                
            cube = f[frame_path][()][:,:,64:]
            tarr = f[time_arr_path]['time'] 

            header = dict(texp = (tarr-tarr.min()).max(), 
                          tframe=np.diff(tarr).mean())
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
            
        self.refpix_map = ud_refpix[:,None,:]+lr_refpix[:,:,None]

    def corr_refpix(self, inplace=False): 
        if not hasattr(self, 'refpix_map'):
            self.get_refpix()
        
        corr_data = self.get_scicube() - self.refpix_map[:,:,self.scislice]        
        return corr_data
    
    def get_scicube(self):
        return self.datacube[:,self.scislice, self.scislice]
        
    @property
    def cube_shape(self):
        """Provides the shape of a single frame in the ramp.
        Returns:
            _type_: _description_
        """
        return (self.datacube.shape[1], self.datacube.shape[2])

    @property
    def sci_shape(self):
        """Provides the shape of a single sci-frame in the ramp.
            Exclude reference pixels.
        """
        _ , shape1, shape2 = self.get_scicube().shape
        return (shape1, shape2)

    @property
    def nframes(self):
        return self.ngroups * self.nfpgroups

    @property
    def texp(self):
        if not hasattr(self, '_texp'):
            self._texp = self.header['texp'] #Might not work for H4RG
        return self._texp
    
    @property
    def t_read(self):
        if not hasattr(self, '_t_read'):
            self._t_read = np.linspace(0,self.texp , num=self.nframes)
        return self._t_read
    
    @property
    def t_gread(self):
        """ Read-time per group. Should return an array of size (ngroups).
        Returns:
            _type_: _description_
        """
        
        if self.grouped : 
            if not hasattr('_t_gread'):
                # Doing nothing for now but should integrate macc modes info here.
                pass
        else : 
            self._t_gread = self.t_read
        return self._t_gread

    @property
    def ngroups(self):
        return self.datacube.shape[0]

    @property
    def nfpgroups(self):
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
        pass

    @property   
    def nrframes(self):
        pass
    
    def show_frame(self, frame_num=0,  fig=None, ax=None, corr_refpix=False, **kwargs):
        if fig is None and ax is None : 
            fig, ax = plt.subplots()
        if ax is None and fig is not None : 
            ax = fig.add_subplot()

        if not corr_refpix : 
            frame_ = self.datacube[frame_num,:,:]
        else : 
            frame_ = self.corr_refpix()[frame_num,:,:]

        vmin, vmax = np.percentile(frame_.ravel() , [1,99], **kwargs)
        
        dp = ax.imshow(frame_, vmin = vmin, vmax=vmax)
        plt.colorbar(dp, ax=ax, shrink=0.4)
        return fig

    def show_pixramp(self, pixel_num=0,  fig=None, ax=None, corr_refpix=False, **kwargs):
        if fig is None and ax is None : 
            fig, ax = plt.subplots()
        if ax is None and fig is not None : 
            ax = fig.add_subplot()
            
        pixel_num = np.atleast_1d(pixel_num)
            
        if not corr_refpix : 
            ramp_ = self.datacube.reshape(self.datacube.shape[0], -1)[:, pixel_num]
        else : 
            ramp_ = self.corr_refpix().reshape(self.datacube.shape[0], -1)[:, pixel_num]

        dp = ax.plot(self.t_read, ramp_, **kwargs)
        return fig 
    
class RawRamp(Ramp):   
    """Ramp class that process Raw data.
        Will probably evolve to take into account the different processing of 
        flats, dark and science ramps.

    Args:
        Ramp (class): A Ramp class to process
    """
    def subtract_offset(self):
        pass
        
    def get_hotpix(self):
        # Boolean
        return True
    
    def get_downpix(self):
        # From external data - disconnected pixel map or else ?
        # Boolean
        return True 
    
    def flag_persistence(self): 
        # Detect and mask persistance pixels 
        return True       
    
    def corr_nl(self):
        # IPNL
        pass 
    
    def corr_bfe(self):
        # Not sure if should applied at this stage
        pass
    
    def corr_ipc(self):
        pass
    
    def corr_dark(self):
        pass
            
    def linearize_ramp(self):
        pass
        
    def raw_to_calib(self):
        
        mask = (self.get_hotpix() & 
                self.get_downpix() & 
                self.flag_persistence())
        
        self.mask = mask 
        #missing refpix corrections. Should change shape ?
        # Should keep refpix and store as well ?
        datacube = self.corr_refpix() #Temp just to make func return something.
        self.subtract_offset()
        self.corr_nl() # Prob not here 
        self.corr_ipc() # Prob not here 
        self.corr_bfe() # Prob not here
        self.corr_dark() #Unclear if here. 
        self.linearize_ramp() # Unclear if relevant to be done here.
        return datacube  
    
  