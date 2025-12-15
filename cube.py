import numpy as np
import h5py
from astropy.io import fits
from scipy.signal import savgol_filter

class Cube : 
    """
    For now placeholder very large func. Will see how this goes. 
    -> Could be rawcube
    -> then sci cube 
    -> then something else. 
    To be discussed.

    Could be Euclid specific or Roman specific. 
    We keep best of both worlds ? 
    """
    def __init__(self, 
                 datacube=None, 
                 refpix=None, 
                 filename=None, 
                 header=None, 
                 detectype=None, 
                 scislice=slice(4,-4),
                 resolution=None):
        
        self.datacube = datacube
        self.filename = filename
        self.detectype = detectype 
        self.scislice = scislice
        self.ismacc = False
        self.header = header
        self.refpix = refpix
        self.detectype = detectype
        self.resolution = resolution        
        self.npix_c = 128 if self.detectype == 'H4' else 64



    @classmethod
    def from_filename(cls, filename):

        if 'fits' in filename : 
            with fits.open(filename) as hdul : 
                header = hdul[0].header
                cube = hdul[1].data[:,:,:-128] #Make sure this is systematically true.
                #resetframe = False if header['RESETFRM'] == 'F' else True

            detectype = 'H4'
        elif 'h5'  in filename : 
            # Mostlikely Euclid
            detectype = 'H2'
            f = h5py.File(filename, 'r')
            list_of_names = [] #Get all tree of H5PY.
            f.visit(list_of_names.append) 
 
            # Match frame path
            frame_path = [path for path in list_of_names if 'frames' in path][0]
            # Match time path (for time array)
            time_arr_path = [path for path in list_of_names if 'measures' in path][0]

            cube = f[frame_path][()][:,:,:-64]
            tarr = f[time_arr_path]['time'] 
            header = dict(texp = tarr.max(), 
                          tframe=np.diff(tarr).mean())
            
        else : 
            raise(NotImplementedError('Other type of HxRG are not yet implemented'))

        return cls.__init__(datacube=cube, 
                            refpix = None, 
                            filename=filename,
                            header=header, 
                            detectype=detectype,
                            resolution=cube.shape[-1]) 

    def get_refpix(self, H4RG=True, saverawref=True): 
        #Depends on detector type I guess (HxRG) - H2 or H4.
        # Ugly func. Will need to be clearer.
        if self.detectype == H4RG : 
            if saverawref: 
                self.ud_refpix = np.concatenate([self.datacube[:,:4,:],
                                                  self.datacube[:,-4:,:]], 
                                                  axis=1).copy()
                
                self.lr_refpix = np.concatenate([self.datacube[:,:,:4], 
                                                 self.datacube[:,:,-4:]], 
                                                 axis=1).copy()

            UD = self.ud_refpix.copy()
            UD = np.median(UD.reshape(self.nframes.shape[0], 8, self.resolution//self.npix_c,self.npix_c),
                           axis=(1,3)) #Hard-coded no good.
            UD = np.repeat(UD, 128, axis=1)
            
            LR = np.concatenate([(self.datacube-UD[:,None,:])[:,4:-4,:4], 
                                 (self.datacube-UD[:,None,:])[:,4:-4,-4:]], 
                                 axis=2).mean(axis=2)
            LR = np.asarray(list(map(lambda x : savgol_filter(x,
                                                               window_length=int(0.95*self.npix_c), 
                                                               polyorder=4), LR)))
            self.refpix = UD[:,None,:]+LR[:,:,None]
            #There's probably a more efficient way to do the stuff I guess / less costly.
        else : 
            raise(NotImplementedError('Other type of HxRG are not yet implemented'))

    def corr_refpix(self): 
        corr_data = self.get_scicube() - self.refpix[:,:,self.scislice]
        return corr_data
    
    def get_scicube(self):
        return self.datacube[:,self.scislice, self.scislice]
        
    @property
    def cube_shape(self):
        return (self.datacube.shape[1], self.datacube.shape[2])

    @property
    def sci_shape(self):
        return (self.datacube.shape[1], self.datacube.shape[2])

    @property
    def nframes(self):
        return self.ngroups * self.nfpgroups

    @property
    def t_read(self):
        return self.linspace(0,self.texp , num=self.nframes)

    @property
    def ngroups(self):
        return self.datacube.shape[0]

    @property
    def nfpgroups(self):
        pass

    @property
    def ndropr(self):
        pass

    @property   
    def nrframes(self):
        pass