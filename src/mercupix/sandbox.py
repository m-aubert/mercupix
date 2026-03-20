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
                 resolution=None, 
                 channel = None,
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
        self.channel = channel
        self.grouped = grouped
        
    @classmethod
    def from_filename(cls,filename, debug=False):

        if 'fits' in filename : 
            with fits.open(filename) as hdul : 
                header = hdul[0].header
                cube = hdul[1].data[:,:,:-128] #Make sure this is systematically true.
                #resetframe = False if header['RESETFRM'] == 'F' else True

            detectype = 'H4'
            resolution = cube.shape[2]
            channel = 128
        elif 'h5'  in filename : 
            # Mostlikely Euclid
            detectype = 'H2'
            f = h5py.File(filename, 'r')
            list_of_names = [] #Get all tree of H5PY.
            f.visit(list_of_names.append) 
            print(list_of_names)
 
            # Match frame path
            try : 
                frame_path = [path for path in list_of_names if 'frames' in path][0]
            except Exception as e : 
                raise ValueError("frames does not exist in this file, is there something wrong ?")
            # Match time path (for time array)
            time_arr_path = [path for path in list_of_names if 'measures' in path][0]

            cube = f[frame_path][()][:,:,64:]
            tarr = f[time_arr_path]['time'] 
            header = dict(texp = tarr.max(), 
                          tframe=np.diff(tarr).mean())
            resolution = cube.shape[2]
            channel = 64

            
        else : 
            raise(NotImplementedError('Other type of HxRG are not yet implemented'))

        return cls(datacube=cube, 
                   refpix=None, 
                   filename=filename, 
                   header=header, 
                   detectype=detectype,
                   resolution=resolution, 
                   channel=channel) 

    def get_refpix(self, saverawref=True): 
        #Depends on detector type I guess (HxRG) - H2 or H4.
        # Ugly func. Will need to be clearer.
        if saverawref: 
            self.ud_refpix = np.concatenate([self.datacube[:,:4,:],
                                              self.datacube[:,-4:,:]], 
                                              axis=1).copy()
            
            self.lr_refpix = np.concatenate([self.datacube[:,:,:4], 
                                             self.datacube[:,:,-4:]], 
                                             axis=1).copy()

        UD = self.ud_refpix.copy()
        UD = np.median(UD.reshape(self.nframes, 8, self.resolution//self.channel,self.channel),
                       axis=(1,3)) #Hard-coded no good.
        UD = np.repeat(UD, self.channel, axis=1)
        
        LR = np.concatenate([(self.datacube-UD[:,None,:])[:,4:-4,:4], 
                             (self.datacube-UD[:,None,:])[:,4:-4,-4:]], 
                             axis=2).mean(axis=2)
        LR = np.asarray(list(map(lambda x : savgol_filter(x,
                                                           window_length=112, 
                                                           polyorder=4), LR)))
        self.refpix_map = UD[:,None,:]+LR[:,:,None]
        #There's probably a more efficient way to do the stuff I guess / less costly.

    def corr_refpix(self): 
        if not hasattr(self, 'refpix_map'):
            self.get_refpix()
        corr_data = self.get_scicube() - self.refpix_map[:,:,self.scislice]
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
    def ngroups(self):
        return self.datacube.shape[0]

    @property
    def nfpgroups(self):
        if not self.grouped : 
            return 1
        else :
            return self.framespgp

    @property
    def ndropr(self):
        pass

    @property   
    def nrframes(self):
        pass
    
    
from scipy.special import legendre, hermitenorm


def poly_(x,deg=1, poly='mono'):
    if poly == "leg" : 
        pol = np.stack([legendre(i)(x) for i in range(deg+1)])
    elif poly == "mono" : 
        pol = np.stack([x**i for i in range(deg+1)])
    elif poly == "herm": 
        pol = np.stack([hermitenorm(i)(x) for i in range(deg+1)])
    else : 
        return None
    return pol.astype(np.float32)
    

class Rampfit: 
    """
    Should be a class dedicated to fitting stuff. Will see how this goes I guess. 
    """
    def __init__(self, cube, polytype, tarr):
        self.cube = cube
        self.polytype = polytype
        self.tarr = tarr
        if self.polytype == 'leg':
            self.tleg = tarr/tarr.max() * 2  - 1 #Pour legrendre Leander ver. -> assume t0 = 0 .

    def poly_(self, deg=1, leg=True):
        from scipy.special import legendre 
        if leg : 
            pol = np.stack([legendre(i)(self.tleg) for i in range(deg+1)])
        else : 
            pol = np.stack([self.tarr**i for i in range(deg+1)])
        return pol.astype(np.float32)

    def no_w_lstq(self, x, cube):
        cube = cube.reshape(cube.shape[0], -1).astype(np.float32)
        self.coeffs = np.linalg.lstsq(x.T, cube, rcond=None)[0]

    def fit(self, deg=1):
        x = self.poly_(deg=deg, leg=(self.polytype=='leg'))
        self.no_w_lstq(x, self.cube) 
        return self.coeffs
        
class SinglePixelRamp:
        def __init__(self, pixramp, tread):
             self.pixramp = pixramp
             self.tread = tread

        def linearize_single_pixel(self, coef, deg=3, poly='mono'): 
            lin_meas = 1 - self.pixramp / np.sum(poly_(self.tread, deg=1, poly=poly).T*coef[:2], axis=-1)
            coef_lin = np.linalg.lstsq(poly_(self.pixramp, deg=deg, poly=poly).T, lin_meas.astype(np.float32), rcond=None)[0]
            pixlin = self.pixramp * (1+(coef_lin[:,None]*poly_(self.pixramp, deg=deg, poly=poly)).sum(axis=0))
            return pixlin 
        
        def ramp_lstq(self, ramp, deg=1, poly='mono'):
            if poly == 'leg': 
                x= 2/self.tread[-1]*self.tread - 1.
            else: 
                x = self.tread
            x = poly_(x, deg=deg, poly=poly)
            coefs = np.linalg.lstsq(x.T, ramp, rcond=None)[0]
            return coefs
        
        def ramp_to_pixelflux(self, deg=1, poly='mono'):
            coefs = self.ramp_lstq(self.pixramp, deg=deg, poly=poly)
            pixlin = self.linearize_single_pixel(coefs, deg=deg, poly=poly)
            coef_lin = self.ramp_lstq(pixlin,deg=1, poly='mono')
            return coef_lin
        
class Kubik2015 : 

    def __init__(self,ramp, tpf, macc=None):
        self.diff = np.diff(ramp, axis=0)
        self.tpf = tpf
        if macc is not None : 
            self.ng, self.nf, self.nd =macc
        else :
            self.ng = ramp.shape[0]
            self.nf = 1
            self.nd = 0

    @property
    def alpha(self): 
        return (1-self.nf**2)/(3*self.nf+self.nd)
    
    def beta(self, sigma_r, fe):
        return 2*sigma_r**2 * fe / (self.nf * (1+self.alpha))
    
    def estim_f(self,fe, sigma_r):
        alpha_= self.alpha
        beta_ = self.beta(sigma_r, fe)

        term = (1+alpha_)/2*fe
        term *= np.sqrt(1 + 4*fe**2 * np.sum( (self.diff + beta_)**2) / 
                        ((self.ng+1)*(alpha_+1)**2 )) - 1
        term -= beta_
        return term
    
    def estim_sigf(self,sigma_r, fe):
        alpha_ = self.alpha
        beta_ = self.beta(sigma_r,fe)
        g_ = self.estim_f(fe, sigma_r)
        term  = 2*(g_+beta_)**2 /  ((self.ng-1)*
                                    (1+2*fe/(1+alpha_) * 
                                     (g_+beta_)))
        
        return term
