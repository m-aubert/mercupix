
import numpy as np 
from astropy.io import fits
import h5py


def fetch_metadata(path=None):
    """Method to get metadata from path

    Args:
        path (str, optional): Filename path extracted from metadata. 
                              Defaults to None.
    Returns : 
        Metadata cls 
        
    """
    pass

class Metadata:
    
    def __init__(filename):
        """
        Placeholder class to define needed product and operations to get information.        
        """
        
    def parse_filename(filename):
        """
        parse_filename to obtain proper date.
        """
        pass

    def query_metadata(obsdate):
        """
        From obsdate, unique ID get corresponding data path
        """
        # Query nearest calib dark
        # Query nearest calib flat     
        # Query gain map (will probably be unique until better models come up) 
        # Query previous acquisition filepath 
        # Need to think of how format will look like. 
        # Used to pandas dataframe but could be something else. 
        # Query file with nl_coefs
        # Query file with IPC coefs
        # Query file with BFE coefs 
        # What else ? 
        pass
        
    
def load_nl_coefficients(path=None):
    """Method to get classical non-linearity corrections.

    Args:
        path (str, optional): Filename path extracted from metadata. 
                              Defaults to None.
    Returns : 
        ndarray : (N, Npix, Npix) of coefficients per pixel. 
                   N depends on polynomial degree decided to estimate correction.
    """
    if path is None : 
        return 0,0,0,0

    else : 
        pass 
    
def load_downpix(path=None):
    """Method to load identified buggy pixel from caracterization campaing.

    Args:
        path (str, optional): Filename path extracted from metadata. 
                              Defaults to None.
    Returns : 
        ndarray : Boolean N-darray where True marks pixel to mask.
    """
    # From carac, disconnected pixels mask / map
    pass

def load_bfe(path=None):
    """Method to load identified buggy pixel from caracterization campaing.

    Args:
        path (str, optional): Filename path extracted from metadata. 
                              Defaults to None.
    Returns : 
        ndarray : Boolean N-darray where True marks pixel to mask.
    """
    # From carac, disconnected pixels mask / map
    pass

def load_ipc(path=None, nl=False):
    """Method to load identified buggy pixel from caracterization campaing.

    Args:
        path (str, optional): Filename path extracted from metadata. 
                              Defaults to None.
    Returns : 
        ndarray : Boolean N-darray where True marks pixel to mask.
    """
    # From carac, disconnected pixels mask / map
    pass


def load_gain_map(path=None): 
    """Method to get gain map from given path.

    Args:
        path (str, optional): Filename path extracted from metadata. 
                              Defaults to None.
    Returns : 
        ndarray : (Npix, Npix) map of estimated gain. 
    """
    pass

def get_flatfield():
    """Method to get master-flat from given path.

    Args:
        path (str, optional): Filename path extracted from metadata. 
                              Defaults to None.
    Returns : 
        ndarray : (Npix, Npix) flat-field. 
    """

    pass
    
def get_dark():
    """Method to get master dark from given path.

    Args:
        path (str, optional): Filename path extracted from metadata. 
                              Defaults to None.
    Returns : 
        ndarray : (N,Npix, Npix) dark cube. 
    """    
    pass

def get_qe_map():
    """Method to get QE flat from given path.

    Args:
        path (str, optional): Filename path extracted from metadata. 
                              Defaults to None.
    Returns : 
        ndarray : (Npix, Npix) map of pixel response. 
    """   
    pass

def get_previous_exposure(metadata):
    """Method to get previous exposure information to model & correct persistence.

    Args:
        path (str, optional): Filename path extracted from metadata. 
                              Defaults to None.
    Returns : 
        ndarray : The previous estimated flux (or should it be a cube ? )
        float :  Delta-time between current and last exposure. 
        
         
    """  

    return load_data(metadata['path']) 


def load_data(filepath):
    #Use filepath and process those. Should return stuff.
    #Very general function, probably should create a general purpose METADATA class.
    data=0
    return data

def load_hotpix(filepath):
    return 0 
    
## -------------- ##
## RAMP I/O Utils ##
## -------------- ##

def from_roman(filename):
    with fits.open(filename) as hdul: 
        header = hdul[0].header
        data = np.squeeze(hdul[1].data)[:,:,:-128] 
        tframe = header['TFRAME']
        tgroup = header['TGROUP']
        nchannels = header['NOUTPUTS']
        times  = np.linspace(header['TGROUP'], header['XPOSURE'], num=header['NGROUPS'])
            
    return  data , tframe, tgroup, nchannels, times.astype("float32")
    
def from_pyxim(filename):
    with fits.open(filename) as hdul : 
        data = hdul[0].data
        times = hdul[1].data['readout_times'].astype("float32")
        nchannels = hdul[0].header['NCHAN']
            
    tframe = np.diff(times).mean()
    tgroup = np.diff(times).mean()
    return data, tframe, tgroup, nchannels, times.astype("float32")

def from_ip2i(filename, frame_path='/run/seq_1/asic_1/ramp_1', time_arr_path=None):
    f = h5py.File(filename, 'r')

    if time_arr_path is None : 
        time_arr_path = '/'.join(frame_path.split('/')[:2])+'/env_measures/'

    npix_channel = 64
    data = f[frame_path][()][:,:,npix_channel:] 
    tarr = f[time_arr_path]['time'].astype("float32") 

    tframe = np.diff(tarr).mean()
    times = np.arange(1, data.shape[0]+1)*tframe
    nchannels=32
    tgroup = None 

    return data, tframe, tgroup, nchannels, times.astype("float32")

def from_anyfits(filename): 
    with fits.open(filename) as hdul : 
        data = hdul[0].data

    return data