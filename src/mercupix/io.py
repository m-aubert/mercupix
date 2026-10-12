"""Input/output functions for ramps and calibration products."""

import numpy as np 
from astropy.io import fits
import h5py


def fetch_metadata(path=None):
    """Get the metadata associated with a file.

    Parameters
    ----------
    path : str, optional
        Path to the file. Default is None.

    Returns
    -------
    Metadata
        Metadata of the file.

    Notes
    -----
    Not implemented yet: currently returns None.
    """
    pass

class Metadata:
    
    """Calibration products and information needed to process a ramp.

    Parameters
    ----------
    filename : str
        Path to the file the metadata refer to.

    Notes
    -----
    Placeholder class, not implemented yet.
    """
    def __init__(filename):
        pass
        
    def parse_filename(filename):
        """Parse a filename to get the information needed for queries.

        Parameters
        ----------
        filename : str
            Path to the file.

        Notes
        -----
        Not implemented yet.
        """
        pass

    def query_metadata(param):
        """Query the metadata corresponding to a parameter.

        Intended to query the nearest calibration dark and flat, the gain
        map, the previous acquisition and the non-linearity, IPC and BFE
        coefficients. Requires a unified database and data model.

        Parameters
        ----------
        param : object
            Parameter to query.

        Notes
        -----
        Not implemented yet.
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
    """Load the classical non-linearity correction coefficients.

    Parameters
    ----------
    path : str, optional
        Path to the coefficient file, from the metadata. Default is None.

    Returns
    -------
    ndarray or tuple
        Coefficients of shape ``(ncoefs, ny, nx)``, in increasing
        polynomial order. If ``path`` is None, ``(0, 0, 0, 0)``.

    Notes
    -----
    Loading from ``path`` is not implemented yet.
    """
    if path is None : 
        return 0,0,0,0

    else : 
        pass 
    
def load_downpix(path=None):
    """Load the disconnected pixel map from the characterization campaign.

    Parameters
    ----------
    path : str, optional
        Path to the map, from the metadata. Default is None.

    Returns
    -------
    ndarray
        Boolean map where True marks the pixels to mask.

    Notes
    -----
    Not implemented yet: currently returns None.
    """
    # From carac, disconnected pixels mask / map
    pass

def load_bfe(path=None):
    """Load the brighter-fatter coefficients from the characterization campaign.

    Parameters
    ----------
    path : str, optional
        Path to the coefficients, from the metadata. Default is None.

    Returns
    -------
    ndarray
        Brighter-fatter kernel coefficients.

    Notes
    -----
    Not implemented yet: currently returns None.
    """
    # From carac, disconnected pixels mask / map
    pass

def load_ipc(path=None, nl=False):
    """Load the IPC coefficients from the characterization campaign.

    Parameters
    ----------
    path : str, optional
        Path to the coefficients, from the metadata. Default is None.
    nl : bool, optional
        Whether to load the non-linear IPC coefficients. Default is False.

    Returns
    -------
    ndarray
        IPC kernel coefficients.

    Notes
    -----
    Not implemented yet: currently returns None.
    """
    # From carac, disconnected pixels mask / map
    pass


def load_gain_map(path=None): 
    """Load the gain map.

    Parameters
    ----------
    path : str, optional
        Path to the gain map, from the metadata. Default is None.

    Returns
    -------
    ndarray
        Gain map of shape ``(ny, nx)``.

    Notes
    -----
    Not implemented yet: currently returns None.
    """
    pass

def get_flatfield():
    """Load the master flat-field.

    Returns
    -------
    ndarray
        Flat-field of shape ``(ny, nx)``.

    Notes
    -----
    Not implemented yet: currently returns None.
    """

    pass
    
def get_dark():
    """Load the master dark.

    Returns
    -------
    ndarray
        Dark cube of shape ``(ngroups, ny, nx)``.

    Notes
    -----
    Not implemented yet: currently returns None.
    """
    pass

def get_qe_map():
    """Load the quantum efficiency map.

    Returns
    -------
    ndarray
        Pixel response map of shape ``(ny, nx)``.

    Notes
    -----
    Not implemented yet: currently returns None.
    """
    pass

def get_previous_exposure(metadata):
    """Load the previous exposure, used to model and correct persistence.

    Parameters
    ----------
    metadata : dict
        Metadata with the path to the previous exposure under the
        ``'path'`` key.

    Returns
    -------
    object
        Previous exposure, as returned by `load_data`.
    """

    return load_data(metadata['path']) 


def load_data(filepath):
    #Use filepath and process those. Should return stuff.
    #Very general function, probably should create a general purpose METADATA class.
    """Load a processed data product.

    Parameters
    ----------
    filepath : str
        Path to the file.

    Returns
    -------
    object
        Loaded data.

    Notes
    -----
    Placeholder: currently returns 0.
    """
    data=0
    return data

def load_hotpix(filepath):
    """Load the hot and warm pixel map.

    Parameters
    ----------
    filepath : str
        Path to the map.

    Returns
    -------
    ndarray or int
        Bitmask of hot and warm pixels.

    Notes
    -----
    Placeholder: currently returns 0.
    """
    return 0 
    
## -------------- ##
## RAMP I/O Utils ##
## -------------- ##

def from_roman(filename):
    """Read a ramp in Roman flight-like format.

    The last 128 columns of the frames are dropped.

    Parameters
    ----------
    filename : str
        Path to the FITS file.

    Returns
    -------
    data : ndarray
        Ramp datacube of shape ``(ngroups, ny, nx)``.
    tframe : float
        Time of a frame (``TFRAME`` keyword).
    tgroup : float
        Time of a group (``TGROUP`` keyword).
    nchannels : int
        Number of readout channels (``NOUTPUTS`` keyword).
    times : ndarray
        Time of each group, of shape ``(ngroups,)``, as float32.
    """
    with fits.open(filename) as hdul: 
        header = hdul[0].header
        data = np.squeeze(hdul[1].data)[:,:,:-128] 
        tframe = header['TFRAME']
        tgroup = header['TGROUP']
        nchannels = header['NOUTPUTS']
        times  = np.linspace(header['TGROUP'], header['XPOSURE'], num=header['NGROUPS'])
            
    return  data , tframe, tgroup, nchannels, times.astype("float32")
    
def from_pyxim(filename):
    """Read a ramp in pyxim simulation format.

    Parameters
    ----------
    filename : str
        Path to the FITS file.

    Returns
    -------
    data : ndarray
        Ramp datacube of shape ``(ngroups, ny, nx)``.
    tframe : float
        Mean time between groups.
    tgroup : float
        Mean time between groups.
    nchannels : int
        Number of readout channels (``NCHAN`` keyword).
    times : ndarray
        Readout time of each group, of shape ``(ngroups,)``, as float32.
    """
    with fits.open(filename) as hdul : 
        data = hdul[0].data
        times = hdul[1].data['readout_times'].astype("float32")
        nchannels = hdul[0].header['NCHAN']
                   
    tframe = np.diff(times).mean()
    tgroup = np.diff(times).mean()
    return data, tframe, tgroup, nchannels, times.astype("float32")

def from_ip2i(filename, frame_path='/run/seq_1/asic_1/ramp_1/frames', time_arr_path=None):
    """Read a ramp in IP2I test bench format.

    The first 64 columns of the frames are dropped and 32 readout channels
    are assumed.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    frame_path : str, optional
        Path of the frame dataset in the file. Default is
        ``'/run/seq_1/asic_1/ramp_1/frames'``.
    time_arr_path : str, optional
        Path of the group holding the ``'time'`` dataset. If None, it is
        derived from ``frame_path`` as ``'/run/env_measures/'``. Default is
        None.

    Returns
    -------
    data : ndarray
        Ramp datacube of shape ``(ngroups, ny, nx)``.
    tframe : float
        Mean time between frames.
    tgroup : None
        Time of a group, not available in this format.
    nchannels : int
        Number of readout channels (32).
    times : ndarray
        Time of each group, ``(1, 2, ..., ngroups) * tframe``, as float32.
    """
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


# Generic functions for other data.
def from_anyfits(filename): 
    """Read the primary HDU data of any FITS file.

    Parameters
    ----------
    filename : str
        Path to the FITS file.

    Returns
    -------
    ndarray
        Data of the primary HDU.
    """
    with fits.open(filename) as hdul : 
        data = hdul[0].data

    return data
