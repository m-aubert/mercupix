




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


