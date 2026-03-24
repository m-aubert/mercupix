import numpy as np
import warnings

class LSQFit(): 
    """ Uses least-square methods to evaluate flux."""
    def __init__(self, data, times=None, 
                 degree=5, polytype='leg'):
        """ """
        self._data = data
        if times is None:
            times = np.arange(data.shape[0])
        
        self.set_times(times)
        self.set_polynome(polytype=polytype, degree=degree)

        
    @classmethod
    def from_rawramp(cls, rawramp, **kwargs):
        """ """
        datacube, varcube, mask = rampdata.preprocess_ramp(**kwargs)
        return cls(data=datacube, times=rampdata.t_read)
        
    # ================ #
    #   method         #
    # ================ #    
    # ------- #
    # SETTER  # 
    # ------- #
    def set_polynome(self, degree, polytype="legendre", load=False):
        """ """
        
        self._polynome = {"type": polytype, 
                          "degree": degree, 
                          "array": None,
                          "coefs": None,
                         }            
        self._hcached_fitted_times = None
        
    def set_times(self, times):
        """ """
        self._times = np.asarray(times, dtype="float")
        self._hcached_fitted_times = None

    # ------- #
    # GETTER  # 
    # ------- #
    def get_times(self, as_fitted=False, polytype=None):
        """ """
        if not as_fitted:
            return self.times

        if polytype is None:
            polytype = self.polynome_type

        return self._times_to_polynomex(times=self.times, polytype=polytype)
    
    def fit(self, data=None, inplace=False):
        """ """
        if data is None:
            data = self.data
            
        coefs = self._fit_polynome_on_data(polynome=self.polynome_basis,
                                            data=data)
        if inplace:
            self._polynome["coefs"] = coefs

        return coefs


    def scan_fit(self, data=None, progress_bar=False,
                     frame_range=[4, None],
                     x_range=None, y_range=None):
        """ """
        if progress_bar:
            from tqdm import tqdm
            
        if data is None:
            data = self.data

        if x_range is not None:
            print("x_range")
            data = data[:, :, x_range[0]:x_range[1]]

        if y_range is not None:
            print("y_range")
            data = data[:, y_range[0]:y_range[1], :]
            
        # parse input
        firstframe, lastframe = frame_range
        if firstframe is None:
            firstframe = self.polynome_degree+1
            
        elif firstframe < self.polynome_degree:
            warnings.warn("firstframe is lower than the number of degree. Issues are expected.")
        
        if lastframe is None:
            lastframe = data.shape[0]
        elif lastframe < 0:
            lastframe = data.shape[0] + lastframe +1
            
        # ---- #
        xframes = np.arange(*frame_range)
        coefs = []
        times = []
        for lastframe in xframes if not progress_bar else tqdm(xframes):
            this_data = data[:lastframe]
            this_times = self.times[:lastframe]
            
            # this polynome case:
            polynome = self._polynome_of_times(this_times,
                                                 degree=self.polynome_degree,
                                                 polytype=self.polynome_type)
    
            coefs.append( self._fit_polynome_on_data(polynome, data=this_data) )
            times.append( this_times.max() )
            
        return xframes, np.asarray(times), np.asarray(coefs)

    def get_model(self, pixels=None, coefs=None):
        """ """
        if coefs is None:
            coefs = self.polynome_coefs

        if pixels is None:
            model = np.einsum('ij,jkl->ikl', self.polynome_basis.T, coefs)
        else:
            xpixel, ypixel = pixels
            model = self.polynome_basis.T @ coefs[:, xpixel, ypixel]
            
        return model
        
    def get_residuals(self, pixels=None, coefs=None, model=None):
        """ """
        if model is None:
            model = self.get_model(pixels=pixels, coefs=coefs)
            
        if pixels is None:
            residuals = self.data - model
        else:
            xpixel, ypixel = pixels
            residuals = self.data[:, xpixel, ypixel] - model
            
        return residuals

    def parse_coefs(self, coefs=None, times=None, axis=None, in_adu=False):
        """ """
        if coefs is None:
            if self.polynome_coefs is None:
                raise ValueError("no coefs given, none stored.")
                
            coefs = self.polynome_coefs
            
        if axis is None:
            # fetch which axis corresponds that of the degree:
            index = np.argwhere(np.asarray(coefs.shape) == self.polynome_degree+1).squeeze()
            if index.ndim != 0: # mean int
                raise ValueError("No dimension of coefs matches that of the polynome degree. "+\
                                 "Please set the degree axis")
            else:
                axis = int(index)
    
        # split the coefficients per interest
        offset, slope, nlinearities = np.split(coefs, [1, 2], axis=axis)
        offset = offset.squeeze()
        slope = slope.squeeze()
    
        if in_adu:
            if self.polynome_type in ["leg", "legendre", "herm", "hermite"]:
                # 
                offset -= slope

                
                if times is None:
                    times = self.times
                    
                if slope.ndim > 2 and times.ndim != slope.ndim:
                    missing_dims = slope.ndim - times.ndim
                    times = times.reshape([times.shape[0] ]+ [1, ]*missing_dims)
                        
                elif slope.ndim == 2:
                    times = float(self.times.max())
                else:
                    # let's assume user know what they are doing.
                    pass
                    
                # broadcast.
                slope = slope * 2/times
    
        return offset, slope, nlinearities
    
    # ------------- #
    #   internal    #
    # ------------- #
    @staticmethod
    def _times_to_polynomex(times, polytype):
        """ """
        # careful, -1 must corresponds to t=0 so tmin=0, *not* np.min(times)
        if polytype in ["leg", "legendre", "herm", "hermite"]:
            tmin, tmax = 0, np.max(times)
            times = ((times-tmin) / (tmax-tmin))*2 -1
        else:
            raise NotImplementedError("Unknown {polytype=} (legendre or hermite implemented).")

        return times

    @staticmethod
    def _fit_polynome_on_data(polynome, data):
        """ """
        coefs = np.linalg.lstsq(polynome.T, 
                                data.reshape(data.shape[0], -1).astype(np.float32), 
                                rcond=None)[0]
        # reshaped back as data.
        return coefs.reshape(coefs.shape[0], *data.shape[1:])

    @staticmethod
    def _get_polynome(degree, polytype, xnormed):
        """ """
        if polytype in ["leg", "legendre"]: 
            from scipy.special import legendre
            pol = np.stack([legendre(i)(xnormed) for i in range(degree+1)])
                
        elif polytype in ["herm", "hermite"]: 
            from scipy.special import hermitenorm
            pol = np.stack([hermitenorm(i)(xnormed) for i in range(degree+1)])
                
        else: 
            raise NotImplementedError("Unknown {polytype=} (legendre or hermite implemented)")

        return pol
    
    @classmethod
    def _polynome_of_times(cls, times, degree, polytype):
        """ """
        times = cls._times_to_polynomex(times, polytype=polytype)
        return cls._get_polynome(degree=degree,
                                 polytype=polytype,
                                 xnormed=times)
        
    # ================ #
    #   properties     #
    # ================ #    
    @property
    def data(self):
        """ datacube to be fitted """
        return self._data

    @property
    def times(self):
        """ """
        return self._times

    @property
    def _cached_fitted_times(self):
        """ """
        if not hasattr(self, "_hcached_fitted_times") or self._hcached_fitted_times is None:
            self._hcached_fitted_times = self.get_times()
            
        return self._hcached_fitted_times

    #  polynome
    @property
    def polynome_type(self):
        """ kind of polynome """
        return self._polynome.get("type")

    @property
    def polynome_degree(self):
        """ kind of polynome """
        return self._polynome.get("degree")    

    @property
    def polynome_basis(self):
        """ kind of polynome """
        if self._polynome.get("array", None) is None:
            self._polynome["array"] = self._get_polynome(degree=self.polynome_degree,
                                                         polytype=self.polynome_type,
                                                         xnormed=self._cached_fitted_times)
        return self._polynome["array"]

    @property
    def polynome_coefs(self):
        """ """
        return self._polynome.get("coefs", None)
