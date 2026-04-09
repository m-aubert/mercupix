import numpy as np 

# Utils func
def get_ptc(data):
    return data.mean(axis=(1,2)) , data.std(axis=(1,2), ddof=1) 
    
# Classical NL carac.
class ClassicalNL : 
    
    def __init__(self, ramp, times, degree=3):
        self.degree= degree
        self.ramp= ramp 
        self.times= times 
        
    @staticmethod
    def poly_model(x, *params, degree=3): 
        model = 0 
        for i in range(degree+1):
            if i != 1 : 
                model += params[i]*(params[1]*x)**i 
            else : 
                model += (params[1]*x)**i 
                
        return model 
    
    def get_model(self, x, *params):
        return self.poly_model(x, *params, degree=self.degree)
        
    def get_nlcorr(self, on_mean=True, init_val=None, corr_degree=4): 
        from scipy.optimize import curve_fit 

        if init_val is None : 
            init_val = [1]*(self.degree+1)
            init_val[1] = self.ramp[0].mean()/2 

        if on_mean : 
            data_vec = self.ramp.mean(axis=(1,2))
            err = self.ramp.std(axis=(1,2), ddof=1)
        else : 
            raise NotImplementedError('Fit per pixel not implemented yet')

        # Fit b+a*x + c*(a*x)**2 + .... 
        # Could probably minimize on degrees but unsure.
        # Legendre could be tested as well but unsure as well.
        p = curve_fit(self.get_model, self.times, data_vec,p0=init_val, maxfev = 10000)[0]

        linearized_data = np.poly1d(p[:2][::-1])(self.times)/data_vec  - 1      
        ps = np.polyfit(data_vec, linearized_data, corr_degree) 
        nl_coefs = ps[::-1] 
        
        return nl_coefs