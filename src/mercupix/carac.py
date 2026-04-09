import numpy as np 


def get_ptc(data):
    return data.mean(axis=(1,2)) , data.std(axis=(1,2), ddof=1) 
    
    
class ClassicalNL : 
    """
    Purpose : caracterize classical NL.
    Needs to provide a best-fitting map / params for the detector. 
    Ideally -> fast fitting procedure.
    Goal : optimize the choice of number of degrees. 
    Best-chi2 selection or AIC / pixel ? 
    Think on that. Need to test on Leader datapoints.
    Canipe 2017, Hilbert 2023 like.
    """
    
    def __init__(self, ptc, degree=3):
        self.degree=degree
        self.x = ptc[0]
        self.y = ptc[1]
    
    @staticmethod
    def poly_model(x, params, degree=3): 
        model = 0 
        for i in range(degree):
            model += params[i]*(params[1]*x)**i 
        return model 
    
    def get_model(self, params):
        return self.poly_model(self.x, params, self.degree)
    
    def measure_nl(self): 
        # Here should be fitting procedure.
        pass 
        
        
    def fit_params(): 
        from scipy.optimize import curve_fit 
        
        pass
        
    