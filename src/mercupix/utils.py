import numpy as np
import matplotlib.pyplot as plt

def plot_frame(data, fig=None, ax=None, clim=[1,99], **kwargs):
    if fig is None and ax is None : 
        fig, ax = plt.subplots()
    if ax is None and fig is not None : 
        ax = fig.add_subplot()
    
    vmin, vmax = np.nanpercentile(data.ravel(), clim)
            
    dp = ax.imshow(data, vmin = vmin, vmax=vmax, **kwargs)
    plt.colorbar(dp, ax=ax, shrink=0.4)
    return fig, ax

def plot_ramp(data, x=None, fig=None, ax=None, **kwargs):
    if fig is None and ax is None : 
        fig, ax = plt.subplots()
    if ax is None and fig is not None : 
        ax = fig.add_subplot()
    
    if x is None :             
        ax.plot(data, **kwargs)
    else : 
        ax.plot(x,data, **kwargs)

    return (fig,ax)