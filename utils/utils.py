import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plotTTW(TTW, cmap = 'Greens', saveName = None):
    vmax = np.max(TTW)
    vmin = np.min(TTW)
    if np.isinf(vmin): vmin = 0
    
    ax = plt.imshow(TTW, cmap = cmap, vmax = vmax, vmin = vmin)
    plt.xlabel('Work Suburb'); plt.ylabel('Home Suburb')
    
    fig = ax.get_figure()
    cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    sm = plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin, vmax))
    sm._A = []
    fig.colorbar(sm, cax=cax)

    if saveName is not None:
        plt.savefig(saveName, dpi = 250, format = 'png', bbox_inches = 'tight')
        plt.close()