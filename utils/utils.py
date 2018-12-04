import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats


def loadData(year, model_number, bootstrap_id):
    if bootstrap_id is None or bootstrap_id == 'Data':
        TTW = pd.read_table('Data/{}_TTW.csv'.format(year), sep = ',', index_col= 0)
    elif bootstrap_id == 'Null' or bootstrap_id == 'Shuffled':
        TTW = pd.read_table('Data/ShuffledTTW{}.csv'.format(year), sep = ',', index_col= 0)
    else:
        TTW = pd.read_table('Data/BootstrapTTW{}/BootstrapTTW{}.csv'.format(year, bootstrap_id), sep = ',', index_col= 0)

    rent = pd.read_csv('Data/{}_AveragedMedianValuesPerM-SA2.csv'.format(year), index_col = 0, usecols = [0,2], squeeze = True)

    # Transpose TTW array
    TTW = np.transpose(TTW)
    TTW = TTW.values

    distances = pd.read_csv("Data/{}_SA2-DriveTimes-OSRM.csv".format(year), header = None)
    distances = ((distances + distances.transpose()) / 2 / 60).values + 5

    return TTW, distances, rent



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
    else:
        plt.show()

def hellingerDistance(p, q):
    distance = (np.sqrt(p) - np.sqrt(q)) ** 2
    distance = np.sqrt(np.sum(distance)) / np.sqrt(2)
    return distance

def jsDivergence(p, q):
    m = (p + q) / 2
    js = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2
    return js

# sns.pairplot(r, markers="x", plot_kws = {'alpha': 0.1})
# plt.savefig(savePath + 'Generated-Recovered4Params{}.png'.format(trial), dpi = 250, format = 'png', bbox_inches = 'tight')
# plt.close()



# plotTTW(np.round(P * np.sum(TTW)), saveName = savePath + 'RecoveredTTW{}.png'.format(trial_name))
# plotTTW(np.log(np.round(P * np.sum(TTW))), saveName = savePath + 'RecoveredTTWLog{}.png'.format(trial_name))



### Generate Sydney data using maximum likelihood parameters

# entropyHH = []
# entropyTTW = []
# averageTimeToWork = []
# for _ in range(1000):
#     mllArray = a.arrayFromProbability(P, households = totalHouseholds, minimiseRandomness = False)
#     timeToWork = np.sum(mllArray * distances) / totalHouseholds
#     averageTimeToWork.append(timeToWork)
#     entropyTTW.append(scipy.stats.entropy(mllArray.ravel()))
#     entropyHH.append(scipy.stats.entropy(mllArray.sum(axis = 1)))

# df = pd.DataFrame([entropyHH, entropyTTW, averageTimeToWork])
# df = df.transpose()
# df.columns = ['EntropyHH', 'EntropyTTW', 'AverageTimeToWork']
# df.to_csv(savePath + 'MeasuresFromGeneratedHouseholds{}.csv'.format(trial), index = None)

