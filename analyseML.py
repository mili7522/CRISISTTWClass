import TTWMaximumLikelihood as TTWML
from utils import utils
import numpy as np
import pandas as pd
import scipy.stats
import json
import os
import sys

### Parameters
model_number = 2  # Model 1: Gravitational; 2: Log; 3: c_w = 0; 4: c_l = 0 (gravitational); 5: c_l = 0 (log)
bootstrap_id = 'Data'
year = 2016
if bootstrap_id is not 'Data':
    bootstrap_id = '{:02d}'.format(bootstrap_id)

trial_name = 'Sydney{}_{}-{}'.format(year, model_number, bootstrap_id)
savePath = 'Results/{}'.format(trial_name)


###
def loadProbability(savePath):
    P = pd.read_csv(os.path.join(savePath, 'ProbabilityMatrix{}.csv'.format(savePath.split('/')[-1])), header = None).values
    return P

def rescaleParameters(params):
    c_w = params['c_w']
    c_l = params['c_l']
    if c_l == 0:
        factor = c_w
    else:
        factor = c_l
    params['c_w'] /= factor
    params['c_l'] /= factor
    params['beta'] *= factor
    return params

def loadParameters(savePath):
    with open(os.path.join(savePath, 'Parameters{}.txt'.format(savePath.split('/')[-1])), 'r') as outfile:
        params = json.load(outfile)
    params['local_energy'] = np.array(params['local_energy'])
    params = rescaleParameters(params)
    return params

def getEnergyComponents(ttwml, params, TTW):
    '''
    TTW is from actual (or bootstrapped) data
    '''
    expected_H_local = np.sum(TTW * ttwml.local_energy(params)) / ttwml.totalHouseholds
    expected_H_work = np.sum(TTW * ttwml.work_energy(params)) / ttwml.totalHouseholds
    expected_H_total = np.abs(expected_H_local) + np.abs(expected_H_work)

    beta = params['beta']

    return expected_H_local / expected_H_total * 100, expected_H_total, np.divide(1,beta) / expected_H_total * 100

### Comparison with null model

# a.randomiseEdges()
# TTWRand = a.TTWArray

# q = TTWRand.ravel() / totalHouseholds
# print('---')
# print('Hellinger Distance - With Null:', a.hellingerDistance(q, p))
# m = (p + q) / 2
# js = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2
# print('Jensen-Shannon Divergence - With Null:', js)



###
if __name__ == "__main__":
    TTW, distances, rent = utils.loadData(year, model_number, bootstrap_id)
    ttwml = TTWML.TTWML(distances, TTW, logForm = True if model_number in [2, 5] else False)

    params = loadParameters(savePath)
    
    p = loadProbability(savePath).ravel()
    q = TTW.ravel() / ttwml.totalHouseholds

    hellingerDistance = utils.hellingerDistance(p,q)
    JSDivergence = utils.jsDivergence(p,q)