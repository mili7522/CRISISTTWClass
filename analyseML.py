import TTWMaximumLikelihood as TTWML
from utils import utils
import numpy as np
import pandas as pd
import json
import os
import scipy.stats

### Parameters
model_number = 5  # Model 1: Gravitational; 2: Log; 3: c_w = 0; 4: c_l = 0 (gravitational); 5: c_l = 0 (log)
bootstrap_id = 'Data'
year = 2011
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
        if c_w == 0:
            params['beta'] = 0
            return params
        else:
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

def getEnergyComponents(ttwml, params, TTW, print_ = False):
    '''
    TTW is from actual (or bootstrapped) data
    '''
    expected_H_local = np.sum(TTW * ttwml.local_energy(params)) / ttwml.totalHouseholds
    expected_H_work = np.sum(TTW * ttwml.work_energy(params)) / ttwml.totalHouseholds
    expected_H_total = np.abs(expected_H_local) + np.abs(expected_H_work)

    beta = params['beta']

    if print_:
        print("Percentage H_local:", expected_H_local / expected_H_total * 100)
        print("H_total:", expected_H_total)
        print("Percentage 1/beta:", np.divide(1,beta) / expected_H_total * 100)

    return expected_H_local / expected_H_total * 100, expected_H_total, np.divide(1,beta) / expected_H_total * 100

def getLogLikelihood(ttwml, params, TTW, print_ = False):
    keys = params.keys()
    value = [params[key] for key in keys]
    negLogLikelihood = ttwml.negLogLikelihood(value, keys, TTWArray = TTW)
    
    if print_:
        print("Log Likelihood:", -negLogLikelihood)

    return -negLogLikelihood

def getComparisonWithData(ttwml, savePath, TTW, print_ = False):
    if type(savePath) == str:
        p = loadProbability(savePath).ravel()
    else:  # Can input a TTW array instead
        p = savePath.ravel() / ttwml.totalHouseholds
    q = TTW.ravel() / ttwml.totalHouseholds

    hellingerDistance = utils.hellingerDistance(p,q)
    JSDivergence = utils.jsDivergence(p,q)

    if print_:
        print("Hellinger Distance:", hellingerDistance)
        print("JS Divergence:", JSDivergence)

    return hellingerDistance, JSDivergence

def generateTTWs(ttwml, savePath, repeats = 100):
    filename = os.path.join(savePath, 'TTWFromGeneratedHouseholds{}.npy'.format(savePath.split('/')[-1]))
    if os.path.exists(filename):
        TTWs = np.load(filename)
        return list(TTWs)

    np.random.seed(10)
    P = loadProbability(savePath)
    TTWs = []
    for _ in range(repeats):
        mllArray = ttwml.arrayFromProbability(P, households = ttwml.totalHouseholds)
        TTWs.append(mllArray)
    TTWs_array = np.array(TTWs)
    np.save(filename, TTWs_array)
    return TTWs


def getEntropy(TTWs, print_ = False):
    if not type(TTWs) == list:
        TTWs = [TTWs]

    entropyHH = np.zeros(len(TTWs))
    entropyTTW = np.zeros(len(TTWs))

    for i, TTW in enumerate(TTWs):
        entropyHH[i] = scipy.stats.entropy(TTW.ravel())
        entropyTTW[i] = scipy.stats.entropy(TTW.sum(axis = 1))
    
    if print_:
        print("Mean of Entropy HH:", entropyHH.mean())
        print("Std of Entropy HH:", entropyHH.std())
        print("Mean of Entropy TTW:", entropyTTW.mean())
        print("Std of Entropy TTW:", entropyTTW.std())

    return entropyHH.mean(), entropyHH.std(), entropyTTW.mean(), entropyTTW.std()
      

def getAverageTimeToWork(TTWs, ttwml, distances, print_ = False):
    if not type(TTWs) == list:
        TTWs = [TTWs]

    averageTimeToWork = np.zeros(len(TTWs))

    for i, TTW in enumerate(TTWs):
        averageTimeToWork[i] = np.sum(TTW * distances) / ttwml.totalHouseholds
    
    if print_:
        print("Mean of Average Time to Work:", averageTimeToWork.mean())
        print("Std of Average Time to Work:", averageTimeToWork.std())

    return averageTimeToWork.mean(), averageTimeToWork.std()

def printParams(params):
    print("c:", params['c_w'])
    print("d* (alpha_w):", params['alpha_w'])
    print("beta:", params['beta'])

def analyseShuffleNull(repeats = 10):
    from utils.generate_bootstrap import getCurrentHouseholds, getTTW

    TTW, distances, _ = utils.loadData(year, model_number, bootstrap_id)
    ttwml = TTWML.TTWML(distances, TTW, logForm = False)
    householdsHome, householdsWork = getCurrentHouseholds(TTW)
    np.random.seed(10)

    comparisons_with_data = []
    entropy = []
    time_to_work = []

    for _ in range(repeats):
        np.random.shuffle(householdsWork)  # Modifies in-place
        shuffledHouseholds = np.vstack([householdsHome, householdsWork]).T
        newTTW = getTTW(shuffledHouseholds, len(TTW))
    
        comparisons_with_data.append(getComparisonWithData(ttwml, newTTW, TTW))
        entropy.append(getEntropy(newTTW))
        time_to_work.append(getAverageTimeToWork(newTTW, ttwml, distances))
    return np.array(comparisons_with_data).mean(axis = 0), np.array(entropy).mean(axis = 0), np.array(time_to_work).mean(axis = 0)

def analyseData():
    TTW, distances, rent = utils.loadData(year, model_number, bootstrap_id)
    ttwml = TTWML.TTWML(distances, TTW, logForm = True if model_number in [2, 5] else False)
    getEntropy(TTW, print_ = True)
    getAverageTimeToWork(TTW, ttwml, distances, print_ = True)

###
if __name__ == "__main__":
    TTW, distances, rent = utils.loadData(year, model_number, bootstrap_id)
    ttwml = TTWML.TTWML(distances, TTW, logForm = True if model_number in [2, 5] else False)

    params = loadParameters(savePath)

    TTWs = generateTTWs(ttwml, savePath, repeats = 100)

    printParams(params)
    getLogLikelihood(ttwml, params, TTW, print_ = True)
    getComparisonWithData(ttwml, savePath, TTW, print_ = True)
    getEntropy(TTWs, print_ = True)
    getAverageTimeToWork(TTWs, ttwml, distances, print_ = True)
    getEnergyComponents(ttwml, params, TTW, print_ = True)