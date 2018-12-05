import TTWMaximumLikelihood as TTWML
from utils import utils
from analyseML import loadParameters, getLogLikelihood
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import json


### Parameters
model_number = 2  # Model 1: Gravitational; 2: Log; 3: c_w = 0; 4: c_l = 0 (gravitational); 5: c_l = 0 (log)
bootstrap_id = 'Data'
year = 2016
if bootstrap_id is not 'Data':
    bootstrap_id = '{:02d}'.format(bootstrap_id)

trial_name = 'Sydney{}_{}-{}'.format(year, model_number, bootstrap_id)
savePath = 'Results/{}'.format(trial_name)


def fisher(p1, p2, difference):
    '''Calculate Fisher Information'''
    average = (p1 + p2) / 2  # To reduce the chance of dividing by 0
    fisher = ((p1 - p2) / difference) ** 2 / average
    return np.nansum(fisher)

def probabilityDistOfDistanceHist(p, dist, binwidth = 5):
    '''Get probability distribution of distance histogram'''
    p = p.ravel()
    dist_bin_index = (dist // binwidth).ravel().astype(int)
    assert len(p) == len(dist_bin_index)
    
    dist_bins = np.zeros(np.max(dist_bin_index) + 1)

    for i in range(len(dist_bin_index)):
        dist_bins[dist_bin_index[i]] += p[i]
    
    return dist_bins

def plotDistanceHistograms(params_original, ttwml, distances, param_to_change, param_ratios):
    p_by_distance_bins = pd.DataFrame()
    for ratio in param_ratios:
        new_params = params_original.copy()
        new_params[param_to_change] = params_original[param_to_change] * ratio
        p = ttwml.probabilityFromParameters(new_params, printParams = False)

        p_by_distance_bins["c ratio = {}".format(ratio)] = probabilityDistOfDistanceHist(p, distances, binwidth = 5)
    p_by_distance_bins.index = p_by_distance_bins.index * 5
    p_by_distance_bins.iloc[1:20].plot(kind = 'bar', rot = 0, width = 0.75)
    plt.xlabel('Time to Work (min)')
    plt.ylabel('Probability')

    plt.show()

def plotFisher(params_original, ttwml, distances):
    paramSteps = np.linspace(params_original['c_w']*0.1, params_original['c_w']*5, 300)
    step = paramSteps[1] - paramSteps[0]
    for beta_ratio in [0.5, 1, 1.5]:
        oldPHist = None
        fisherHistOutput = []
        for x in paramSteps:
            new_params = params_original.copy()
            new_params['c_w'] = x
            new_params['beta'] = params_original['beta'] * beta_ratio
            p = ttwml.probabilityFromParameters(new_params, printParams=False)
            
            pHist = probabilityDistOfDistanceHist(p, distances)
            if oldPHist is not None:
                fisherHistOutput.append(fisher(oldPHist, pHist, step))
            oldPHist = pHist
        plt.plot(paramSteps[1:], fisherHistOutput, label = r'$\beta$ Ratio = {}'.format(beta_ratio))
        # Get parameter at peak of fisher curve
        if beta_ratio == 1:
            maxidx = np.argmax(fisherHistOutput)
            c_at_max = paramSteps[maxidx+1]
    plt.axvline(params['c_w'], color = 'k', linestyle = '--', label = 'Maximum Likelihood c')
    plt.legend()
    plt.xlabel('c')
    plt.ylabel('Fisher Information')
    plt.show()

    return c_at_max

if __name__ == "__main__":
    TTW, distances, rent = utils.loadData(year, model_number, bootstrap_id)
    ttwml = TTWML.TTWML(distances, TTW, logForm = True if model_number in [2, 5] else False)

    params = loadParameters(savePath)

    plotDistanceHistograms(params, ttwml, distances, 'c_w', [0.5, 1, 1.5])
    
    c_at_max = plotFisher(params, ttwml, distances)
    new_params = params.copy()
    new_params['c_w'] = c_at_max
    max_loglikelihood = getLogLikelihood(ttwml, params, TTW, print_ = True)
    loglikelihood_at_peak = getLogLikelihood(ttwml, new_params, TTW, print_ = True)
    print("Log likelihood ratio:", max_loglikelihood / loglikelihood_at_peak)
