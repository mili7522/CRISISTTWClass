import TTWMaximumLikelihood as TTWML
from utils import utils
from analyseML import loadProbability, rescaleParameters, loadParameters, generateTTWs
from analyseML import getEnergyComponents, getLogLikelihood, getComparisonWithData, getEntropy, getAverageTimeToWork
import analyseML
import numpy as np
import pandas as pd


model_number = 5  # Model 1: Gravitational; 2: Log; 3: c_w = 0; 4: c_l = 0 (gravitational); 5: c_l = 0 (log)
year = 2016
bootstrap_repeats = 50

def getParams(params):
    return params['c_w'], params['alpha_w'], params['beta']

def generateAllTTWs():
    for year in [2011, 2016]:
        for model_number in range(1, 6):
            for i in range(bootstrap_repeats):
                bootstrap_id = "{:02d}".format(i)
                trial_name = 'Sydney{}_{}-{}'.format(year, model_number, bootstrap_id)
                savePath = 'Results/{}'.format(trial_name)
                print("Generating for:", model_number, bootstrap_id, year)
                TTW, distances, _ = utils.loadData(year, model_number, bootstrap_id)
                ttwml = TTWML.TTWML(distances, TTW, logForm = True if model_number in [2, 5] else False)

                generateTTWs(ttwml, savePath, repeats = 100)


distance_from_data = []
energy_components = []
logLikelihood = []
entropy = []
time_to_work = []
params_list = []

for i in range(bootstrap_repeats):
    bootstrap_id = "{:02d}".format(i)
    trial_name = 'Sydney{}_{}-{}'.format(year, model_number, bootstrap_id)
    savePath = 'Results/{}'.format(trial_name)

    TTW, distances, rent = utils.loadData(year, model_number, bootstrap_id)
    ttwml = TTWML.TTWML(distances, TTW, logForm = True if model_number in [2, 5] else False)

    params = loadParameters(savePath)
    TTWs = generateTTWs(ttwml, savePath, repeats = 100)

    distance_from_data.append(getComparisonWithData(ttwml, savePath, TTW))
    energy_components.append(getEnergyComponents(ttwml, params, TTW))
    logLikelihood.append(getLogLikelihood(ttwml, params, TTW))
    entropy.append(getEntropy(TTWs))
    time_to_work.append(getAverageTimeToWork(TTWs, ttwml, distances))
    params_list.append(getParams(params))

np.set_printoptions(precision = 10)
# np.set_printoptions(formatter = {'float': '{:0.4f}'.format})
print('Params:', np.array(params_list).mean(axis = 0), "±", np.array(params_list).std(axis = 0))
print('Log Likelihood:', np.array(logLikelihood).mean(axis = 0, keepdims = True), "±", np.array(logLikelihood).std(axis = 0, keepdims = True))
print('Distance from data:', np.array(distance_from_data).mean(axis = 0), "±", np.array(distance_from_data).std(axis = 0))
print('Entropy:', np.array(entropy).mean(axis = 0), "±", np.array(entropy).std(axis = 0))
print('Time to Work:', np.array(time_to_work).mean(axis = 0), "±", np.array(time_to_work).std(axis = 0))
print('Energy components:', np.array(energy_components).mean(axis = 0), "±", np.array(energy_components).std(axis = 0))