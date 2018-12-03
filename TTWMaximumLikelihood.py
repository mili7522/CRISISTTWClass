import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json
import os
import time

class TTWML():

    def __init__(self, distances, TTWArray, logForm = False):
        self.distances = distances
        self.arraySize = distances.shape
        self.logForm = logForm
        
        assert distances.shape == TTWArray.shape
        
        self.totalHouseholds = np.sum(TTWArray)
        self.workDistribution = np.sum(TTWArray, axis = 0, keepdims = True) / self.totalHouseholds

        # Default parameters. These are overridden by the inputs for the maximum likelihood calculation  
        self.parameters = {"c_w": 1,
                           "c_l": 1,
                           "alpha_w": 1,
                           "alpha_l": 1,
                           "beta": 5}

    def getParameters(self, parameters = None):
        if parameters is None:
            parameters = self.parameters
        for key, value in self.parameters.items():
            parameters.setdefault(key, value)
        
        return parameters

    def local_energy(self, params):
        distances = self.distances
        workDistribution = self.workDistribution

        e = workDistribution * np.exp(-distances ** 2 / (2 * params['alpha_l'] ** 2))
        e = np.sum(e, axis = -1, keepdims = True)
        # Replace Gaussian form with individual local costs for each suburb if this is available
        e = params.get('local_energy', e)
        e = params['c_l'] * e  # 'c_l' may be kept at 1
        
        e = e.reshape((-1,1))  # Keep this to ensure that 'local_energy' is a column vector
        
        return e

    def work_energy(self, params):
        distances = self.distances
        logForm = self.logForm

        if logForm:
            e = params['c_w'] * np.log(distances / params['alpha_w'])
        else:
            e = -params['c_w'] / (distances ** params['alpha_w'])
            #e = -params['c_w'] * np.exp(-distances ** 2 / (2 * params['alpha_w'] ** 2))  # Gaussian

        return e

    def energyOfSingleHousehold(self, params = None):

        params = self.getParameters(params)

        return self.local_energy(params) + self.work_energy(params)



    def negLogLikelihood(self, params, param_keys, fixed_params = {}, TTWArray = None):

        params = dict(zip(param_keys, params))  # Combine with the keys to turn the list into a dictionary
        for key, value in fixed_params.items():
            params.setdefault(key, value)
        params = self.getParameters(params)
        
        H = self.energyOfSingleHousehold(params = params)

        Z = np.sum(np.exp(-params['beta'] * H), axis = 0, keepdims = True)  # Sum for each workplace

        # TTWArray gives the observed data (Need to be supplied)
        lnP = np.multiply(TTWArray, -params['beta'] * H - np.log(Z))
        negLL = np.sum(-lnP)
        return negLL


    def maximiseLikelihood(self, initParams, bounds, trials = 100, fixed_params = {}, TTWArray = None, printAllResults = False, saveAllPath = None):
        # InitParams and bounds are input as dictionaries
        
        keys = initParams.keys()
            
        finalParams = []
        loglikelihood = [] 
        
        for i in range(trials):
            # Take init as drawn from a normal distribution with mean and variance set as the initParams value
            #if trials == 1:
            #    init = [initParams[key] for key in keys]
            #else:
            init = [np.maximum(np.random.normal(initParams[key], initParams[key]), bounds[key][0]) for key in keys]  # Did not include a check against the bound maximum because None values are sometimes used. The minimizer should automatically correct this anyway
            results = minimize(self.negLogLikelihood, init, args = (keys, fixed_params, TTWArray),
                                bounds = [bounds[x] for x in keys])
            
            if results.success:
                finalParams.append(results.x)
                loglikelihood.append(results.fun)
            if printAllResults:
                print('Trial', i)
                print('Likelihood', results.fun)
                print(dict(zip(keys, results.x)))
            if saveAllPath is not None:
                data = [results.fun, json.dumps(dict(zip(keys, results.x))), results.success]
                while True:
                    try:
                        if not os.path.exists(saveAllPath):
                            headers = ['NegLogLikelihood', 'Params', 'Success']
                            with open(saveAllPath, 'w') as f:
                                f.write(','.join(headers) + '\n')
                        with open(saveAllPath, 'a') as f:
                            f.write(','.join(map(str, data)) + '\n')
                        break
                    except:
                        print("Sleeping")
                        time.sleep(5)

        if len(loglikelihood):
            idx = np.nanargmin(loglikelihood)
            return dict(zip(keys, finalParams[idx])), loglikelihood[idx]
        else:
            return None, None
        

    def probabilityFromParameters(self, params, printParams = True):
        params = self.getParameters(params)
        if printParams:
            print(params)
        
        workDistribution = np.atleast_2d(self.workDistribution)
        
        H = self.energyOfSingleHousehold(params = params)
        
        exps = np.exp(-params['beta'] * H)
        P = exps / np.sum(exps, axis = 0, keepdims = True)
        P = P * workDistribution  # Converting P(i|j) to P(i,j)
        assert not np.any(np.isnan(P))
        
        self.mllProbability = P
        
        return P

    def arrayFromProbability(self, P = None, households = None):
        if P is None:
            P = self.mllProbability
        
        if households is None:
            households = self.totalHouseholds
        
        suburbs = self.arraySize

        householdTTWChoices = np.random.choice(range(suburbs[0] * suburbs[1]), size = households, p = P.ravel(order = 'C'))
        householdPerTTW_output = np.bincount(householdTTWChoices, minlength = suburbs[0] * suburbs[1])
        TTWArray_output = householdPerTTW_output.reshape((suburbs[0], suburbs[1]))  # Generate a TTWArray_output
        
        self.mllArray = TTWArray_output
        self.mllHHDist = np.sum(TTWArray_output, axis = 1)
        return self.mllArray
        
