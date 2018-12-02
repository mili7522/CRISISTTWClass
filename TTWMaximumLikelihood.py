import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
        elif len(parameters) < len(self.parameters):
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

        return e

    def work_energy(self, params):
        distances = self.distances
        logForm = self.logForm

        if logForm:
            e = params['c_w'] * np.log(distances / params['alpha_w'])
        else:
            e = -params['c_w'] / (distances ** params['alpha_w'])
            #e = -params['c_w'] * np.exp(-distances ** 2 / (2 * params['alpha_w'] ** 2))

        return e

    def energyOfSingleHousehold(self, params = None):

        params = self.getParameters(params)

        return self.local_energy(params) + self.work_energy(params)



    def negLogLikelihood(self, params, param_keys, fixed_params = {}, TTWArray = None):

        params = dict(zip(param_keys, params))  # Combine with the keys to turn the list into a dictionary
        for key, value in fixed_params.items():
            params.setdefault(key, value)
        params = self.getParameters(params)

        
        # local energy params
        local_energy_keys = []
        local_energy_values = []
        for key, value in params.items():
            if key.startswith('local_energy'):
                try:
                    local_energy_keys.append(int(key[-4:]))
                    local_energy_values.append(value)
                except ValueError:  #  'local_energy' comes from fixed_params and does not need to be recombined
                    pass
        local_energy = list(zip(local_energy_keys, local_energy_values))
        if len(local_energy) > 0:
            local_energy_sorted = sorted(local_energy)  # Sort based on 'local_energy_keys'
            local_energy = np.array(local_energy_sorted)[:,1]  # Keep just 'local_energy_values'
            params.update({'local_energy': local_energy})
        
        H = self.energyOfSingleHousehold(params = params)

        Z = np.sum(np.exp(-params['beta'] * H), axis = 0, keepdims = True)  # Sum for each workplace

        # TTWArray gives the observed data
        lnP = np.multiply(TTWArray, -params['beta'] * H - np.log(Z))
        negLL = np.sum(-lnP)
        return negLL


    def maximiseLikelihood(self, initParams, bounds, trials = 100, useDE = False, fixed_params = {}, TTWArray = None):
        # InitParams and bounds are input as dictionaries
        
        keys = initParams.keys()
        
        if 'local_energy' in keys:
            initParams = initParams.copy()
            bounds = bounds.copy()
            local_energy = initParams['local_energy']
            local_energy_dict = dict(zip(('local_energy_{:04d}'.format(i) for i in range(len(local_energy))), local_energy))
            local_energy_bounds_dict = dict(zip(('local_energy_{:04d}'.format(i) for i in range(len(local_energy))), [bounds['local_energy']] * len(local_energy)))
            initParams.update(local_energy_dict)
            initParams.pop('local_energy')
            bounds.update(local_energy_bounds_dict)
            bounds.pop('local_energy')
            keys = initParams.keys()
            
        finalParams = []
        loglikelihood = [] 
        
        for i in range(trials): 
            # Take init as drawn from a normal distribution with mean and variance set as the initParams value
            if trials == 1:
                init = [initParams[key] for key in keys]
            else:
                init = [np.maximum(np.random.normal(initParams[key], initParams[key]), bounds[key][0]) for key in keys]  # Did not include a check against the bound maximum because None values are sometimes used. The minimizer should automatically correct this anyway
            results = minimize(self.negLogLikelihood, init, args = (keys, fixed_params, TTWArray),
                                bounds = [bounds[x] for x in keys])
            
            if results.success:
                finalParams.append(results.x)
                loglikelihood.append(results.fun)

        idx = np.nanargmin(loglikelihood)
        
        return dict(zip(keys, finalParams[idx])), loglikelihood[idx]
        

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

    def arrayFromProbability(self, P = None, households = None, minimiseRandomness = True):
        if P is None:
            P = self.mllProbability
        
        if households is None:
            households = self.totalHouseholds
        
        suburbs = self.arraySize
        if minimiseRandomness:
            # First assign some of the households to minimise randomness
            TTWArray_output = np.floor(P * households).astype(int)
        else:
            TTWArray_output = np.zeros_like(P)
        
        # Probabilistically assign remaining households
        households_distributed = np.sum(TTWArray_output).astype(int)
        householdTTWChoices = np.random.choice(range(suburbs[0] * suburbs[1]), size = households - households_distributed, p = P.ravel())
        householdPerTTW_output = np.bincount(householdTTWChoices, minlength = suburbs[0] * suburbs[1])
        TTWArray_output += householdPerTTW_output.reshape((suburbs[0], suburbs[1]))  # Generate a TTWArray_output
        
        self.mllArray = TTWArray_output
        self.mllHHDist = np.sum(TTWArray_output, axis = 1)
        return self.mllArray
        


def hellingerDistance(distA, distB):
    distance = (np.sqrt(distA) - np.sqrt(distB)) ** 2
    distance = np.sqrt(np.sum(distance)) / np.sqrt(2)
    return distance
