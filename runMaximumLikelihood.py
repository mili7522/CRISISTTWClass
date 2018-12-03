import TTWMaximumLikelihood as TTWML
import numpy as np
import pandas as pd
import json
import os
import sys



### Get command-line arguments
if len(sys.argv) > 1:
    model_number = int(sys.argv[1])
else:
    model_number = 2
    # Model 1: Gravitational; 2: Log; 3: c_w = 0; 4: c_l = 0 (gravitational); 5: c_l = 0 (log)

if len(sys.argv) > 2:
    bootstrap_id = int(sys.argv[2])
else:
    bootstrap_id = None

if len(sys.argv) > 3:
    year = int(sys.argv[3])
else:
    year = 2011

if bootstrap_id is None:
    trial_name = 'Sydney{}_{}-Data'.format(year, model_number)
else:
    trial_name = 'Sydney{}_{}-{:02d}'.format(year, model_number, bootstrap_id)
savePath = 'Results/{}'.format(trial_name)
os.makedirs(savePath, exist_ok=True)



### Load Sydney Geography
if bootstrap_id is None:
    TTW = pd.read_table('Data/{}_TTW.csv'.format(year), sep = ',', index_col= 0)
else:
    TTW = pd.read_table('Data/BootstrapTTW{}/BootstrapTTW{:02d}.csv'.format(year, bootstrap_id), sep = ',', index_col= 0)

rent = pd.read_csv('Data/{}_AveragedMedianValuesPerM-SA2.csv'.format(year), index_col = 0, usecols = [0,2], squeeze = True)

# Transpose TTW array
TTW = np.transpose(TTW)
TTW = TTW.values

distances = pd.read_csv("Data/{}_SA2-DriveTimes-OSRM.csv".format(year), header = None)
distances = ((distances + distances.transpose()) / 2 / 60).values + 5

a = TTWML.TTWML(distances, TTW, logForm = True if model_number in [2, 5] else False)

suburbs = len(TTW)
workDistribution = a.workDistribution
totalHouseholds = a.totalHouseholds



###  Obtain parameters using maximum likelihood
other_params = {'c_w': 0,
                'c_l': 0,
                'alpha_w': 1,
                'beta': 1,
                'local_energy': rent.values
                }

bounds = {'c_w': (0, None),
          'c_l': (0, None),
          'alpha_w': (-100, None)}


resultList = []
loglikelihoodList = []
for i in range(10):
    initParams = {'c_w': 5, 'c_l': 1, 'alpha_w': 5}
    if model_number == 3:
        del initParams['c_w']
    if model_number in [4,5]:
        del initParams['c_l']
    print(i)
    finalParams, loglikelihood = a.maximiseLikelihood(initParams, bounds, trials = 100, fixed_params = other_params, TTWArray = TTW)
    if loglikelihood is not None:
        resultList.append(finalParams)
        loglikelihoodList.append(loglikelihood)
        print('loglikelihoodList', loglikelihoodList)
        sys.stdout.flush()
    
r = pd.DataFrame(resultList)
r['NegLogLikelihood'] = loglikelihoodList
r.to_csv(os.path.join(savePath,'MinLLRepetitions{}.csv'.format(trial_name)), index = False)

print('Parameter Mean:', r.mean())
print('Parameter Median:', r.median())
print('Likelihood Mean:', np.mean(loglikelihoodList))

###
print('Minimum likelihood', np.min(loglikelihoodList))
argmin = np.argmin(loglikelihoodList)
r_min = resultList[argmin]

# Get P(i,j)
combined_params = {**other_params, **r_min}
P = a.probabilityFromParameters(combined_params, printParams = False)

# Save probability matrix
np.savetxt(os.path.join(savePath,'ProbabilityMatrix{}.csv'.format(trial_name)), P, delimiter = ',')
# Save the parameters
combined_params['local_energy'] = list(combined_params['local_energy'])
with open(os.path.join(savePath,'Parameters{}.txt'.format(trial_name)), 'w') as outfile:
    json.dump(combined_params, outfile)
combined_params['local_energy'] = np.array(combined_params['local_energy'])



p = P.ravel()
q = TTW.ravel() / totalHouseholds
print('---')
print('Hellinger Distance:', TTWML.hellingerDistance(p,q))
print('Jensen-Shannon Divergence:', TTWML.jsDivergence(p,q))