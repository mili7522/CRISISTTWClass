import TTWMaximumLikelihood as TTWML
from utils import utils
import numpy as np
import pandas as pd
import json
import os
import sys



### Get command-line arguments
if len(sys.argv) > 1:
    model_number = int(sys.argv[1])
else:
    model_number = 4
    # Model 1: Gravitational; 2: Log; 3: c_w = 0; 4: c_l = 0 (gravitational); 5: c_l = 0 (log)

if len(sys.argv) > 2:
    if sys.argv[2].lower() == 'none' or sys.argv[2].lower() == 'data':
        bootstrap_id = None
    else:
        bootstrap_id = "{:02d}".format(int(sys.argv[2]))
else:
    bootstrap_id = None

if len(sys.argv) > 3:
    year = int(sys.argv[3])
else:
    year = 2011

if bootstrap_id is None:
    trial_name = 'Sydney{}_{}-Data'.format(year, model_number)
else:
    trial_name = 'Sydney{}_{}-{}'.format(year, model_number, bootstrap_id)
savePath = 'Results/{}'.format(trial_name)
os.makedirs(savePath, exist_ok=True)



### Load Sydney Geography and Data
TTW, distances, rent = utils.loadData(year, model_number, bootstrap_id)

a = TTWML.TTWML(distances, TTW, logForm = True if model_number in [2, 5] else False)


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
for i in range(20):
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

print('Mean of repeats:\n', r.mean())
print('Std of repeats:\n', r.std())


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
q = TTW.ravel() / a.totalHouseholds
print('---')
print('Hellinger Distance:', utils.hellingerDistance(p,q))
print('Jensen-Shannon Divergence:', utils.jsDivergence(p,q))