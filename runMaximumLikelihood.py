import TTWMaximumLikelihood as TTWML
import numpy as np
import pandas as pd
import scipy.stats
import json
import os
import sys



### Get command-line arguments
if len(sys.argv) > 1:
    model_number = int(sys.argv[1])
else:
    model_number = 1
    # Model 1: Gravitational; 2: Log; 3: c_w = 0; 4: c_l = 0 (gravitational); 5: c_l = 0 (log)

if len(sys.argv) > 2:
    bootstrap_id = int(sys.argv[2])
else:
    bootstrap_id = None

if len(sys.argv) > 3:
    year = int(sys.argv[3])
else:
    year = 2016

trial_name = 'Sydney{}{}'.format(model_number, year)
savePath = 'Results/{}'.format(trial_name)
os.makedirs(savePath, exist_ok=True)



### Load Sydney Geography
if bootstrap_id is None:
    TTW = pd.read_table('Data/{}_TTW.csv'.format(year), sep = ',', index_col= 0)  # TODO: Fix 2011 import
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
fixed_params = {'c_w': 0,
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
for i in range(30):
    initParams = {'c_w': 5, 'c_l': 1, 'alpha_w': 5}
    if model_number == 3:
        del initParams['c_w']
    if model_number in [4,5]:
        del initParams['c_l']
    print(i)
    finalParams, loglikelihood = a.maximiseLikelihood(initParams, bounds, trials = 2000, fixed_params = fixed_params, TTWArray = TTW)
    resultList.append(finalParams)
    loglikelihoodList.append(loglikelihood)
    print('loglikelihoodList', loglikelihoodList)
    
r = pd.DataFrame(resultList)
# sns.pairplot(r, markers="x", plot_kws = {'alpha': 0.1})
# plt.savefig(savePath + 'Generated-Recovered4Params{}.png'.format(trial), dpi = 250, format = 'png', bbox_inches = 'tight')
# plt.close()

print('Parameter Mean:', r.mean())
print('Parameter Median:', r.median())
print('Likelihood Mean:', np.mean(loglikelihoodList))

###
print('Minimum likelihood', np.min(loglikelihoodList))
argmin = np.argmin(loglikelihoodList)
r = resultList[argmin]

combined_params = {**fixed_params, **r}

P = a.probabilityFromParameters(combined_params)

# Save probability matrix
np.savetxt(savePath + 'ProbabilityMatrix{}.csv'.format(trial_name), P, delimiter = ',')
# Save the parameters
combined_params['local_energy'] = list(combined_params['local_energy'])
with open(savePath + 'Parameters{}.txt'.format(trial_name), 'w') as outfile:
    json.dump(combined_params, outfile)
combined_params['local_energy'] = np.array(combined_params['local_energy'])

# plotTTW(np.round(P * np.sum(TTW)), saveName = savePath + 'RecoveredTTW{}.png'.format(trial_name))
# plotTTW(np.log(np.round(P * np.sum(TTW))), saveName = savePath + 'RecoveredTTWLog{}.png'.format(trial_name))

p = P.ravel()
q = TTW.ravel() / totalHouseholds
print('---')
print('Hellinger Distance:', TTWML.hellingerDistance(q, p))
m = (p + q) / 2
js = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2
print('Jensen-Shannon Divergence:', js)

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


### Comparison with null model

# a.randomiseEdges()
# TTWRand = a.TTWArray

# q = TTWRand.ravel() / totalHouseholds
# print('---')
# print('Hellinger Distance - With Null:', a.hellingerDistance(q, p))
# m = (p + q) / 2
# js = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2
# print('Jensen-Shannon Divergence - With Null:', js)
