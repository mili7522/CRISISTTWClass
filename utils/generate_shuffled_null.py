import numpy as np
import pandas as pd
import os
from utils.generate_bootstrap import getTTW

year = 2016

savePath = 'Data/'

### Load Sydney Geography
TTW = pd.read_table('Data/{}_TTW.csv'.format(year), sep = ',', index_col= 0, header = 0)
suburb_id = TTW.index  # SA2_MAINCODE
TTW = TTW.transpose().values

### Get current households according to TTW data
householdsHome = []
householdsWork = []
for i,row in enumerate(TTW):
    for j,n in enumerate(row):
        householdsHome += [i]*n
        householdsWork += [j]*n

np.random.seed(10)
np.random.shuffle(householdsWork)  # Modifies in-place
shuffledHouseholds = np.vstack([householdsHome, householdsWork]).T

newTTW = getTTW(shuffledHouseholds, len(suburb_id))
newTTWDf = pd.DataFrame(newTTW, index = suburb_id, columns = suburb_id).transpose()  # Transpose to match original TTW csv
newTTWDf.to_csv(os.path.join(savePath, 'ShuffledTTW{}.csv'.format(year)))