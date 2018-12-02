import numpy as np
import pandas as pd
import os

year = 2016

savePath = 'Data/BootstrapTTW{}/'.format(year)
os.makedirs(savePath, exist_ok=True)

### Load Sydney Geography
# TTW = pd.read_table('Data/{}_TTW.csv'.format(year), sep = ',', header = None)  # TODO: Need to fix for 2011
TTW = pd.read_table('Data/{}_TTW.csv'.format(year), sep = ',', index_col= 0, header = 0)
suburb_id = TTW.index  # SA2_MAINCODE_2016
TTW = TTW.transpose().values


### Get current households according to TTW data
householdsHome = []
householdsWork = []
for i,row in enumerate(TTW):
    for j,n in enumerate(row):
        householdsHome += [i]*n
        householdsWork += [j]*n

realHouseholds = np.vstack([householdsHome, householdsWork]).T
suburbs = len(suburb_id)

def bootstapHouseholds(households):
    '''Creates bootstrap datasets from the original TTW array

    Keywork arguments:
    households -- A numpy array of two columns and one row for each household.
                  The first column is the home suburb id, the second column is the work suburb id.
    '''
    redrawnIdx = np.random.choice(range(len(households)), len(households), replace = True)
    redrawnHouseholds = households[redrawnIdx]
    return redrawnHouseholds

def getTTW(households):
    '''Creates TTW array using a numpy array of households (1st column is home suburb id, 2nd column is work suburb id)'''
    households = pd.DataFrame(households)
    distributionTable = np.zeros((suburbs, suburbs), dtype = int)
    for row in range(suburbs):
        distributionTable[row] = np.bincount(households[households[0] == row][1], minlength = suburbs)
    return distributionTable

###
np.random.seed(10)
for i in range(50):
    redrawnHouseholds = bootstapHouseholds(realHouseholds)
    newTTW = getTTW(redrawnHouseholds)
    newTTWDf = pd.DataFrame(newTTW, index = suburb_id, columns = suburb_id).transpose()  # Transpose to match original TTW csv
    newTTWDf.to_csv(savePath + 'BootstrapTTW{:02d}.csv'.format(i))