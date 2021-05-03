import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set()
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
from statsmodels.stats.outliers_influence import variance_inflation_factor    
from sklearn.cluster import KMeans

# import data as a dataframe from script python arg
brainMetsFeaturesRaw = pd.read_csv('./radiomicfeatures.csv')

# visualize data 
# NOTE: most of visualization done in MATLAB
pd.set_option('display.max_columns', 500)
brainMetsFeaturesRaw.columns
brainMetsFeaturesRaw
brainMetsFeaturesRaw.describe()

# remove letters from patient number
for i, name in enumerate(brainMetsFeaturesRaw.patient):
    brainMetsFeaturesRaw.patient[i] = (name.replace('FSRTCASE', '')).split('.')[0]

# create patient dataframe and covert patient id to numerical
patients = brainMetsFeaturesRaw['patient']
patients = pd.to_numeric(patients)

brainMetsMinusPatients = brainMetsFeaturesRaw.drop(columns = ['patient'])

# Apply min-max column wise normilization
rawComparisonData = pd.read_csv('rawComparisonPredictionData.csv')
rawComparisonData = rawComparisonData.drop(columns = ['patient'])

brainMetsSelectedFeaturesNormalized = (brainMetsMinusPatients-rawComparisonData.min())/(rawComparisonData.max()-rawComparisonData.min())

# Compare with static predictionData csv to keep certain columns
comparisonData = pd.read_csv('comparisonPredictionData.csv')
columnsCurrent = list(brainMetsSelectedFeaturesNormalized.columns.values)
columnsToKeep = list(comparisonData.columns.values)
columnsToDrop = []

brainMetsSelectedFeaturesFinal = brainMetsSelectedFeaturesNormalized

for col in columnsCurrent:
    if col not in columnsToKeep:
        columnsToDrop.append(col)

brainMetsSelectedFeaturesFinal.drop(columns=columnsToDrop)
brainMetsSelectedFeaturesFinal = brainMetsSelectedFeaturesFinal.dropna(axis='columns')

# add patient number back to dataframe
brainMetsSelectedFeaturesFinal['patient'] = patients 

# create outcome data dataframe
outcomeData = pd.read_csv('brainMets_features_survivalInDays.csv')

# join outcome and feature tables using patient number as key
predictingData = pd.merge(left=outcomeData, right=brainMetsSelectedFeaturesFinal, left_on='Study', right_on='patient')
predictingData = predictingData.drop(columns = [ 'Study','patient', 'ofMets', 'FSRTcourse', 'Ariacourse', 'Age', 'DateofdeathorLastFU', 'DateofBrainmetdiagnosis', 'FSRTcompletiondate', 'GTVVolumecc', 'GTVEqRadiuscm' ])



# save selected features as csv for prediction model
predictingData.to_csv(r'./finalPredictionData.csv', index = False)



