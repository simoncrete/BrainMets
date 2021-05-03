#*****************************************************************************
# Simon Crete| 20047585 | 16sjcc
# PATH 828 PROJECT
# Feature selection
#*****************************************************************************

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

# import data as a dataframe
brainMetsFeaturesRaw = pd.read_csv('./brainMetsMriRadiomicFeatures.csv')

# visualize data 
# NOTE: most of visualization done in MATLAB
pd.set_option('display.max_columns', 500)
brainMetsFeaturesRaw.columns
brainMetsFeaturesRaw
brainMetsFeaturesRaw.describe()

# visualize correlations of all features as a heatmap (VERY SLOW)
#corrMatrix = brainMetsFeaturesRaw.corr()
#plt.rcParams['figure.figsize'] = [40, 20]
#fig = sns.heatmap(corrMatrix, annot=True)
#plt.show()

# remove letters from patient number
for i, name in enumerate(brainMetsFeaturesRaw.patient):
    brainMetsFeaturesRaw.patient[i] = name.replace('FSRTCASE', '')

# create patient dataframe and covert patient id to numerical
patients = brainMetsFeaturesRaw['patient']
patients = pd.to_numeric(patients)

# Remove colinear features
X = brainMetsFeaturesRaw.drop(columns = ['patient'])
thresh = 5.0
variables = list(range(X.shape[1]))
dropped = True
while dropped:
    dropped = False
    vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
         for ix in range(X.iloc[:, variables].shape[1])]
    maxloc = vif.index(max(vif))
    if max(vif) > thresh:
        print('dropping \'' + X.iloc[:, variables].columns[maxloc] +'\' at index: ' + str(maxloc))
        del variables[maxloc]
        dropped = True
print('Remaining variables:')
print(X.columns[variables])
brainMetsSelectedFeatures = X.iloc[:, variables]

# Apply min-max column wise normilization
brainMetsSelectedFeaturesNormalized = (brainMetsSelectedFeatures-brainMetsSelectedFeatures.min())/(brainMetsSelectedFeatures.max()-brainMetsSelectedFeatures.min())

#Remove low variance features
for cols in brainMetsSelectedFeaturesNormalized.columns:
    if np.var(brainMetsSelectedFeaturesNormalized[cols]) <= 0.1:
        print('Dropping: ' + cols)
        brainMetsSelectedFeaturesFinal = brainMetsSelectedFeaturesNormalized.drop(columns=cols)
brainMetsSelectedFeaturesFinal= brainMetsSelectedFeaturesNormalized

# add patient number back to dataframe
brainMetsSelectedFeaturesFinal['patient'] = patients 

# create outcome data dataframe
outcomeData = pd.read_csv('brainMets_features_survivalInDays.csv')

# join outcome and feature tables using patient number as key
predictingData = pd.merge(left=outcomeData, right=brainMetsSelectedFeaturesFinal, left_on='Study', right_on='patient')
predictingData = predictingData.drop(columns = [ 'Study','patient', 'ofMets', 'FSRTcourse', 'Ariacourse', 'Age', 'DateofdeathorLastFU', 'DateofBrainmetdiagnosis', 'FSRTcompletiondate', 'GTVVolumecc', 'GTVEqRadiuscm' ])

# visualize correlations as a heatmap
corrMatrix = predictingData.corr()
plt.rcParams['figure.figsize'] = [40, 20]
fig = sns.heatmap(corrMatrix, annot=True)
plt.show()

# save selected features as csv for prediction model
predictingData.to_csv(r'./predictionData.csv', index = False)


# Clustering
#data = predictingData[['NumDays', 'original_shape_MajorAxisLength']]
#data['NumDays'] = np.log2(data['NumDays']) 
#kmeans = KMeans(n_clusters=2).fit(data)
#centroids = kmeans.cluster_centers_
#print(centroids)
#plt.scatter(data['NumDays'], data['original_shape_MajorAxisLength'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
#plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
#plt.show()
