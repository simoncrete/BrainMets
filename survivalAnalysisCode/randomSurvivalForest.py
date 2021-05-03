#*****************************************************************************
# Simon Crete| 20047585 | 16sjcc
# PATH 828 PROJECT
# Random Survival Forest model training and evaluation
#*****************************************************************************

import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set()
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
from sklearn.model_selection import StratifiedKFold
from pysurvival.models.survival_forest import RandomSurvivalForestModel
from pysurvival.utils.metrics import concordance_index
from statistics import mean 
from pysurvival.utils.display import compare_to_actual
from pysurvival.utils.display import integrated_brier_score
from pysurvival.utils import save_model

# import selected features from previous script
predictionData = pd.read_csv('./predictionData.csv')
#predictionData.AliveStatus0Dead1Alive = predictionData.AliveStatus0Dead1Alive.eq(0).mul(1)

# create necessary variables for model training
time_column='NumDays'
event_column='AliveStatus0Dead1Alive'
onlyPredictionData = predictionData.drop(columns = [ 'AliveStatus0Dead1Alive','NumDays'])
features=(onlyPredictionData.columns).tolist()

# Create vars
X = predictionData.drop(columns = [ 'AliveStatus0Dead1Alive'])
E = predictionData['AliveStatus0Dead1Alive']

# create temp variables necessary for model parameter selection
Xtemp = X
Etemp = E
featuresTemp = features

# setting tested parameters for selection
num_tree=(10, 15, 20, 50, 100)
max_depth=(1, 2, 3, 5, 10, 12, 15)
min_node=(1, 2, 3, 5, 10, 12)

for a in num_tree:
    for b in max_depth:
        for c in min_node:
            cc = []
            kf = StratifiedKFold(n_splits=7, random_state=42, shuffle=True)
            i = 1
            for train_index, test_index in kf.split(Xtemp,Etemp):
                X1_train, X1_test = Xtemp.loc[train_index], Xtemp.loc[test_index]
                X_train, X_test = X1_train[featuresTemp], X1_test[featuresTemp]
                T_train, T_test = X1_train['NumDays'].values, X1_test['NumDays'].values
                E_train, E_test = Etemp.loc[train_index].values, Etemp.loc[test_index].values
                xst = RandomSurvivalForestModel(num_trees=a) 
                xst.fit(X_train, T_train, E_train, max_features = 'sqrt', max_depth = b,
                min_node_size = c, num_threads = -1, 
                sample_size_pct = 0.63, importance_mode = 'normalized_permutation',
                seed = None, save_memory = False )
                c_index = concordance_index(xst, X_test, T_test, E_test)
                cc.append(c_index)
                i = i+1
            print(a,b, c, mean(cc))
                    

CI = []
IBS = []
best_num_tree = 15
best_depth = 10
best_min_node = 5
k_folds = 4

i=1
kf=StratifiedKFold(n_splits = k_folds, random_state = 1, shuffle = True)
for train_index, test_index in kf.split(X,E):
    print('\n {} of {}'.format(i,kf.n_splits)) 
    X1_train, X1_test = X.loc[train_index], X.loc[test_index]
    X_train, X_test = X1_train[features], X1_test[features]
    T_train, T_test = X1_train['NumDays'].values, X1_test['NumDays'].values
    E_train, E_test = E.loc[train_index].values, E.loc[test_index].values
    xst = RandomSurvivalForestModel(num_trees=best_num_tree) 
    xst.fit(X_train, T_train, E_train, max_features = 'sqrt', max_depth = best_depth,
        min_node_size = best_min_node, num_threads = -1, 
        sample_size_pct = 0.63, importance_mode = 'normalized_permutation',
        seed = None, save_memory=False )
    c_index = concordance_index(xst, X_test, T_test, E_test)
        
    results = compare_to_actual(xst, X_test, T_test, E_test, is_at_risk = True,  figure_size=(16, 6), 
                                metrics = ['rmse', 'mean', 'median'])
    ibs = integrated_brier_score(xst, X_test, T_test, E_test, t_max=2000, figure_size=(15,5))
    CI.append(c_index)
    IBS.append(ibs)
    print('C-index: {:.2f}'.format(c_index))
    print('IBS: {:.2f}'.format(ibs))
    i = i+1            

# Save the model for use in pipeline
#save_model(xst, '../pipeline/survival_model.zip')         

xst.variable_importance_table.head(20)     
preds = xst.predict_survival(onlyPredictionData.iloc[:,:-2].transpose())
preds_df = pd.DataFrame(preds).T
preds_df.to_excel('preds.xlsx')            
            
            
def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x

preds = pd.DataFrame(columns = ['Preds','Actual'])
preds['Preds'] = xst.times
actuals = selection_sort(predictionData['NumDays'])
preds['Actual'] = predictionData['NumDays']
print(preds)            
            