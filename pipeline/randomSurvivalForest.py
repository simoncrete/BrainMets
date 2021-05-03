import numpy as np
import pandas as pd
from pysurvival.utils import load_model
import matplotlib.pyplot as plt

# Import selected features from finalPrediction csv
predictionData = pd.read_csv('./finalPredictionData.csv')
# Remove unnecessary columns
onlyPredictionData = predictionData.drop(columns = [ 'AliveStatus0Dead1Alive','NumDays'])

survivalModel = load_model('./survival_model.zip')

preds = survivalModel.predict_survival(onlyPredictionData)
preds_df = pd.DataFrame(preds).T
preds_df.to_excel('preds.xlsx') 

plt.plot(preds_df, label = "Survival Data")
plt.legend()
plt.show()
