import pandas as pd
# Task1 | base pred G_ad

input_data = pd.read_csv('../dataPred1.csv')
input_data2 = input_data.drop(input_data.index[422])


import numpy as np
from pandas import DataFrame
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.multioutput import MultiOutputRegressor
import joblib
from sklearn.model_selection import cross_val_score

y=input_data2.iloc[:,0]
x=input_data2.iloc[:,list(range(0,5))+list(range(11,27))]

#Normalize all data
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
x=min_max_scaler.fit_transform(x)
#Split the data into 80% train and 20% WorkFunction data sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#Extra Trees Regressor
etr=ExtraTreesRegressor(n_jobs=-1, n_estimators=500, max_features='auto')
#etr.fit(x_train,y_train)
#etr_y_pred=rfr.predict(x_test)
#person_etr=np.corrcoef(etr_y_pred,y_test,rowvar=0)[0][1]
#y_pred=DataFrame(np.array(etr_y_pred))
#y_testt=DataFrame(np.array(y_test))
#pd.concat([y_testt,y_pred],axis=1).to_csv('test1.csv',header=False,index=False)
etr.fit(x_train,y_train)
etr_y_pred=etr.predict(x_test)
y_test = np.asarray(y_test, dtype=float)
person_etr=np.corrcoef(y_test,etr_y_pred,rowvar=0)[0][1]
y_pred=DataFrame(np.array(etr_y_pred))
y_testt=DataFrame(np.array(y_test))
# pd.concat([y_testt,y_pred],axis=1).to_csv('test1.csv',header=False,index=False)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_percentage_error
mae_score = mean_absolute_error(y_test, etr_y_pred)
mse_score = mean_squared_error(y_test, etr_y_pred)
r_2_score = r2_score(y_test, etr_y_pred)
mape = mean_absolute_percentage_error(y_test, etr_y_pred)
plt.scatter(y_test, y_test, s=11)
plt.scatter(y_test, etr_y_pred, s=11)
plt.show()

# Importance analysis
for xx in etr.feature_importances_:
    print('%.3f'%xx)
print('---------------------')
result = permutation_importance(etr, x, y, n_repeats=10) #random_state=42,n_jobs=-1
for xx in result.importances_mean:
    print('%.3f'%xx)
