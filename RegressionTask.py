import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

input_data = pd.read_csv('dataPred1.csv')  # Collate task-specific data inputs
y = input_data.iloc[:, 0]
x = input_data.iloc[:, 1:]

# Standardisation
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
x = min_max_scaler.fit_transform(x)

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=0)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=0)

# Training
etr = ExtraTreesRegressor(n_jobs=-1, n_estimators=500, max_features='auto')
etr.fit(x_train, y_train)
etr_y_pred = etr.predict(x_test)

# Evaluations
mae_score = mean_absolute_error(y_test, etr_y_pred)
mse_score = mean_squared_error(y_test, etr_y_pred)
r_2_score = r2_score(y_test, etr_y_pred)
r_score = np.corrcoef(y_test, etr_y_pred, rowvar=0)[0][1]
print(mae_score, mse_score, r_2_score, r_score)

plt.scatter(y_test, y_test, s=11)
plt.scatter(y_test, etr_y_pred, s=11)
plt.show()

# Importance Analysis
print("Feature Importances:")
for importance in etr.feature_importances_:
    print(f'{importance:.3f}')
print('---------------------')

from sklearn.inspection import permutation_importance
result = permutation_importance(etr, x, y, n_repeats=10)
print("Permutation Importances:")
for importance_mean in result.importances_mean:
    print(f'{importance_mean:.3f}')
