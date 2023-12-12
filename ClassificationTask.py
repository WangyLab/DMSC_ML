import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

input_data = pd.read_csv('Pred3.csv')  # Dataset
y = input_data2.iloc[:, 1].astype(int)
x = input_data2.iloc[:, 2:]

# Standardisation
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
x = min_max_scaler.fit_transform(x)

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=0)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=0)

# Training
etr = ExtraTreesClassifier(n_estimators=1000, max_features='auto')
etr.fit(x_train, y_train)

# Evaluations
etr_y_pred_val = etr.predict(x_val)
roc_val = roc_auc_score(y_val, etr_y_pred_val)
acc_val = accuracy_score(y_val, etr_y_pred_val)
print("Validation Set Metrics:")
print("ROC AUC Score:", roc_val)
print("Accuracy:", acc_val)

etr_y_pred_test = etr.predict(x_test)
roc_test = roc_auc_score(y_test, etr_y_pred_test)
acc_test = accuracy_score(y_test, etr_y_pred_test)
print("Test Set Metrics:")
print("ROC AUC Score:", roc_test)
print("Accuracy:", acc_test)

# Importance Analysis
print("Feature Importances:")
for importance in etr.feature_importances_:
    print(f'{importance:.3f}')

from sklearn.inspection import permutation_importance
result = permutation_importance(etr, x, y, n_repeats=1)
print("Permutation Importances:")
for importance_mean in result.importances_mean:
    print(f'{importance_mean:.3f}')
