
!pip install tensorflow pandas numpy sklearn

!pip install pandas scikit-learn
!pip install openpyxl

import pandas as pd
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.metrics import AUC
from tensorflow.keras.metrics import Precision, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, AUC
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from keras.layers import SimpleRNN, Dense
import random
import tensorflow as tf
from google.colab import drive

drive.mount('/content/drive')

data_path_UCI = '[UCI_dataset_path]'

"""# UCI dataset

## 1. Data stats
"""

df = pd.read_csv(data_path_UCI)

df.dtypes

df.isna().sum()

df.info()

df.drop('id', axis=1, inplace = True)
df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'aanemia', 'class']

df.shape

df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')
df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')
df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')

df.columns

df['serum_creatinine'].head()

df['serum_creatinine'] = df['serum_creatinine']*88.4

df['serum_creatinine'].info()

cat_cols = [col for col in df.columns if df[col].dtype == 'object']
num_cols = [col for col in df.columns if df[col].dtype != 'object']

for col in cat_cols:
    print(f"{col}: {df[col].unique()}")

df['diabetes_mellitus'].replace(to_replace = { '\tno':'no', '\tyes':'yes', ' yes':'yes'}, inplace=True)
df['coronary_artery_disease'].replace(to_replace = {'\tno':'no'}, inplace = True)
df['class'].replace(to_replace = {'ckd\t':'ckd', 'notckd': 'not ckd'}, inplace = True)

for col in cat_cols:
    print(f"{col}: {df[col].unique()}")

df['class'] = df['class'].map({'ckd':0, 'not ckd':1})
df['class'] = pd.to_numeric(df['class'], errors = 'coerce')

for col in ['diabetes_mellitus','coronary_artery_disease','class' ]:
  print(f" {col} has {df[col].unique()} ")

"""##Data pre-processing"""

df.head(10)

def random_sampling(feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample

def impute_mode(feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)

# random sampling for numerical value
for col in num_cols:
    random_sampling(col)

df[num_cols].isnull().sum()

for col in cat_cols:
    impute_mode(col)

df[cat_cols].isnull().sum()

df.head()

df.info()

"""##Feature encoding"""

for col in cat_cols:
    print(f"{col} has {df[col].unique()}")

# label_encoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
# print("Label Mapping:", label_mapping)

df.head()

"""#Checking the performance of the model using the features slected"""

df.info()



selected_features = ['specific_gravity', 'albumin', 'serum_creatinine', 'haemoglobin', 'packed_cell_volume',
       'diabetes_mellitus','red_blood_cell_count']
target_variable = 'class'



"""#Logistic regression"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression


X = df[selected_features]
y = df[target_variable]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=1000)

accuracies = []
precisions = []
recalls = []
f1s = []
specificities = []
roc_aucs = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))
    specificities.append(specificity)
    roc_aucs.append(roc_auc_score(y_test, y_prob))

print(f"Average Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Average Precision: {np.mean(precisions):.4f} (+/- {np.std(precisions):.4f})")
print(f"Average Recall: {np.mean(recalls):.4f} (+/- {np.std(recalls):.4f})")
print(f"Average F1 Score: {np.mean(f1s):.4f} (+/- {np.std(f1s):.4f})")
print(f"Average Specificity: {np.mean(specificities):.4f} (+/- {np.std(specificities):.4f})")
print(f"Average ROC-AUC: {np.mean(roc_aucs):.4f} (+/- {np.std(roc_aucs):.4f})")

y_pred_cv = cross_val_predict(model, X, y, cv=skf, method='predict')
conf_matrix_cv = confusion_matrix(y, y_pred_cv)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_cv, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Cross-validated Confusion Matrix')
plt.show()

print("\nCross-validated classification report:")
print(classification_report(y, y_pred_cv))

"""#Decision Tree"""

from sklearn.tree import DecisionTreeClassifier


X = df[selected_features]
y = df[target_variable]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = DecisionTreeClassifier(random_state=42)

accuracies = []
precisions = []
recalls = []
f1s = []
specificities = []
roc_aucs = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))
    specificities.append(specificity)
    roc_aucs.append(roc_auc_score(y_test, y_prob))

print(f"Average Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Average Precision: {np.mean(precisions):.4f} (+/- {np.std(precisions):.4f})")
print(f"Average Recall: {np.mean(recalls):.4f} (+/- {np.std(recalls):.4f})")
print(f"Average F1 Score: {np.mean(f1s):.4f} (+/- {np.std(f1s):.4f})")
print(f"Average Specificity: {np.mean(specificities):.4f} (+/- {np.std(specificities):.4f})")
print(f"Average ROC-AUC: {np.mean(roc_aucs):.4f} (+/- {np.std(roc_aucs):.4f})")

y_pred_cv = cross_val_predict(model, X, y, cv=skf, method='predict')
conf_matrix_cv = confusion_matrix(y, y_pred_cv)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_cv, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Cross-validated Confusion Matrix')
plt.show()

print("\nCross-validated classification report:")
print(classification_report(y, y_pred_cv))

"""#XGBoost"""

from xgboost import XGBClassifier

X = df[selected_features]
y = df[target_variable]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = XGBClassifier(max_depth=6, n_estimators=100, random_state=42)

accuracies = []
precisions = []
recalls = []
f1s = []
specificities = []
roc_aucs = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))
    specificities.append(specificity)
    roc_aucs.append(roc_auc_score(y_test, y_prob))

print(f"Average Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Average Precision: {np.mean(precisions):.4f} (+/- {np.std(precisions):.4f})")
print(f"Average Recall: {np.mean(recalls):.4f} (+/- {np.std(recalls):.4f})")
print(f"Average F1 Score: {np.mean(f1s):.4f} (+/- {np.std(f1s):.4f})")
print(f"Average Specificity: {np.mean(specificities):.4f} (+/- {np.std(specificities):.4f})")
print(f"Average ROC-AUC: {np.mean(roc_aucs):.4f} (+/- {np.std(roc_aucs):.4f})")

"""#LightGB"""

import lightgbm as lgb

X = df[selected_features]
y = df[target_variable]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = lgb.LGBMClassifier(max_depth=6, n_estimators=100, random_state=42)

accuracies = []
precisions = []
recalls = []
f1s = []
specificities = []
roc_aucs = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))
    specificities.append(specificity)
    roc_aucs.append(roc_auc_score(y_test, y_prob))

print(f"Average Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Average Precision: {np.mean(precisions):.4f} (+/- {np.std(precisions):.4f})")
print(f"Average Recall: {np.mean(recalls):.4f} (+/- {np.std(recalls):.4f})")
print(f"Average F1 Score: {np.mean(f1s):.4f} (+/- {np.std(f1s):.4f})")
print(f"Average Specificity: {np.mean(specificities):.4f} (+/- {np.std(specificities):.4f})")
print(f"Average ROC-AUC: {np.mean(roc_aucs):.4f} (+/- {np.std(roc_aucs):.4f})")

y_pred_cv = cross_val_predict(model, X, y, cv=skf, method='predict')
conf_matrix_cv = confusion_matrix(y, y_pred_cv)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_cv, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Cross-validated Confusion Matrix')
plt.show()

print("\nCross-validated classification report:")
print(classification_report(y, y_pred_cv))

"""#RF"""

from sklearn.ensemble import RandomForestClassifier

X = df[selected_features]
y = df[target_variable]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(max_depth=6, n_estimators=100, random_state=42)

accuracies = []
precisions = []
recalls = []
f1s = []
specificities = []
roc_aucs = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))
    specificities.append(specificity)
    roc_aucs.append(roc_auc_score(y_test, y_prob))


print(f"Average Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Average Precision: {np.mean(precisions):.4f} (+/- {np.std(precisions):.4f})")
print(f"Average Recall: {np.mean(recalls):.4f} (+/- {np.std(recalls):.4f})")
print(f"Average F1 Score: {np.mean(f1s):.4f} (+/- {np.std(f1s):.4f})")
print(f"Average Specificity: {np.mean(specificities):.4f} (+/- {np.std(specificities):.4f})")
print(f"Average ROC-AUC: {np.mean(roc_aucs):.4f} (+/- {np.std(roc_aucs):.4f})")

y_pred_cv = cross_val_predict(model, X, y, cv=skf, method='predict')
conf_matrix_cv = confusion_matrix(y, y_pred_cv)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_cv, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Cross-validated Confusion Matrix')
plt.show()

print("\nCross-validated classification report:")
print(classification_report(y, y_pred_cv))