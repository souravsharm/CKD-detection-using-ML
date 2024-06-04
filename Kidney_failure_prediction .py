

!pip install tensorflow pandas numpy sklearn
!pip install xgboost
!pip install lightgbm
!pip install lime
!pip install shap

!pip install pandas scikit-learn

import pandas as pd
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.metrics import AUC
from tensorflow.keras.metrics import Precision, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, AUC
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.linear_model import LinearRegression
from keras.layers import SimpleRNN, Dense
import random
import tensorflow as tf
from google.colab import drive

drive.mount('/content/drive')


processed_data_path = '[Place the path for the data]'
df = pd.read_csv(processed_data_path)

df.isna().sum()

# Train and Evaluate Models

# Map the 'Time' column to integer values
time_mapping = {
    'eGFR(0M)': 0,
    'eGFR(6M)': 6,
    'eGFR(12M)': 12,
    'eGFR(18M)': 18,
    'eGFR(24M)': 24,
    'eGFR(30M)': 30,
    'eGFR(36M)': 36
}

# Apply the mapping to create a new 'time_category' column
df['time_category'] = df['Time'].map(time_mapping)

# Fill any NaN values in 'time_category' with a default value, for example, 0
df['time_category'].fillna(0, inplace=True)

# Function to calculate the slope of eGFR over time
def compute_slope(group):
    x = np.array(group['time_category']).reshape(-1, 1)
    y = np.array(group['eGFR'])

    if len(x) < 2:
        return 0.0

    return LinearRegression().fit(x, y).coef_[0]

# Group the DataFrame by 'ID'
grouped = df.groupby('ID')

# Compute slopes for each group
slopes = grouped.apply(lambda group: compute_slope(group))

# Remove the existing 'egfr_slope' column if it exists
if 'egfr_slope' in df.columns:
    df.drop(columns=['egfr_slope'], inplace=True)

# Assign slopes back to the DataFrame
df = df.join(slopes.rename('egfr_slope'), on='ID')

# Calculate mean eGFR for each group
df['egfr_mean'] = grouped['eGFR'].transform(np.mean)

# Fill any NaN values that might have cropped up during calculation
df.fillna(0, inplace=True)

# Display the DataFrame after all the calculations
print(df.head())

df[['egfr_slope']].sum()


"""### *UNTUNED MODEL"""


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, accuracy_score
import random
import xgboost as xgb
import lightgbm as lgb

# Set random seeds
seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)

# Adjust this threshold as needed
threshold = 0.5

# Create a dataframe to store the results
df_results = pd.DataFrame(columns=['TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives',
                                   'TPR', 'TNR', 'FPR', 'FNR',
                                   'Accuracy', 'Precision', 'Recall', 'f1_score', 'Specificity', 'ROC-AUC'])

# Dataset features and target
features = ['AGE', 'SEX', 'egfr_slope', 'egfr_mean'] #'UACR_mean', 'UACR_std'
X = df[features]
y = df['Kidney_Failure']

# Use Stratified KFold for cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_value)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train a Decision Tree model
    clf  = DecisionTreeClassifier(max_depth=15, random_state=seed_value) #
    clf.fit(X_train, y_train)

    #print("Feature importances:", clf.feature_importances_)

    # Evaluate on the held-out test set
    y_pred = clf.predict(X_test)
    print("Final Accuracy:", accuracy_score(y_test, y_pred))
    print("Final Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Final ROC-AUC Score:")
    print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

    # Compute the statistics
    true_positives = np.sum((y_pred == 1) & (y_test == 1))
    true_negatives = np.sum((y_pred == 0) & (y_test == 0))
    false_positives = np.sum((y_pred == 1) & (y_test == 0))
    false_negatives = np.sum((y_pred == 0) & (y_test == 1))

    # Print individual metrics
    tpr = true_positives / (true_positives + false_negatives)
    tnr = true_negatives / (true_negatives + false_positives)
    fpr = false_positives / (false_positives + true_negatives)
    fnr = false_negatives / (false_negatives + true_positives)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = 2 * precision * recall / (precision + recall)
    specificity = true_negatives / (true_negatives + false_positives)
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    # Add the statistics to the dataframe
    df_results.loc[len(df_results)] = [true_positives, true_negatives, false_positives, false_negatives, tpr, tnr, fpr, fnr, accuracy, precision, recall, f1_score, specificity, roc_auc]

# Print the average of each statistic over the 5 folds
print(df_results.mean())

# Train the final model on the full Australian dataset
clf = DecisionTreeClassifier(max_depth=15, random_state=seed_value) #
clf.fit(X, y)

"""### *TUNING"""

# DT TUNING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay
import random

# Set random seeds
seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)

# Adjust this threshold as needed
threshold = 0.5

# Create a dataframe to store the results
df_results = pd.DataFrame(columns=['TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives',
                                   'TPR', 'TNR', 'FPR', 'FNR',
                                   'Accuracy', 'Precision', 'Recall', 'f1_score', 'Specificity', 'ROC-AUC'])

# Sample dataset features and target
features = ['AGE', 'SEX', 'egfr_slope', 'egfr_mean']
X = df[features]
y = df['Kidney_Failure']

# Initialize accumulators for confusion matrix elements
total_true_positives = 0
total_true_negatives = 0
total_false_positives = 0
total_false_negatives = 0

# Initialize a Decision Tree model
clf = DecisionTreeClassifier(random_state=seed_value) #

param_grid = {
    #'ccp_alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
    'max_depth': [5, 10, 15, 20, 25, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    #'max_features': ['auto', 'sqrt', 'log2', None],
    #'max_leaf_nodes': [None, 10, 20, 30, 40, 50],
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'class_weight': ['balanced', None]
}                                                     #


# Initialize GridSearch
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                           cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_value),
                           scoring='roc_auc', n_jobs=-1, verbose=1)

# Fit the GridSearch model
grid_search.fit(X, y)

# Extract the best estimator
best_clf = grid_search.best_estimator_

# Use Stratified KFold for cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_value)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train a Decision Tree model using the best estimator
    best_clf.fit(X_train, y_train)

    # Evaluate on the held-out test set
    y_pred = best_clf.predict(X_test)
    print("Final Accuracy:", accuracy_score(y_test, y_pred))
    print("Final Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Final ROC-AUC Score:")
    print(roc_auc_score(y_test, best_clf.predict_proba(X_test)[:, 1]))

    # Compute the statistics
    true_positives = np.sum((y_pred == 1) & (y_test == 1))
    true_negatives = np.sum((y_pred == 0) & (y_test == 0))
    false_positives = np.sum((y_pred == 1) & (y_test == 0))
    false_negatives = np.sum((y_pred == 0) & (y_test == 1))

    # Accumulate confusion matrix values
    total_true_positives += true_positives
    total_true_negatives += true_negatives
    total_false_positives += false_positives
    total_false_negatives += false_negatives

    # Print individual metrics
    tpr = true_positives / (true_positives + false_negatives)
    tnr = true_negatives / (true_negatives + false_positives)
    fpr = false_positives / (false_positives + true_negatives)
    fnr = false_negatives / (false_negatives + true_positives)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = 2 * precision * recall / (precision + recall)
    specificity = true_negatives / (true_negatives + false_positives)
    roc_auc = roc_auc_score(y_test, best_clf.predict_proba(X_test)[:, 1])

    print(accuracy, precision, recall, f1_score, specificity, roc_auc)

    # Add the statistics to the dataframe
    df_results.loc[len(df_results)] = [true_positives, true_negatives, false_positives, false_negatives, tpr, tnr, fpr, fnr, accuracy, precision, recall, f1_score, specificity, roc_auc]

# Compute mean confusion matrix values
mean_true_positives = total_true_positives / n_splits
mean_true_negatives = total_true_negatives / n_splits
mean_false_positives = total_false_positives / n_splits
mean_false_negatives = total_false_negatives / n_splits

# Construct the mean confusion matrix
mean_confusion_matrix = np.array([[mean_true_negatives, mean_false_positives],
                                 [mean_false_negatives, mean_true_positives]])

# Plot the mean confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=mean_confusion_matrix, display_labels=['No Failure', 'Kidney Failure'])
cm_display.plot(cmap='Blues')  # Using the 'Blues' colormap
plt.title('Mean Confusion Matrix')
plt.show()

# Print the average of each statistic over the 5 folds
print(df_results.mean())

# Print out the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Train the final model on the full Australian dataset
best_clf = DecisionTreeClassifier(**best_params, random_state=seed_value) #
best_clf.fit(X, y)



# Assuming clf is your trained decision tree classifier
# features is the list of feature names, e.g., ['age', 'gender', 'eGFR mean', 'eGFR slope']

importances = best_clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

#Feature importances: [0.18257789 0.03538208 0.03366508 0.74837495]

"""#XGB"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, accuracy_score
import random
import xgboost as xgb
import lightgbm as lgb


seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)

# Adjust this threshold as needed
threshold = 0.5

# Create a dataframe to store the results
df_results = pd.DataFrame(columns=['TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives',
                                   'TPR', 'TNR', 'FPR', 'FNR',
                                   'Accuracy', 'Precision', 'Recall', 'f1_score', 'Specificity', 'ROC-AUC'])

# Dataset features and target
features = ['AGE', 'SEX', 'egfr_slope', 'egfr_mean']
X = df[features]
y = df['Kidney_Failure']

# Use Stratified KFold for cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_value)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train a XGBoost model
    clf  =  xgb.XGBClassifier(objective='binary:logistic', seed=seed_value) #
    clf.fit(X_train, y_train)

    #print("Feature importances:", clf.feature_importances_)

    # Evaluate on the held-out test set
    y_pred = clf.predict(X_test)
    print("Final Accuracy:", accuracy_score(y_test, y_pred))
    print("Final Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Final ROC-AUC Score:")
    print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

    # Compute the statistics
    true_positives = np.sum((y_pred == 1) & (y_test == 1))
    true_negatives = np.sum((y_pred == 0) & (y_test == 0))
    false_positives = np.sum((y_pred == 1) & (y_test == 0))
    false_negatives = np.sum((y_pred == 0) & (y_test == 1))

    # Print individual metrics
    tpr = true_positives / (true_positives + false_negatives)
    tnr = true_negatives / (true_negatives + false_positives)
    fpr = false_positives / (false_positives + true_negatives)
    fnr = false_negatives / (false_negatives + true_positives)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = 2 * precision * recall / (precision + recall)
    specificity = true_negatives / (true_negatives + false_positives)
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    # Add the statistics to the dataframe
    df_results.loc[len(df_results)] = [true_positives, true_negatives, false_positives, false_negatives, tpr, tnr, fpr, fnr, accuracy, precision, recall, f1_score, specificity, roc_auc]

# Print the average of each statistic over the 5 folds
print(df_results.mean())

# Train the final model on the full Australian dataset
clf =  xgb.XGBClassifier(objective='binary:logistic', seed=seed_value) #
clf.fit(X, y)

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, accuracy_score
import random
import xgboost as xgb
import lightgbm as lgb

# Set random seeds
seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)

# Adjust this threshold as needed
threshold = 0.5

# Create a dataframe to store the results
df_results = pd.DataFrame(columns=['TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives',
                                   'TPR', 'TNR', 'FPR', 'FNR',
                                   'Accuracy', 'Precision', 'Recall', 'f1_score', 'Specificity', 'ROC-AUC'])

# Sample dataset features and target
features = ['AGE', 'SEX', 'egfr_slope', 'egfr_mean']
X = df[features]
y = df['Kidney_Failure']

# Initialize a Decision Tree model
clf =  xgb.XGBClassifier(objective='binary:logistic', seed=seed_value) #

param_grid = {
   'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 300],
    # 'max_depth': [3, 5],

}                                                #

# Initialize GridSearch
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                           cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_value),
                           scoring='roc_auc', n_jobs=-1, verbose=1)

# Fit the GridSearch model
grid_search.fit(X, y)

# Extract the best estimator
best_clf = grid_search.best_estimator_

# Use Stratified KFold for cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_value)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train a Decision Tree model using the best estimator
    best_clf.fit(X_train, y_train)

    # Evaluate on the held-out test set
    y_pred = best_clf.predict(X_test)
    print("Final Accuracy:", accuracy_score(y_test, y_pred))
    print("Final Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Final ROC-AUC Score:")
    print(roc_auc_score(y_test, best_clf.predict_proba(X_test)[:, 1]))

    # Compute the statistics
    true_positives = np.sum((y_pred == 1) & (y_test == 1))
    true_negatives = np.sum((y_pred == 0) & (y_test == 0))
    false_positives = np.sum((y_pred == 1) & (y_test == 0))
    false_negatives = np.sum((y_pred == 0) & (y_test == 1))

    # Print individual metrics
    tpr = true_positives / (true_positives + false_negatives)
    tnr = true_negatives / (true_negatives + false_positives)
    fpr = false_positives / (false_positives + true_negatives)
    fnr = false_negatives / (false_negatives + true_positives)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = 2 * precision * recall / (precision + recall)
    specificity = true_negatives / (true_negatives + false_positives)
    roc_auc = roc_auc_score(y_test, best_clf.predict_proba(X_test)[:, 1])

    print(accuracy, precision, recall, f1_score, specificity, roc_auc)

    # Add the statistics to the dataframe
    df_results.loc[len(df_results)] = [true_positives, true_negatives, false_positives, false_negatives, tpr, tnr, fpr, fnr, accuracy, precision, recall, f1_score, specificity, roc_auc]

# Print the average of each statistic over the 10 folds
print(df_results.mean())

# Print out the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Train the final model on the full Australian dataset
best_clf = xgb.XGBClassifier(**best_params, random_state=seed_value, objective='binary:logistic') #
best_clf.fit(X, y)

"""#RF"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, accuracy_score
import random

# Set random seeds
seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)

# Adjust this threshold as needed
threshold = 0.5

# Create a dataframe to store the results
df_results = pd.DataFrame(columns=['TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives',
                                   'TPR', 'TNR', 'FPR', 'FNR',
                                   'Accuracy', 'Precision', 'Recall', 'f1_score', 'Specificity', 'ROC-AUC'])
features = ['AGE', 'SEX', 'egfr_slope', 'egfr_mean']
X = df[features]
y = df['Kidney_Failure']

# Use Stratified KFold for cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_value)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


    clf  = RandomForestClassifier(max_depth=15, random_state=seed_value) #
    clf.fit(X_train, y_train)

    #print("Feature importances:", clf.feature_importances_)

    # Evaluate on the held-out test set
    y_pred = clf.predict(X_test)
    print("Final Accuracy:", accuracy_score(y_test, y_pred))
    print("Final Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Final ROC-AUC Score:")
    print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

    # Compute the statistics
    true_positives = np.sum((y_pred == 1) & (y_test == 1))
    true_negatives = np.sum((y_pred == 0) & (y_test == 0))
    false_positives = np.sum((y_pred == 1) & (y_test == 0))
    false_negatives = np.sum((y_pred == 0) & (y_test == 1))

    # Print individual metrics
    tpr = true_positives / (true_positives + false_negatives)
    tnr = true_negatives / (true_negatives + false_positives)
    fpr = false_positives / (false_positives + true_negatives)
    fnr = false_negatives / (false_negatives + true_positives)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = 2 * precision * recall / (precision + recall)
    specificity = true_negatives / (true_negatives + false_positives)
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    # Add the statistics to the dataframe
    df_results.loc[len(df_results)] = [true_positives, true_negatives, false_positives, false_negatives, tpr, tnr, fpr, fnr, accuracy, precision, recall, f1_score, specificity, roc_auc]

# Print the average of each statistic over the 5 folds
print(df_results.mean())

# Train the final model on the full Australian dataset
clf = RandomForestClassifier(max_depth=15, random_state=seed_value) #
clf.fit(X, y)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, accuracy_score
import random

# Set random seeds
seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)

# Adjust this threshold as needed
threshold = 0.5

# Create a dataframe to store the results
df_results = pd.DataFrame(columns=['TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives',
                                   'TPR', 'TNR', 'FPR', 'FNR',
                                   'Accuracy', 'Precision', 'Recall', 'f1_score', 'Specificity', 'ROC-AUC'])

features = ['AGE', 'SEX', 'egfr_slope', 'egfr_mean']

X = df[features]
y = df['Kidney_Failure']

# Initialize a Decision Tree model
clf = RandomForestClassifier(random_state=seed_value) #

param_grid = {
    'n_estimators': [30, 50],
    'max_depth': [5, 10],
    'min_samples_split': [4, 5],
    'min_samples_leaf': [2, 3],
    'max_features': ['auto']
}                                                     #

# Initialize GridSearch
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                           cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_value),
                           scoring='roc_auc', n_jobs=-1, verbose=1)

# Fit the GridSearch model
grid_search.fit(X, y)

# Extract the best estimator
best_clf = grid_search.best_estimator_

# Use Stratified KFold for cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_value)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train a Decision Tree model using the best estimator
    best_clf.fit(X_train, y_train)

    # Evaluate on the held-out test set
    y_pred = best_clf.predict(X_test)
    print("Final Accuracy:", accuracy_score(y_test, y_pred))
    print("Final Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Final ROC-AUC Score:")
    print(roc_auc_score(y_test, best_clf.predict_proba(X_test)[:, 1]))

    # Compute the statistics
    true_positives = np.sum((y_pred == 1) & (y_test == 1))
    true_negatives = np.sum((y_pred == 0) & (y_test == 0))
    false_positives = np.sum((y_pred == 1) & (y_test == 0))
    false_negatives = np.sum((y_pred == 0) & (y_test == 1))

    # Print individual metrics
    tpr = true_positives / (true_positives + false_negatives)
    tnr = true_negatives / (true_negatives + false_positives)
    fpr = false_positives / (false_positives + true_negatives)
    fnr = false_negatives / (false_negatives + true_positives)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = 2 * precision * recall / (precision + recall)
    specificity = true_negatives / (true_negatives + false_positives)
    roc_auc = roc_auc_score(y_test, best_clf.predict_proba(X_test)[:, 1])

    print(accuracy, precision, recall, f1_score, specificity, roc_auc)

    # Add the statistics to the dataframe
    df_results.loc[len(df_results)] = [true_positives, true_negatives, false_positives, false_negatives, tpr, tnr, fpr, fnr, accuracy, precision, recall, f1_score, specificity, roc_auc]

# Print the average of each statistic over the 10 folds
print(df_results.mean())

# Print out the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Train the final model on the full Australian dataset
best_clf = RandomForestClassifier(**best_params, random_state=seed_value) #
best_clf.fit(X, y)

"""#LightGB"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, accuracy_score
import random

# Set random seeds
seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)

# Adjust this threshold as needed
threshold = 0.5

# Create a dataframe to store the results
df_results = pd.DataFrame(columns=['TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives',
                                   'TPR', 'TNR', 'FPR', 'FNR',

                                   'Accuracy', 'Precision', 'Recall', 'f1_score', 'Specificity', 'ROC-AUC'])
features = ['AGE', 'SEX', 'egfr_slope', 'egfr_mean']
X = df[features]
y = df['Kidney_Failure']

# Use Stratified KFold for cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_value)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train a Decision Tree model
    clf  = lgb.LGBMClassifier(objective='binary', random_state=seed_value) #
    clf.fit(X_train, y_train)

    #print("Feature importances:", clf.feature_importances_)

    # Evaluate on the held-out test set
    y_pred = clf.predict(X_test)
    print("Final Accuracy:", accuracy_score(y_test, y_pred))
    print("Final Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Final ROC-AUC Score:")
    print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

    # Compute the statistics
    true_positives = np.sum((y_pred == 1) & (y_test == 1))
    true_negatives = np.sum((y_pred == 0) & (y_test == 0))
    false_positives = np.sum((y_pred == 1) & (y_test == 0))
    false_negatives = np.sum((y_pred == 0) & (y_test == 1))

    # Print individual metrics
    tpr = true_positives / (true_positives + false_negatives)
    tnr = true_negatives / (true_negatives + false_positives)
    fpr = false_positives / (false_positives + true_negatives)
    fnr = false_negatives / (false_negatives + true_positives)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = 2 * precision * recall / (precision + recall)
    specificity = true_negatives / (true_negatives + false_positives)
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    # Add the statistics to the dataframe
    df_results.loc[len(df_results)] = [true_positives, true_negatives, false_positives, false_negatives, tpr, tnr, fpr, fnr, accuracy, precision, recall, f1_score, specificity, roc_auc]

# Print the average of each statistic over the 5 folds
print(df_results.mean())

# Train the final model on the full Australian dataset
clf = lgb.LGBMClassifier(objective='binary', random_state=seed_value) #
clf.fit(X, y)

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, accuracy_score
import random

# Set random seeds
seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)

# Adjust this threshold as needed
threshold = 0.5

# Create a dataframe to store the results
df_results = pd.DataFrame(columns=['TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives',
                                   'TPR', 'TNR', 'FPR', 'FNR',
                                   'Accuracy', 'Precision', 'Recall', 'f1_score', 'Specificity', 'ROC-AUC'])

# Sample dataset features and target
features = ['AGE', 'SEX', 'egfr_slope', 'egfr_mean']
X = df[features]
y = df['Kidney_Failure']
# Initialize a Decision Tree model
clf = lgb.LGBMClassifier(objective='binary', random_state=seed_value) #

param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'num_leaves': [31, 50],
    'min_child_samples': [20, 30]
}                                                     #

# Initialize GridSearch
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                           cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_value),
                           scoring='roc_auc', n_jobs=-1, verbose=1)

# Fit the GridSearch model
grid_search.fit(X, y)

# Extract the best estimator
best_clf = grid_search.best_estimator_

# Use Stratified KFold for cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_value)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train a Decision Tree model using the best estimator
    best_clf.fit(X_train, y_train)

    # Evaluate on the held-out test set
    y_pred = best_clf.predict(X_test)
    print("Final Accuracy:", accuracy_score(y_test, y_pred))
    print("Final Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Final ROC-AUC Score:")
    print(roc_auc_score(y_test, best_clf.predict_proba(X_test)[:, 1]))

    # Compute the statistics
    true_positives = np.sum((y_pred == 1) & (y_test == 1))
    true_negatives = np.sum((y_pred == 0) & (y_test == 0))
    false_positives = np.sum((y_pred == 1) & (y_test == 0))
    false_negatives = np.sum((y_pred == 0) & (y_test == 1))

    # Print individual metrics
    tpr = true_positives / (true_positives + false_negatives)
    tnr = true_negatives / (true_negatives + false_positives)
    fpr = false_positives / (false_positives + true_negatives)
    fnr = false_negatives / (false_negatives + true_positives)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = 2 * precision * recall / (precision + recall)
    specificity = true_negatives / (true_negatives + false_positives)
    roc_auc = roc_auc_score(y_test, best_clf.predict_proba(X_test)[:, 1])

    print(accuracy, precision, recall, f1_score, specificity, roc_auc)

    # Add the statistics to the dataframe
    df_results.loc[len(df_results)] = [true_positives, true_negatives, false_positives, false_negatives, tpr, tnr, fpr, fnr, accuracy, precision, recall, f1_score, specificity, roc_auc]

# Print the average of each statistic over the 10 folds
print(df_results.mean())

# Print out the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Train the final model on the full Australian dataset
best_clf = lgb.LGBMClassifier(**best_params, random_state=seed_value, objective='binary') #
best_clf.fit(X, y)