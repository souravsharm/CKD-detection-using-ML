**CKD Prognosis using AI**

This repository contains a project focused on predicting Chronic Kidney Disease (CKD) and kidney failure using Artificial Intelligence (AI) techniques. The project involves two main components:

### 1. CKD Prediction

The goal of this component is to detect CKD based on clinical features. The raw dataset used for this task is the UCI Kidney Disease dataset, available at [https://www.kaggle.com/datasets/mansoordaku/ckdisease](https://www.kaggle.com/datasets/mansoordaku/ckdisease). The dataset is pre-processed and the best features are selected for model training. The code for the CKD prediction model is provided in `ckd_prediction.py`. The performance of the model using the top features is evaluated and presented in `Model_trained_top_features_UCI.py`.

### 2. Kidney Failure Prediction

This component aims to predict kidney failure for CKD stage 3 and above patients using temporal data of eGFR. The original dataset can be obtained from [https://datadryad.org/stash/dataset/doi:10.5061/dryad.kq23s](https://datadryad.org/stash/dataset/doi:10.5061/dryad.kq23s). After pre-processing and applying the inclusion criterion, the pre-processed dataset is provided in `kidney_failure_data_preprocessed.csv`. The final dataset used for model training is `final_kidney_failure_prediction_data_for_model_training.csv`. The code for the kidney failure prediction model is available in `Kidney_failure_prediction.py`.

### Files

- `UCI_kidney_disease.csv`: Raw dataset for CKD prediction.
- `ckd_prediction.py`: Code for CKD prediction model.
- `Model_trained_top_features_UCI.py`: Code for evaluating the performance of the CKD prediction model using top features.
- `kidney_failure_data_preprocessed.csv`: Pre-processed dataset for kidney failure prediction.
- `final_kidney_failure_prediction_data_for_model_training.csv`: Final dataset used for kidney failure prediction model training.
- `Kidney_failure_prediction.py`: Code for kidney failure prediction model.

### Usage

To replicate the results, follow these steps:

1. Pre-process the raw datasets using the provided code.
2. Train the CKD prediction model using `ckd_prediction.py`.
3. Evaluate the performance of the CKD prediction model using `Model_trained_top_features_UCI.py`.
4. Train the kidney failure prediction model using `Kidney_failure_prediction.py`.

### Contributions

This project contributes to the development of AI-based models for CKD prognosis, providing insights into the clinical features and temporal data that are most relevant for predicting CKD and kidney failure.
