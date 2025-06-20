# modelling_tuning.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import dagshub

# Simpan token di environment variable
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/gimnastiarhrn/Membangun_Model.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'gimnastiarhrn'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '685ca98a3cd5b6ed208e7c7c361ad1f40590ff66'

dagshub.init(repo_owner='gimnastiarhrn',
             repo_name='Membangun_Model',
             mlflow=True)

import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("DagsHub - Tuned Laptop Price Prediction")



# ========== Fungsi Metrik Tambahan ==========
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def adjusted_r2(r2, n, k):
    return 1 - ((1 - r2) * (n - 1)) / (n - k - 1)

# ========== Load Data ==========
X_train = pd.read_csv("processed/X_train.csv")
X_test = pd.read_csv("processed/X_test.csv")
y_train = pd.read_csv("processed/y_train.csv")
y_test = pd.read_csv("processed/y_test.csv")

# ========== Hyperparameter Tuning ==========
params = {
    "alpha": [0.01, 0.1, 1, 10, 100]
}
ridge = Ridge()
grid = GridSearchCV(ridge, param_grid=params, scoring='r2', cv=5)
grid.fit(X_train, y_train.values.ravel())

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# ========== Evaluasi ==========
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test.values.ravel(), y_pred)
adj_r2 = adjusted_r2(r2, X_test.shape[0], X_test.shape[1])

# ========== MLflow Manual Logging ==========
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("Tuned Laptop Price Prediction")

with mlflow.start_run():
    mlflow.log_param("model", "Ridge")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("best_alpha", grid.best_params_['alpha'])

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("MAPE", mape)
    mlflow.log_metric("Adjusted_R2", adj_r2)

    # Simpan model
    os.makedirs("model", exist_ok=True)
    model_path = "model/ridge_model.pkl"
    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path)
    mlflow.sklearn.log_model(best_model, artifact_path="ridge-model")

print("✅ Model tuning & logging selesai.")

