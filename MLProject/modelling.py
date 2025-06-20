# modelling.py

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
import joblib
import os

import dagshub

# Simpan token di environment variable
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/gimnastiarhrn/Membangun_Model.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'gimnastiarhrn'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '685ca98a3cd5b6ed208e7c7c361ad1f40590ff66'

dagshub.init(repo_owner='gimnastiarhrn',
             repo_name='Membangun_Model',
             mlflow=True)

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("DagsHub - Tuned Laptop Price Prediction")

# =======================
# Load dataset siap latih
# =======================
X_train = pd.read_csv("processed/X_train.csv")
X_test = pd.read_csv("processed/X_test.csv")
y_train = pd.read_csv("processed/y_train.csv")
y_test = pd.read_csv("processed/y_test.csv")

# =====================
# Setup MLflow Tracking
# =====================
#mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Ubah ke DagsHub URL jika sudah
#mlflow.set_experiment("Baseline Laptop Price Prediction")

# ============
# Autolog Mode
# ============
mlflow.sklearn.autolog()

# ===================
# Training dan Logging
# ===================
with mlflow.start_run():

    model = LinearRegression()
    model.fit(X_train, y_train.values.ravel())

    y_pred = model.predict(X_test)

    # Simpan model ke file .pkl
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")

    print("✅ Model Linear Regression berhasil dilatih dan disimpan.")
