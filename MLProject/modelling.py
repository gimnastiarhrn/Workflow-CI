import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
import joblib
import dagshub

# Ambil credential dari environment
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Inisialisasi dagshub dengan tracking yang aktif
# Username dan token diatur di environment oleh GitHub Actions
# melalui secrets

dagshub.init(
    repo_owner="gimnastiarhrn",
    repo_name="Membangun_Model",
    mlflow=True
)

mlflow.set_experiment("DagsHub - Tuned Laptop Price Prediction")

# =======================
# Load dataset siap latih
# =======================
X_train = pd.read_csv("processed/X_train.csv")
X_test = pd.read_csv("processed/X_test.csv")
y_train = pd.read_csv("processed/y_train.csv")
y_test = pd.read_csv("processed/y_test.csv")

# =============
# Autolog Mode
# =============
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
