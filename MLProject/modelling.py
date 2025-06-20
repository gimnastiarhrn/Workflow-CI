import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
import joblib
import dagshub

# Ambil kredensial dari environment (diatur dari GitHub Actions secrets)
username = os.environ.get("MLFLOW_TRACKING_USERNAME")
password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
uri = "https://dagshub.com/gimnastiarhrn/Membangun_Model.mlflow"

# Konfigurasi koneksi ke DagsHub
os.environ["MLFLOW_TRACKING_URI"] = uri
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = password

dagshub.init(
    repo_owner="gimnastiarhrn",
    repo_name="Membangun_Model",
    mlflow=True
)

mlflow.set_tracking_uri(uri)

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
