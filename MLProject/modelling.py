import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.linear_model import LinearRegression
import dagshub

# ======================
# Argumen dari CLI
# ======================
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="processed", help="Path ke folder dataset")
args = parser.parse_args()

# ======================
# Ambil kredensial dari ENV (GitHub Actions Secrets)
# ======================
username = os.environ["MLFLOW_TRACKING_USERNAME"]
password = os.environ["MLFLOW_TRACKING_PASSWORD"]
uri = "https://dagshub.com/gimnastiarhrn/Membangun_Model.mlflow"

# ======================
# Inisialisasi DagsHub Tracking
# ======================
os.environ["MLFLOW_TRACKING_URI"] = uri
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = password

dagshub.init(
    repo_owner="gimnastiarhrn",
    repo_name="Membangun_Model",
    mlflow=True,
    token=os.environ["MLFLOW_TRACKING_PASSWORD"]
)

mlflow.set_tracking_uri(uri)
mlflow.set_experiment("DagsHub - Tuned Laptop Price Prediction")

# ======================
# Load Dataset
# ======================
X_train = pd.read_csv(os.path.join(args.data_dir, "X_train.csv"))
X_test = pd.read_csv(os.path.join(args.data_dir, "X_test.csv"))
y_train = pd.read_csv(os.path.join(args.data_dir, "y_train.csv"))
y_test = pd.read_csv(os.path.join(args.data_dir, "y_test.csv"))

# ======================
# Autolog & Training
# ======================
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train.values.ravel())

    y_pred = model.predict(X_test)

    # Simpan model ke file
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")

    print("✅ Model Linear Regression berhasil dilatih dan disimpan.")
