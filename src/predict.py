import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import joblib
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingRegressor

# opening config.yaml
with open("config.yaml") as r:
    cfg = yaml.safe_load(r)

# adding data set selection for reproducability
model_selector = int(input("""Please select your data set:
                       1) train.csv
                       2) test.csv
                       => """))

# acknowledging paths
train_path = cfg["paths"]["train"]
test_path = cfg["paths"]["test"]
test_size = cfg["training"]["test_size"]

if model_selector == 1:
    chosen_path = train_path

elif model_selector == 2:
    chosen_path = test_path

else:
    raise ValueError("Invalid Choice")

# loading data and checkpoint
df = pd.read_csv(chosen_path)
print(f"Loaded test.csv with shape {df.shape}")

# sort date so time series protected
df = df.sort_values(cfg["schema"]["date_col"]).reset_index(drop=True)

# defining coulmuns to be dropped and defining target
drop_cols = ["market_forward_excess_returns", "risk_free_rate"]
if cfg["schema"]["date_col"] in df.columns:
    drop_cols.append(cfg["schema"]["date_col"])

X = df.select_dtypes(include=[np.number]).drop(
    columns=drop_cols, errors="ignore")

X = X.fillna(X.mean())

# loading trained model
model_name = cfg["training"]["model_name"]
model_path = f"models/{model_name}_model.joblib"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"model not found error file name: {model_path}")
model = joblib.load(model_path)
print(f"Loaded trained model: {model_path}")

# make actual predictions
if hasattr(model, "feature_names_in_"):
    X = X[model.feature_names_in_]

df["predicted_forward_returns"] = model.predict(X)

os.makedirs("outputs", exist_ok=True)
df[["date_id", "predicted_forward_returns"]].to_csv(
    "outputs/predictions.csv", index=False)
df[["date_id", "predicted_forward_returns"]].to_parquet(
    "outputs/predictions.parquet", index=False)

print("Predictions saved to: ")
print("outputs/predictions.csv")
print("outputs/predictions.parquet")
