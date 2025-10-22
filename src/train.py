import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import joblib
import os
from sklearn.model_selection import TimeSeriesSplit

# loading the config.yaml file into the script
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

train_path = cfg["paths"]["train"]
target_col = cfg["schema"]["target_col"]
test_size = cfg["training"]["test_size"]
n_splits = cfg["training"]["n_splits"]

# loading data into pandas
df = pd.read_csv(train_path)
print(f"Loaded train.csv with shape {df.shape}")

# sorting it by date to ensure time series is protected
df = df.sort_values(cfg["schema"]["date_col"]).reset_index(drop=True)

# building features and defining a target
drop_cols = [target_col, "market_forward_excess_returns"]
if cfg["schema"]["date_col"] in df.columns:
    drop_cols.append(cfg["schema"]["date_col"])

X = df.select_dtypes(include=[np.number]).drop(
    columns=drop_cols, errors="ignore")
y = df[target_col].values

# handle missing values
X = X.fillna(X.mean())

# train test split
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=test_size, random_state=67)


# Cross_validation
kf = TimeSeriesSplit(n_splits=n_splits)

# initializing lists
cv_ic_list = []
cv_rmse_list = []

for train_idx, val_idx in kf.split(X_train):
    model = Ridge(alpha=cfg["models"]["ridge"]["alpha"])
    model.fit(X_train.iloc[train_idx], y_train[train_idx])
    preds = model.predict(X_train.iloc[val_idx])
    rmse = np.sqrt(mean_squared_error(y_train[val_idx], preds))
    cv_rmse_list.append(rmse)

    ic = np.corrcoef(y_train[val_idx], preds)[0, 1]
    if np.isnan(ic):  # guard against degenerate folds
        ic = 0.0
    cv_ic_list.append(ic)

cv_rmse = np.mean(cv_rmse_list)

cv_ic_mean = np.mean(cv_ic_list)
cv_ic_std = np.std(cv_ic_list)

print(f"Average CV RMSE: {cv_rmse:.6f}")
print(f"Average CV IC: {cv_ic_mean:.4f} ± {cv_ic_std:.4f}")

print("Feature columns:", X.columns.tolist())
