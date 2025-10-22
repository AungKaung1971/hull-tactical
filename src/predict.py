import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

# --- Load config.yaml ---
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# --- Paths & schema ---
train_path = cfg["paths"]["train"]
test_path = cfg["paths"]["test"]
date_col = cfg["schema"]["date_col"]
target_col = cfg["schema"]["target_col"]
model_name = cfg["training"]["model_name"]

# --- Load data ---
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print(f"Loaded train.csv ({train_df.shape}) and test.csv ({test_df.shape})")

# --- Combine train + test for consistent feature creation ---
df_all = pd.concat([train_df, test_df], ignore_index=True)
df_all = df_all.sort_values(date_col).reset_index(drop=True)

# --- Build features (same logic as in train.py) ---
drop_cols = [target_col, "market_forward_excess_returns", "risk_free_rate"]
if date_col in df_all.columns:
    drop_cols.append(date_col)

X_all = df_all.select_dtypes(include=[np.number]).drop(
    columns=drop_cols, errors="ignore")
X_all = X_all.fillna(X_all.mean())

# --- Split out test features ---
X_test = X_all.iloc[len(train_df):].reset_index(drop=True)

# --- Load trained model or reinstantiate it ---
# Option A: if you've saved your trained model from train.py using joblib.dump()
# model = joblib.load(f"models/{model_name}.joblib")

# Option B: rebuild the same model using config params (since we haven't saved yet)
if model_name == "ridge":
    params = cfg["models"]["ridge"]
    model = Ridge(**params)
elif model_name == "hgbr":
    params = cfg["models"]["hgbr"]
    model = HistGradientBoostingRegressor(**params)
else:
    raise ValueError(f"Invalid model_name: {model_name}")

# --- Fit model on the full train data ---
# (we need this step since we didn’t save the model yet)
drop_cols = [target_col, "market_forward_excess_returns", "risk_free_rate"]
if date_col in train_df.columns:
    drop_cols.append(date_col)

X_train = train_df.select_dtypes(include=[np.number]).drop(
    columns=drop_cols, errors="ignore")
X_train = X_train.fillna(X_train.mean())
y_train = train_df[target_col].values

print(f"Training final {model_name} model on all train data...")
model.fit(X_train, y_train)

# --- Generate predictions on test data ---
print("Generating predictions on test.csv...")
test_preds = model.predict(X_test)
test_df["prediction"] = test_preds

# --- Save output for teammate ---
output = test_df[[date_col, "prediction"]]
os.makedirs("outputs", exist_ok=True)
output.to_csv("outputs/predictions.csv", index=False)
print("✅ Saved predictions.csv in outputs/ folder.")
print(output.head())
