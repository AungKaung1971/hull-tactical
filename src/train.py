import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import joblib
import os

# loading the config.yaml file into the script
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

train_path = cfg["paths"]["train"]
target_col = cfg["schema"]["target_col"]

# loading data into pandas
df = pd.read_csv(train_path)
print(f"Loaded train.csv with shape {df.shape}")

# building features and defining a target
drop_cols = [target_col]
if cfg["schema"]["date_col"] in df.columns:
    drop_cols.append(cfg["schema"]["date_col"])

X = df.select_dtypes(include=[np.number]).drop(
    columns=drop_cols, errors="ignore")
y = df[target_col].values
