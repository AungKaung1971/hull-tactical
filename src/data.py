from __future__ import annotations
import pandas as pd
import numpy as np

FEATURE_FAMILIES = ["M", "E", "I", "P", "V", "S", "MOM", "D"]

LABEL_COLS = {
    "forward_returns",
    "lagged_risk_free_rate",
    "lagged_market_forward_excess_returns",
}

LAGGED_AUX_COLS = {
    "lagged_forward_returns",
    "lagged_risk_free_rate",
    "lagged_market_forward_excess_returns",
}

META_COLS = {"date_id", "is_scored"}

def load_csv_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    for df in (train, test):
        assert "date_id" in df.columns, "date_id column is missing lol"
        df.sort_values("date_id", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return train, test  

def feature_columns(df: pd.DataFrame) -> list[str]:
    # cut down columns which dont have useful info
    cols = []
    for col in df.columns:
        if col in LABEL_COLS or col in META_COLS:
            continue
        cols.append(col)
    return cols

