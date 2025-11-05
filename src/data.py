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
    col = []
    for col in df.columns:
        if col in LABEL_COLS or col in META_COLS:
            continue
        col.append(col)
    return col

def deterministic_impute(train: pd.Dataframe, test: pd.Dataframe, feature_cols: list[str]) -> pd.DataFrame:
    # get values to impute and fill with medians so its deterministic and using medians
    stats = {}
    train_impute = train.copy()
    test_impute = test.copy()
    for col in feature_cols:
        median = train_impute[col].median()
        if median == np.nan:
            median = 0.0
        train_impute[col] = train[col].fillna(median)
        test_impute[col] = test[col].fillna(median)
        stats[col] = median
    return train_impute, test_impute

def get_market_excess_returns(train: pd.Dataframe, feature_cols: list[str]) -> pd.Series:
    for col in feature_cols:
        # check if it exists
        if col == train.columns:
            return train[col].astype(float)
        raise TypeError("market excess returns column not found")
    
def load_preds_clean(path: str) -> pd.DataFrame:
    """ load model predictions from parquet and ensure correct columns """
    parquet_file = pd.read_parquet(path)

    # accept predicted_forward_returns instead of pred_excess
    if "pred_excess" not in parquet_file.columns:
        if "predicted_forward_returns" in parquet_file.columns:
            parquet_file = parquet_file.rename(columns={"predicted_forward_returns": "pred_excess"})
        else:
            raise TypeError(
                f"predictions file at {path} must contain either 'pred_excess' or 'predicted_forward_returns'"
            )

    expected_cols = {"date_id", "pred_excess"}

    # check required columns
    if not expected_cols.issubset(parquet_file.columns):
        raise TypeError(f"predictions file at {path} does not have the right columns")

    parquet_file.sort_values("date_id", inplace=True)
    parquet_file.reset_index(drop=True, inplace=True)

    if "pred_conf" in parquet_file.columns:
        parquet_file["pred_conf"] = parquet_file["pred_conf"].clip(lower=0.0, upper=1.0)
        expected_cols.add("pred_conf")

    df = parquet_file[list(expected_cols)]
    return df

