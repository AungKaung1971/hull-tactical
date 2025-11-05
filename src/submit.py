from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

from .data import load_preds_clean
from .strategy import SizingConfig, preds2allocs

def submit_allocations(
    train_csv: str,
    test_csv: str,
    preds_test_parquet: str,
    out_path: str,
    cfg: SizingConfig,
) -> pd.DataFrame:
    """
    Generate submission file:
      - load train and test
      - load model predictions for test
      - estimate market vol from train history only (no lookahead)
      - map predictions to allocations in [0, 2]
      - write CSV with columns: date_id, allocation
    Notes:
      - do not scale to the 1.2x vol target here; evaluation handles that
    """

    # load and sort data
    train = pd.read_csv(train_csv).sort_values("date_id").reset_index(drop=True)
    test = pd.read_csv(test_csv).sort_values("date_id").reset_index(drop=True)

    # load model predictions for test
    preds_test = load_preds_clean(preds_test_parquet)

    # ensure columns exist pre merge
    if "pred_excess" not in preds_test.columns and "forward_predicted_returns" in preds_test.columns:
        preds_test = preds_test.rename(columns={"forward_predicted_returns": "pred_excess"})

    # align types on join key
    test["date_id"] = test["date_id"].astype(int)
    preds_test["date_id"] = preds_test["date_id"].astype(int)

    # keep only what we need from preds to avoid suffix weird stuff
    preds_test = preds_test[["date_id", "pred_excess"] + (["pred_conf"] if "pred_conf" in preds_test.columns else [])]

    # now merge
    test = test.merge(preds_test, on="date_id", how="left")

    # little sanity check
    if "pred_excess" not in test.columns:
        raise ValueError("pred_excess missing after merge; check predictions file columns and join key")



    # market history for volatility estimate (train only)
    market_history = train["market_forward_excess_returns"].astype(float)

    # build market series for sizing rule timeline:
    #   train market history followed by zeros for test length (unknown future)
    market_for_vol = pd.concat(
        [
            market_history,
            pd.Series(np.zeros(len(test)), index=test.index, dtype=float),
        ],
        ignore_index=True,
    )

    # build predictions series aligned to same timeline:
    #   zeros over train length, then test predictions
    pred_full = pd.concat(
        [
            pd.Series(np.zeros(len(train)), dtype=float),
            test["pred_excess"].astype(float),
        ],
        ignore_index=True,
    )

    conf_full = None
    if "pred_conf" in test.columns:
        conf_full = pd.concat(
            [
                pd.Series(np.zeros(len(train)), dtype=float),
                test["pred_conf"].astype(float).clip(0.0, 1.0),
            ],
            ignore_index=True,
        )

    # map predictions to allocations
    mapped = preds2allocs(
        pred_excess=pred_full,
        market=market_for_vol.shift(1).fillna(0.0),  # no lookahead
        conf=conf_full,
        config=cfg,
    )

    # take only the test period allocations
    allocations_test = (
        mapped.iloc[len(train):]["alloc"].clip(0.0, 2.0).reset_index(drop=True)
    )

    # build final submission
    sub = (
        pd.DataFrame({
            "date_id": test["date_id"].astype(int),
            "allocation": allocations_test.round(6),
        })
        .sort_values("date_id")
        .reset_index(drop=True)
    )

    # write
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    return sub
