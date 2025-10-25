from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

from .data import load_preds_clean
from .strategy import SizingConfig, preds2allocs

def submit_allocations(
    train_csv: str,
    test_csv: str,
    preds_train_parquet: str, # not necessary atm but might be good for debugs
    preds_test_parquet: str,
    out_path: str,
    cfg: SizingConfig,
) -> pd.DataFrame:
    """generate submission file for competition
    to do so:
        load files
        load model predictions
        estimate market vol from train set
        map predictions to allocations using our [0, 2] sizing rule
        write submsission .csv file with columns date_id, allocation

    notes:
        dont touch future returns. use only train history
        dont prescale it to hit the 1.2x vol target, 
        that scaling will be done in the evaluation
    """

    # load and sort data
    train = (
        pd.read_csv(train_csv)
        .sort_values("date_id")
        .reset_index(drop=True)
    )
    test = (
        pd.read_csv(test_csv)
        .sort_values("date_id")
        .reset_index(drop=True)
    )

    # load model predictions for test then merge with date_id
    preds_test = load_preds_clean(preds_test_parquet)
    test = test.merge(preds_test, on="date_id", how="left")

    # build volatility estimate 
    # feed pred2allocs as a long series that starts with historical 
    # market excess then continues with test market excess
    # for future unseen excess returns we just fill with 0.0 since we dont know them
    # hleps move EMWA forward deterministcally
    market_history = train["market_forward_excess_returns"].astype(float)

    # join train and test together helps ease transition for vol est
    market_for_vol = pd.concat(
        [
            market_history,
            pd.Series(
                np.zeros(len(test)),
                index=test.index,
                dtype=float,
            ),
            test["pred_excess"].astype(float),
        ],
        ignore_index=True,
    )

    # build matching pred series of same length
    # firs part is dummy zeros for train history
    # second part is test preds

    pred_full = pd.concat(
        [
            pd.Series(
                np.zeros(len(train)),
                dtype=float,
            ),
            test["pred_excess"].astype(float),
        ],
        ignore_index=True,
    )

    conf_full = None
    if "pred_conf" in test.columns:
        conf_full = pd.concat(
            [
            pd.Series(
                    np.zeros(len(train)),
                    dtype=float,
                ),
                test["pred_conf"].clip(0.0, 1.0),
            ],
            ignore_index=True,
        )

    # now run sizing rule on combined timeline
    mapped = preds2allocs(
        pred_excess=pred_full,
        market=market_for_vol.shift(1).fillna(0.0),
        conf=conf_full,
        config=cfg,
    )

    # only need allocations for test set now
    allocations_test = (
                                    mapped.iloc[len(train):]["alloc"]
                                    .clip(0.0, 2.0)
                                    .reset_index(drop=True)
                                    )
    
    # now build the final submission dataframe
    sub = (
        pd.DataFrame({
                "date_id": test["date_id"].astype(int),
                "allocation": allocations_test.round(decimals=6),
        })
        .sort_values("data_id")
        .sort_index(drop=True)
    )

    # now write to csv
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)

    return sub

        


