# see how strategy would perform on historical data, using no look ahead

from __future__ import annotations
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

from .data import load_csv_data, load_preds_clean
from .strategy import (
    preds2allocs, 
    SizingConfig, 
    apply_drawdown_brake, 
    cap_volatility,
)

from .eval_metrics import (
    ann_return,
    ann_volatility,
    sharpe_ratio,
    max_drawdown,
    turnover,
)

def expanding_forward(
        train_csv: str,
        preds_parquet: str,
        out_dir: str,
        cfg: SizingConfig,
        start_frac: float = 0.6, # ignore first north of half the data for backtest -> poor info
        vol_cap_ratio: float = 1.2,
) -> pd.DataFrame:
    """ run a backtest to simulate daily allocations. using only past data (no lookahead)
    
    args:
        train_csv: path to training data csv
        preds_parquet: path to model predictions from Aung Kaung
        out_dir: directory to save backtest outputs
        cfg: sizing config for strategy such as sigmoid k, smoothing alpha etc...
        start_frac: fraction of data to skip at start for backtest
        vol_cap_ratio: ratio of strategy vol to market vol to cap at
    """

    # load training data
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv)
    train.sort_values("date_id", inplace=True).reset_index(drop=True, inplace=True)
    mkt_excess = train["market_forward_excess_returns"].astype(float)

    # load model predictions
    if preds_parquet is not None:
        preds_df = load_preds_clean(preds_parquet)
        df = pd.merge(train[["date_id"]], preds, on="date_id", how="left")
        pred = df["pred_excess"].astype(float)
        conf = df["pred_conf"] if "pred_conf" in df.columns else None
    else:
        print("dont know what to do here yet lol")

    n = len(train)
    start_idx = int(n * start_frac)

    # map predictions and allocations
    # use emwa volatility estimate from market in past as normalizer
    mapped = preds2allocs(pred, mkt_excess.shift(1).fillna(0.0),
                           conf, cfg) # HAVE TO SHIFT cant have leakage, future
    alloc_series = mapped["alloc"].copy()
    alloc_series.iloc[:start_idx] = 1.0 # start after the initial frac 

    # run day to day simulation
    records = []
    equity = 1.0
    equity_path = []

    for i in range(n):
        alloc = float(alloc_series.iloc[i]) # get allocation for day i
        mkt_excess = float(mkt_excess.iloc[i]) # get market excess for day i
        strat_excess = alloc * mkt_excess # strategy excess return
        equity *= (1.0 + strat_excess) # update equity
        equity_path.append(equity)
        records.append({
            "date_id": int(train["date_id"].iloc[i]),
            "market_excess": r,
            "pred_excess": float(pred.iloc[i]),
            "pred_conf": float(conf.iloc[i]) if conf is not None else np.nan,
            "alloc": alloc,
            "strategy_excess": strat_excess,
        })
    daily = pd.Dataframe.from_records(records)
    daily["equity"] = equity_path

    # apply drawdown brake
    if cfg.use_dd_brake:
        alloc_brake = apply_drawdown_brake(daily["equity"], daily["alloc"], cfg)
        daily["alloc_brake"] = alloc_brake
        equity = 1.0
        equity_path = []
        for alloc, mkt_excess in zip(daily["alloc_brake"], daily["market_excess"]):
            strat_excess = alloc * mkt_excess
            equity *= (1.0 + strat_excess)
            equity_path.append(equity)
        daily["strategy_excess"] = daily["alloc"] * daily["market_excess"]
        daily["equity"] = equity_path

    # apply volatility cap
    scale = cap_volatility(daily["strategy_excess"], daily["market_excess"], vol_cap_ratio)
    daily["alloc_scaled"] = (daily["alloc"] * scale).clip(0.0, 2.0)
    daily["strategy_excess_scaled"] = daily["alloc_scaled"] * daily["market_excess"]

    # now make equity again after capping
    equity_scaled = []
    equity = 1.0
    for strat_excess in daily["strategy_excess"].values:
        equity *= (1.0 + strat_excess)
        equity_scaled.append(equity)
    daily["equity_scaled"] = equity_scaled

    # get all the metrics
    metrics = {
        "ann_return": ann_return(daily["strategy_excess_scaled"]),
        "ann_volatility": ann_volatility(daily["strategy_excess_scaled"]),
        "mkt_ann_volatility": ann_volatility(daily["market_excess"]),
        "vol_ratio": ann_volatility(daily["strategy_excess_scaled"]) / ann_volatility(daily["market_excess"]),
        "sharpe_ratio": sharpe_ratio(daily["strategy_excess_scaled"]),
        "max_drawdown": max_drawdown(daily["equity_scaled"]),
        "turnover": turnover(daily["alloc_scaled"]),
        "scale_applied": float(scale),
        "n_days": int(len(daily)),
    }

    # save metrics to json
    out = Path(out_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    daily.to_csv(out / "backtest_daily.csv", index=False)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # now make plots for it uinsg matplotlib (best one)

    # equity curve: strategy vs market
    plt.figure()
    plt.plot(daily["date_id"], daily["equity_scaled"], label="Strategy Equity")
    mkt_eq = (1.0 + daily["market_excess"]).cumprod() # year cumulative growth
    plt.plot(daily["date_id"], mkt_eq, label="Market Equity")
    plt.title("Equity Curve")
    plt.xlabel("Date ID")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "plots" / "equity_curve.png")
    plt.close()

    # rolling 63-day volatility a quarter
    plt.figure()
    roll = daily["strategy_excess_scaled"].rolling(window=63)
    roll_sharpe = (roll.mean() * 252) / (roll.std(ddof=0) * np.sqrt(252))
    plt.plot(daily["date_id"], roll_sharpe)
    plt.title("Rolling 63-Day Sharpe Ratio")
    plt.xlabel("Date ID")
    plt.ylabel("Sharpe Ratio")
    plt.tight_layout()
    plt.savefig(out / "plots" / "rolling_sharpe.png")
    plt.close()

    # allocation path what did we allocate each day
    plt.figure()
    plt.plot(daily["date_id"], daily["alloc_scaled"])
    plt.title("Daily Allocation")
    plt.xlabel("Date ID")
    plt.ylabel("Allocation")
    plt.ylim(0.0, 2.0)
    plt.tight_layout()
    plt.savefig(out / "plots" / "daily_allocation.png")
    plt.close()

    return daily
