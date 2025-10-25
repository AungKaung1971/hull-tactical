# this file tells us how good our trading strategy is
# using returns, voltatility, sharpe ratio, drawdown, turnover

from __future__ import annotations
import pandas as pd
import numpy as np

ann = 252 # trading days in a year S&P 500

# returns annualized return
def ann_return(daily: pd.Series) -> float:
    return float(daily.mean() * ann)

# returns annualized volatility
def ann_volatility(daily: pd.Series) -> float:
    return float(daily.std(ddof=0) * np.sqrt(ann))

# returns sharpe ratio
def sharpe_ratio(daily: pd.Series) -> float:
    vol = ann_volatility(daily)
    if vol == 0.0:
        return 0.0
    return float(ann_return(daily) / vol)

# returns the worst drawdown experienced
def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())

# how much we trade on average 
# more turnover -> more trading -> more chaos
def turnover(alloc: pd.Series):
    # dont care about direction of diff so just abs then get mean
    return float(alloc.diff().abs().mean()) 