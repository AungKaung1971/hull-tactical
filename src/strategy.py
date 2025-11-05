# inptus per day
# pred[t] = model predicted exess returns for day t 
# market[t] = market actual excess returns for day t (used to get the shakiness of market)

# span = how many days to look back to get shakiness
# alpha = smoothing factor
# tau = max change per day
# k = how much to scale up or down from 0 to 2
# use_conf = whether to use model confidence or not
# lambda = gives us lowest bet floor so even if confidence is super low we still do smth

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class SizingConfig:
    k: float = 6.0 # sigmoid 1/temperature (how aggresive you bet)
    vol_span: int = 21 # how quickly to adapt to new market volatility
    alpha: float = 0.5 # how much you smooth predictions
    tau: float = 0.25 # clamp from day to day 
    lam: float = 0.5 # whether you trust model or not
    use_conf: bool = True # whether to use confidence scaling or not
    use_dd_brake: bool = False # risk management brake 
    dd_window: int = 63 # lookback window for measuring drawdown
    dd_trigger: float = 0.10 # what constitutes a drawdown
    dd_horizon: int = 10 # how long to apply brake for
    dd_reduction: float = 0.5 # how much to reduce bet by during drawdown

# pass prediction through squish function 
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

# first check out jumpy the market is right now
# divide prediction by shakiness to get adjusted prediction
def ewma_volatility(preds: pd.Series, span: int, market: pd.Series) -> pd.Series:
    mkt_squared = (market.fillna(0.0))**2
    var_ewma = mkt_squared.ewm(span=span, adjust=False, min_periods=span).mean()
    vol_mkt = np.sqrt(var_ewma)
    vol = vol_mkt.replace(0.0, np.nan).ffill().fillna(1e-6)  # deprecation-safe
    return vol

def preds2allocs(pred_excess: pd.Series,
                 market: pd.Series,
                 conf: pd.Series | None,
                 config: SizingConfig) -> pd.DataFrame:
    """Turn model predictions into allocations in [0, 2]."""
    pred = pred_excess.astype(float).copy()

    # normalize by market volatility
    vol_est = ewma_volatility(pred, config.vol_span, market)
    score = pred / vol_est

    # base sigmoid mapping to [0, 2]
    alloc_sig = 2.0 * sigmoid(config.k * score)

    # optional confidence blend
    if config.use_conf and conf is not None:
        conf = conf.clip(0.0, 1.0).fillna(0.0)
        blend = config.lam + (1.0 - config.lam) * conf
        alloc_sig = blend * alloc_sig + (1.0 - blend) * 1.0  # blend toward neutral 1.0

    # smooth and clamp changes day-to-day
    prev = 1.0
    out_vals = []
    for a in alloc_sig:
        smoothed = config.alpha * a + (1.0 - config.alpha) * prev
        sm_clamped = np.clip(smoothed, prev - config.tau, prev + config.tau)
        sm_clipped = float(np.clip(sm_clamped, 0.0, 2.0))
        out_vals.append(sm_clipped)
        prev = sm_clipped

    alloc = pd.Series(out_vals, index=pred.index)

    return pd.DataFrame({
        "alloc": alloc,
        "alloc_unsmoothed": alloc_sig,
        "pred_adjusted": score,
        "vol_est": vol_est,
    })


def apply_drawdown_brake(equity: pd.Series, alloc: pd.Series, config: SizingConfig) -> pd.Series:
    """ apply drawdown brake to allocations if equity exceeds threshold
    then hold drawn for a certain horizon"""
    if not config.use_dd_brake:
        return alloc
    
    alloc_adj = alloc.copy()
    peak = equity.cummax()
    drawdown = (peak / equity) - 1.0

    brake_active = drawdown.rolling(config.dd_window).min() <= -config.dd_trigger

    active = 0
    alloc_adj_vals = alloc_adj.to_numpy().copy()
    for i, flag in enumerate(brake_active.values()):
        if flag:
            active = config.dd_horizon
        if active > 0:
            alloc_adj_vals[i] *= (1.0 - config.dd_reduction)
            active -= 1

    return pd.Series(alloc_adj_vals, index=alloc.index).clip(0.0, 2.0)

def cap_volatility(strategy_excess: pd.Series, market_excess: pd.Series, vol_cap_ratio: float = 1.2) -> float:
    """ if the strategy excess returns higher than vol_cap_ratio then we get penalized greatly
    therefore we have to minimize this risk by capping it
    you minimize the whole allocation by a constant multiplier"""
    # tradings days of the S&P 500
    trade_days = 252

    # annualize voltatilities
    port_vol = strategy_excess.std(ddof=0) * np.sqrt(trade_days)
    market_vol = market_excess.std(ddof=0) * np.sqrt(trade_days)

    # if market vol is 0, then scaling would make no sense -> dont do it
    if market_vol == 0.0:
        return 1.0
    
    max_allowed_vol = vol_cap_ratio * market_vol # set up upper limit
    # now check if capping is necessary
    if port_vol <= max_allowed_vol or port_vol == 0.0:
        return 1.0
    
    # calculate scale down factor
    scale_down = max_allowed_vol / port_vol
    return float(np.clip(scale_down, 0.0, 1.0))
 