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

# first check out jumpy the market is right now
# divide prediction by shakiness to get adjusted prediction
def ewma_volatility(preds: pd.Series, span: int, market: pd.Series) -> pd.Series:
    mkt_squared = (market.fillna(0.0))**2
    var_ewma = mkt_squared.ewm(span=span, adjust=False, min_periods=span).mean()
    vol_mkt = np.sqrt(var_ewma)
    vol = vol_mkt.replace(0.0, np.nan).fillna(method="ffill").fillna(1e-6)

    return vol

# pass prediction through squish function 
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

# map predictions to allocations
def preds2allocs(pred_excess: pd.Series, market: pd.Series, 
                 conf: pd.Series | None, config: SizingConfig) -> pd.Series:
    """turn model predictions pred_excess into portfolio allocations
    between 0 and 2"""
    pred = pred_excess.astype(float).copy()
    vol_est = ewma_volatility(pred, config.vol_span, market) # get market vol estimate
    score = pred / vol_est # adjust prediction by market vol

    alloc_sig = 2.0 * sigmoid(config.k * score) # map to [0, 2] using sigmoid
    
    # if model provides confidence, scale allocation by it
    if config.use_conf and conf is not None:
        conf = conf.clip(0.0, 1.0).fillna(0.0) # ensure conf is between 0 and 1
        blend = config.lam + (1 - config.lam) * conf
        alloc_sig = blend * alloc_sig + (1 - blend) * 1.0 # blend using config lam

        alloc = alloc_sig.copy() # have non-smoothed alloc for later use
        prev = 1.0 # start from netural allocation so not risk averse but not risk seeking
        output = []
        for a in alloc_sig:
            # smooth allocation using ema and then clamp
            smoothed = config.alpha * a + (1 - config.alpha) * prev
            sm_clamped = np.clip(smoothed, prev - config.tau, prev + config.tau) # prevent big jumps
            sm_clipped = float(np.clip(sm_clamped, 0.0, 2.0)) # hard clip to [0, 2]
            output.append(sm_clipped)
            prev = sm_clipped
        # reallocate after smoothing
        alloc = pd.Series(output, index=pred.index)

        return pd.DataFrame({
            "alloc": alloc,
            "alloc_unsmoothed": alloc_sig,
            "pred_adjusted": score,
            "vol_est": vol_est,
        })



# smooth the prediction so it doesnt jump day after day
# limit how much it can change from day to day

# safety brake at the end
# if prediction is more than 1.2x market voltatily, scale it down
# maybe -> if portfolio loses too much money get out the game for a bit
 