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


# first check out jumpy the market is right now
# divide prediction by shakiness to get adjusted prediction
def ewma_volatility(preds: pd.Series, span: int, market: pd.Series) -> pd.Series:
    vol_est = market.ewma(arg=market,span=span)
    score = preds / vol_est

    # now use sigmoid function to squish everything to 0-1 and then scale to 0-2 
    # competion uses 2-0 range
    score = 2 / 



# pass prediction through squish function 
# smooth the prediction so it doesnt jump day after day
# limit how much it can change from day to day

# safety brake at the end
# if prediction is more than 1.2x market voltatily, scale it down
# maybe -> if portfolio loses too much money get out the game for a bit
 