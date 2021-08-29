try:
    import Trade.market as m
except:
    import market as m
import pandas as pd
import numpy as np

def entryStrategyRSI():
    rsi = m.get_rsi()
    print(rsi[-1])
    if rsi[-1] > 30:
        if rsi[-2] < 30:
            return True
    return False

def exitStrategyRSI():
    rsi = m.get_rsi()
    if rsi[-1] > 70:
        return True
    return False

