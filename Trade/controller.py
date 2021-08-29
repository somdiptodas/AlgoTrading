try:
    import Trade.strategy as s 
    import Trade.RSConn as rsc
except:
    import strategy as s 
    import RSConn as rsc

import pandas as pd
from datetime import datetime
import time

# Order buy based off of true/false of entry strategy
#position size
#strike Prices 1 & 2
#option type (call or put)
# open a position and catalog the info and same for close 
# CSV file or XL file using pandas
        
#figure out how to make this in a constant loop

# use multithreading to check for the need to buy in addition to checking for sell signals
# test over fifteen second intervals on the one minute

class algo():
    def __init__(self, ticker = "SPY", quantity = 1):
        self.ticker = ticker
        self.quantity = quantity

    def run(self):
        orderEntered = False
        while(True):
            if orderEntered == False:
                print('checking rsi     ' + str(datetime.now()))
                if s.entryStrategyRSI():
                    orderEntered = rsc.openCallSpread(algo.ticker, algo.quantity)
                time.sleep(60)
            else:
                time.sleep(1800)
                orderEntered =  False

a = algo()
a.run()