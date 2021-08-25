import market as m
import strategy as s 
import RSConn as RSC 
import pandas as pd

# Order buy based off of true/false of entry strategy
def Order():
    if s.entryStrategyRSI:
       buy()
       print("purchasing SPY SPREADS \n")
       class save_data: 
            saved_price =     
            option_type = "Call"
            strike_price_one = 
            strike_price_two = 
            position_size = 
#price 
#position size
#strike Prices 1 & 2
#option type (call or put)
# open a position and catalog the info and same for close 
# CSV file or XL file using pandas
        
#figure out how to make this in a constant loop
while(True):
    Order()

# use multithreading to check for the need to buy in addition to checking for sell signals
# test over fifteen second intervals on the one minute