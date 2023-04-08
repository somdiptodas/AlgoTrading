import robin_stocks.robinhood as r
try:
    import Trade.rConfig as rc
    import Trade.market as m
    import Trade.data as d
except:
    import rConfig as rc
    import market as m
    import data as d
import time
import pyotp

totp  = pyotp.TOTP(rc.MFA).now()

r.login(username = rc.USERNAME, password = rc.PASSWORD, mfa_code = totp)

def closeCallSpread(expiry, strPrice, limitPrice, ticker = "SPY", quantity = 1):
    """
    Closes a call debit spread

    returns
    ---------
    Dictionary that contains information regarding the trading of options, 
    such as the order id, the state of order (queued, confired, filled, failed, canceled, etc.), 
    the price, and the quantity.
    """
    OPTIONTYPE = "call"
    EFFECT = "close"
    limitPrice = round(float(limitPrice*1.3), 2)

    leg1 = m.format(expiry, str(strPrice + 3), OPTIONTYPE, EFFECT, action = "sell")
    leg2 = m.format(expiry, str(strPrice + 4), OPTIONTYPE, EFFECT, action = "buy")
    spread = [leg1,leg2]

    return r.order_option_spread("credit", limitPrice, ticker, quantity, spread), limitPrice, spread



def openCallSpread(ticker = "SPY", quantity = 1):
    """
    Opens a call debit spread

    returns
    ---------
    Dictionary that contains information regarding the trading of options, 
    such as the order id, the state of order (queued, confired, filled, failed, canceled, etc.), 
    the price, and the quantity.
    """
    OPTIONTYPE = "call"
    EFFECT = "open"
    expiry = m.getOptionDates()[4]
    strPrice = int(float(r.get_stock_quote_by_symbol(ticker)['last_trade_price']))
    
    buyLeg = r.get_option_market_data(ticker, expiry, str(strPrice + 3), OPTIONTYPE)[0][0]
    sellLeg = r.get_option_market_data(ticker, expiry, str(strPrice + 4), OPTIONTYPE)[0][0]
    limitPrice = round(float(buyLeg['adjusted_mark_price']) - float(sellLeg['adjusted_mark_price']) + .01, 2)

    leg1 = m.format(expiry, str(strPrice + 3), OPTIONTYPE, EFFECT, action = "buy")
    leg2 = m.format(expiry, str(strPrice + 4), OPTIONTYPE, EFFECT, action = "sell")
    spread = [leg1,leg2]

    print(limitPrice) 
    print(spread)

    receipt = r.order_option_spread("debit", limitPrice, ticker, quantity, spread)
    print('Order sent')

    for i in range(3):
        time.sleep(10)
        CCS = closeCallSpread(expiry, strPrice, limitPrice, ticker = ticker, quantity = quantity)
        if 'detail' in CCS[0]:
            print('Previous order not executed! -- Try #'+ str(i+1))
        else:
            print(CCS[1])
            print(CCS[2])
            print("Order executed. Take profits set")
            d.saveData([limitPrice,spread,CCS[1],CCS[2]])
            return True

    r.cancel_option_order(receipt['id'])
    print("Order cancelled")
    return False

#print(openCallSpread())

#print(r.cancel_all_option_orders())

