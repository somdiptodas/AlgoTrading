import robin_stocks.robinhood as r
import rConfig as rc
import pyotp
import market as m

totp  = pyotp.TOTP(rc.MFA).now()

r.login(username = rc.USERNAME, password = rc.PASSWORD, mfa_code = totp)

def printOdata():
    optionData = r.find_options_by_expiration(['SPY'],
            expirationDate='2021-08-27',optionType='call')
    for item in optionData:
        print(item)
        #print(' price -',item['strike_price'],' exp - ',item['expiration_date'],' symbol - ',
        #    item['chain_symbol'],' delta - ',item['delta'],' theta - ',item['theta'])


def orderCallDebitSpread(ticker = "SPY"):
    """
    Orders a call debit spread

    Parameter
    ---------
    ticker : Str (optional)

    returns
    ---------
    Dictionary that contains information regarding the trading of options, 
    such as the order id, the state of order (queued, confired, filled, failed, canceled, etc.), 
    the price, and the quantity.
    """
    options = m.getOptions()
    buyLeg = options[11]
    sellLeg = options[12]
    limitPrice = round(buyLeg[3] - sellLeg[3] + .01, 2)
    leg1 = m.format(buyLeg, optiontype = "call", effect = "open", action = "buy")
    leg2 = m.format(sellLeg, optiontype = "call", effect = "open", action = "sell")
    spread = [leg1,leg2]

    order = r.order_option_spread("debit", limitPrice, ticker, 1, spread)
    print(limitPrice)

orderCallDebitSpread()