import robin_stocks.robinhood as r
import robinConfig
import pyotp

totp  = pyotp.TOTP(robinConfig.MFA).now()

r.login(username = robinConfig.USERNAME, password = robinConfig.PASSWORD, mfa_code = totp)

def printOdata():
    optionData = r.find_options_by_expiration(['SPY'],
            expirationDate='2021-08-27',optionType='call')
    for item in optionData:
        print(item)
        #print(' price -',item['strike_price'],' exp - ',item['expiration_date'],' symbol - ',
        #    item['chain_symbol'],' delta - ',item['delta'],' theta - ',item['theta'])

r.order_option_spread()