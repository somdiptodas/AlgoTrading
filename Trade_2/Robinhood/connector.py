import pyotp
import robin_stocks.robinhood as r

WEBSITE_URL = 'https://robinhood.com/login'
USERNAME = "somdiptodas@gmail.com"
PASSWORD = "Som@470273"
MFA = "O3AISHCSRZQNEOAW"
totp  = pyotp.TOTP(MFA).now()
r.login(username = USERNAME, password = PASSWORD, mfa_code = totp)

print(r.get_markets())