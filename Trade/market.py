"""
Gets and manipulates Financial Data from yahoo finance
"""
import io
from flask.helpers import safe_join
from flask.wrappers import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import time

picDirectiory = 'pictures\\graphs\\'

def getRate(ticker = "SPY"):
    """
    Returns ask and bid as a tuple (ask,bid)
    """
    stock = yf.Ticker(ticker).info
    ask = stock['ask']
    bid = stock['bid']
    return [['Ask', ask], ['Bid' , bid]]

def createChart(ticker = "SPY", period = "5d", interval = "15m"):

    """Gets Financial Data and creates a graph

    Parameters
    ----------
    ticker : str
        The ticker symbol to get data for (default SPY)
    period : str
        Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max Either Use period parameter or use start and end
    interval : str
        Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo Intraday data cannot extend last 60 days

    
    Returns
    -------
    renders the plot on the fly as a .png
    """

    stock = yf.Ticker(ticker)
    data = stock.history(period = period, interval = interval)
    name = interval + "_" + period + '.png'
    close = data['Close']
    val = close.tolist()
    fig = plt.figure()
    plt.plot(val,'#0288D1')
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")

def getOptions(ticker = 'SPY'):
    stock = yf.Ticker(ticker)
    options = stock.option_chain(stock.options[4])
    price = int(stock.history(period= "1d", interval= "15m")['Close'][-1])
    for i in range(len(options.calls)):
        if price == options.calls['strike'][i]:
            calls = options.calls[i-8 : i+9]
            return calls.values.tolist()
    return 

def movingAverage(price_df,period = 50, exponential = 1, weightF = 100):
    LenDf = len(price_df)-1
    if LenDf > period:
        PCalc = 0
        total_weight = 0
        for x in range(period):
            pos = LenDf-x
            weight = weightF/((x**exponential)+weightF)
            total_weight += weight
            PCalc += float(price_df[pos])*weight
        ema = PCalc/total_weight
        return round(ema,2)
    else:
        return '-'

def get_rsi(period = "5d", interval = "15m", lookback = 14):
    stock = yf.Ticker("SPY")
    data = stock.history(period = period, interval = interval)
    close = data['Close'].tolist()
    diff = np.subtract(close [1:], close[:-1])
    up = []
    down = []
    for i in range(len(diff)):
        if diff[i] < 0:
            up.append(0)
            down.append(diff[i])
        else:
            up.append(diff[i])
            down.append(0)
    up_series = pd.Series(up)
    down_series = pd.Series(down).abs()
    up_ewm = up_series.ewm(com = lookback - 1, adjust = False).mean()
    down_ewm = down_series.ewm(com = lookback - 1, adjust = False).mean()
    rs = up_ewm/down_ewm
    rsi = 100 - (100 / (1 + rs))
    rsi_df = rsi.to_list()
    #.rename(columns = {0:'rsi'}).set_index(close.index)
    return rsi_df[lookback:]


def format(expiry, strikePrice, optiontype, effect, action):
    '''
    Parameter
    --------
    Expiry : Str
        "YYYY-MM-DD"
    strikePrice : Str
    optiontype : Str
        "call" or "put"
    effect : Str
        "open" or "close"
    action
        "buy or "sell"

    Returns
    --------
    Dictionary\n
    Example:

    leg1 = {"expirationDate":"2019-12-20",
            "strike":"2.00",
            "optionType":"call",
            "effect":"open",
            "action":"buy"}

    '''

    leg = {"expirationDate" : expiry,
            "strike" : strikePrice,
            "optionType" : optiontype,
            "effect" : effect,
            "action" : action}

    return leg

def getOptionDates(ticker = "SPY"):
    stock = yf.Ticker(ticker)
    return stock.options