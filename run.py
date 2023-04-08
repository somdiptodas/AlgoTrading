from matplotlib.pyplot import thetagrids
import Trade.controller as tc
import Web.main as w
import threading

def startAlgo(ticker = "SPY", quantity = 1):
    algo = tc.algo(ticker = ticker, quantity = quantity)
    algo.run()

def startWeb():
    w.app.run(debug = True)

#t1 = threading.Thread(target = startAlgo)
#t1.start()

startWeb()
t2 = threading.Thread(target = startWeb)
t2.start()