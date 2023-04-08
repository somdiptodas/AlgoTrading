from flask import Flask, jsonify, render_template, redirect, Response
import market
app = Flask(__name__)

@app.route('/', methods=("POST", "GET"))
def home():
    return render_template('plot.html', rate = market.getRate(), options = market.getOptions())

@app.route("/month.png")
def plotMonth():
    """ renders the plot on the fly as a .svg"""
    return market.createChart(period = '1mo', interval = '1h')

@app.route("/week.png")
def plotWeek():
    """ renders the plot on the fly as a .svg"""
    return market.createChart()

@app.route("/day.png")
def plotDay():
    """ renders the plot on the fly as a .svg"""
    return market.createChart(period = '1d', interval = '2m')


if __name__ == '__main__':
    app.run(debug = True)