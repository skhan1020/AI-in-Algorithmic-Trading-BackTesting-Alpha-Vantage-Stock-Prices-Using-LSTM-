from flask import Flask, render_template, request
import numpy as np
import backtest_webapp
import sys
import stock_data_webapp

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template("main.html")

@app.route('/graphs', methods=['GET', 'POST'])
def graphs():


    args = request.form.to_dict('company')

    alpha_API = stock_data_webapp
    stock_df, SYMBOL = alpha_API.get_data(args['company'])

    short_window = int(args['short_window'])
    long_window = int(args['long_window'])
    initial_capital = float(args['initial_capital'])


    app_obj = backtest_webapp

    
    n = stock_df.shape[0]

    train_start = 0
    train_end = int(np.floor(0.8*n))
    test_start = train_end
    test_end = n


    graph1_url = app_obj.plot_stock_prices(stock_df, SYMBOL)
    graph2_url = app_obj.arima_plots(stock_df, SYMBOL, train_end, test_end)
    graph3_url, graph4_url, graph5_url, graph6_url = app_obj.lstm_plots(stock_df, SYMBOL, train_start, train_end, test_start, test_end, short_window, long_window, initial_capital)
 
    return render_template('graphs.html',
    graph1=graph1_url,
    graph2=graph2_url,
    graph3=graph3_url,
    graph4=graph4_url,
    graph5=graph5_url,
    graph6=graph6_url)
 
if __name__ == '__main__':
    app.debug = True
    app.run(threaded=False)

