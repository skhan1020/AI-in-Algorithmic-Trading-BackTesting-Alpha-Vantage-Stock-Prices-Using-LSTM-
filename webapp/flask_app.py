from flask import Flask, render_template
import numpy as np
import backtest_webapp
import sys
sys.path.insert(1, '../')
import stock_data

app = Flask(__name__)
 
@app.route('/graphs')
def graphs():

    alpha_API = stock_data
    stock_df, SYMBOL = alpha_API.get_data()

    app_obj = backtest_webapp

    
    n = stock_df.shape[0]

    train_start = 0
    train_end = int(np.floor(0.8*n))
    test_start = train_end
    test_end = n


    graph1_url = app_obj.plot_stock_prices(stock_df, SYMBOL)
    graph2_url = app_obj.arima_plots(stock_df, SYMBOL, train_end, test_end)
    graph3_url, graph4_url = app_obj.lstm_plots(stock_df, SYMBOL, train_start, train_end, test_start, test_end)
 
    return render_template('graphs.html',
    graph1=graph1_url,
    graph2=graph2_url,
    graph3=graph3_url,
    graph4=graph4_url)
 
if __name__ == '__main__':
    app.debug = True
    app.run(threaded=False)

