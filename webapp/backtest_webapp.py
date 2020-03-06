import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import io
import base64
import sys
from tradingstrategy_webapp import MovingAverageCrossStrategy as cs
from tradingstrategy_webapp import MarketOnClosePortfolio as pf
from models import arima_model as am
from models import lstm_model as lm


def create_io_obj():
    
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    graph_url = base64.b64encode(bytes_image.getvalue()).decode()
    plt.close()

    return 'data:image/png;base64,{}'.format(graph_url)

def plot_stock_prices(stock_price_data, company):


    # Plot of Company Stock Prices

    plt.figure(figsize=(8,6))
    stock_price_data[company].plot(label = company)
    plt.legend()
    plt.ylabel('Price in $')
    plt.title(company + ' stock prices obtained from Alpha Vantage API')

    return create_io_obj()
    

def arima_plots(data, symbol, train_len, test_len):
    

    arm = am(data, symbol, train_len)
    predictions, mse_arima = arm.evaluate()

    arima_predict = pd.DataFrame(predictions, index = data[train_len:test_len].index, columns = [symbol])

    # Plot ARIMA predictions

    plt.figure(figsize=(8,6))
    plt.plot(data[:train_len], color = 'b', label = 'Training Data')
    plt.plot(data[train_len:], color = 'g', label = 'Test Data')
    plt.plot(arima_predict, color='orange', label = 'predicted data using ARIMA')
    plt.legend()
    plt.ylabel('Price in $')
    plt.title('ARIMA model predictions for '+ symbol)
   
    return create_io_obj()


def lstm_plots(data, symbol, train_start, train_end, test_start, test_end, short_window, long_window, initial_capital):

    #####   Implementing the LSTM Model to Predict Future Prices   #####

    lst = lm(symbol, data, train_start, train_end, test_start, test_end)
    lstm_predictions, dev_set, look_back, train_loss, test_loss, mse_lstm = lst.evaluate()

    # Plot of Train, Test Set Losses - Check for Overfitting/Underfitting

    plt.figure(figsize=(8,6))
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.legend()
    plt.title('Train and Test Set Loss vs Epochs')
    # plt.show()

    bytes_image1 = create_io_obj()

    # Plot of Training, Testing and Predicted Stock Prices

    plt.figure(figsize=(8,6))
    plt.plot(data[:train_end+look_back+1], color = 'b', label = 'Training Data')
    plt.plot(dev_set, color = 'g', label = 'Test Data')
    plt.plot(lstm_predictions, color = 'orange', label = 'Predicted Data')
    plt.ylabel('Close price in $')
    plt.title('Applying LSTM on ' + symbol + ' stock prices obtained from Alpha Vantage API')
    plt.legend()
    # plt.show()

    bytes_image2 = create_io_obj()

    mac = cs(symbol, lstm_predictions, short_window, long_window)
    signals = mac.generate_signals()
    
    # Plot of Test Prices and Predicted Prices along with Markers for Trading
    # Strategy

    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111, ylabel='Price in $')
    dev_set[symbol].plot(ax=ax1, color = 'k', lw=2, label='actual data')
    lstm_predictions[symbol].plot(ax=ax1, color = 'm', lw=2, label='predicted data')
    signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2)
    plt.legend()

    #buy signal
    ax1.plot(signals.loc[signals.positions == 1.0].index, signals.short_mavg[signals.positions == 1.0], '^', markersize=10, color= 'g')

    #sell signal
    ax1.plot(signals.loc[signals.positions == -1.0].index, signals.short_mavg[signals.positions == -1.0], 'v', markersize=10, color = 'r')
    plt.title('Trading strategy on predicted data with green (buy) and red (sell) markers')

    bytes_image3 = create_io_obj()
    
    pft = pf(symbol, lstm_predictions, signals, initial_capital)
    portfolio = pft.backtest_portfolio()


    fig = plt.figure(figsize=(8,6))

    # Plot equity curve in dollars

    ax1 = fig.add_subplot(111, ylabel='Portfolio Value in $')
    portfolio['total'].plot(ax=ax1, lw=2, label = 'Total Cash = Cash + Holdings')
    ax1.plot(portfolio.loc[signals.positions == 1.0].index, portfolio.total[signals.positions == 1.0], '^', markersize=10, color='g')
    ax1.plot(portfolio.loc[signals.positions == -1.0].index, portfolio.total[signals.positions == -1.0], 'v', markersize=10, color='r')
    plt.title('Plot of equity curve with buy (green) and sell (red) markers')

    bytes_image4 = create_io_obj()

    return bytes_image1, bytes_image2, bytes_image3, bytes_image4

