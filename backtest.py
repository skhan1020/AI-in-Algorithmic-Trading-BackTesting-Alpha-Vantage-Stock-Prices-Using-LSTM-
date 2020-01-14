import stock_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tradingstrategy import MovingAverage as mv
from tradingstrategy import Volatility as vol
from tradingstrategy import MovingAverageCrossStrategy as cs
from tradingstrategy import MarketOnClosePortfolio as pf
from autocorrelation import acf_pacf as cf
from models import arima_model as am
from models import lstm_model as lm


if __name__ == '__main__':

    # Get Stock Price Data and Company Symbol

    alpha_API = stock_data
    stock_df, SYMBOL = alpha_API.get_data()

    
    # Enter Short Window, Long Window and Initial Capital for Trading
    # Strategies

    short_window = 20
    long_window = 100
    initial_capital = 1000

    # Shape of DataFrame

    n = stock_df.shape[0]
    p = stock_df.shape[1]


    # Plot of Company Stock Prices

    plt.figure(figsize=(8,6))
    stock_df[SYMBOL].plot(label = SYMBOL)
    plt.legend()
    plt.ylabel('Price in $')
    plt.title(SYMBOL + ' stock prices obtained from Alpha Vantage API')
    plt.show()

    # Moving Average (Short and Long) of Stock Prices

    mavg = mv(SYMBOL, stock_df, short_window, long_window)
    mavg.generate_mavg()

    # Volatility Mesurement of Company's Stocks

    vol = vol(SYMBOL, stock_df, min_periods=75)
    vol.volatility()


    ###############     Test for Stationarity    ################

    # Setting Training and Testing Data -- 80% of Data (Train)

    train_start = 0
    train_end = int(np.floor(0.8*n))
    test_start = train_end
    test_end = n


    #####  Plot AutoCorrelation, Partial AutoCorrelation Functions and Check for Stationarity ##### 

    autocorrelation =  cf(stock_df, SYMBOL)
    
    # Performing the Dickey-Fuller Test
    
    result =  autocorrelation.check_stationarity() 
    
    # Plot of AutoCorrelation and Partial AutoCorrelation vs Lags -- Choice of
    # ARMA parameters

    acf_pacf_plots = autocorrelation.plot()

    # Stationarize Time Series by Differencing - Choose 'd' parameter

    arima_diff_order = autocorrelation.acf_diff()

    # Check Stationarity After Differencing -- Augmented Dickey Fuller Test

    if result == 0:
        
        print('ACF plots after First Order Differencing')
        stationarity_after_diff = cf(stock_df.diff().dropna(), SYMBOL)

        print('Check for Stationarity after Differencing')
        stationarity_after_diff.check_stationarity()

    
    #####  Implementing ARIMA Model to fit the test set ##### 
    
    arm = am(stock_df, SYMBOL, train_end)
    predictions, mse_arima = arm.evaluate()

    arima_predict = pd.DataFrame(predictions, index = stock_df[train_end:test_end].index, columns = [SYMBOL])

    # Plot ARIMA predictions

    plt.figure(figsize=(8,6))
    plt.plot(stock_df[:train_end], color = 'b', label = 'Training Data')
    plt.plot(stock_df[train_end:], color = 'g', label = 'Test Data')
    plt.plot(arima_predict, color='orange', label = 'predicted data using ARIMA')
    plt.legend()
    plt.ylabel('Price in $')
    plt.title('ARIMA model predictions for '+SYMBOL)
    plt.show()

    #####   Implementing the LSTM Model to Predict Future Prices   #####

    lst = lm(SYMBOL, stock_df, train_start, train_end, test_start, test_end)
    lstm_predictions, dev_set, look_back, train_loss, test_loss, mse_lstm = lst.evaluate()

    # Plot of Train, Test Set Losses - Check for Overfitting/Underfitting

    plt.figure(figsize=(8,6))
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.legend()
    plt.title('Train and Test Set Loss vs Epochs')
    plt.show()


    # Plot of Training, Testing and Predicted Stock Prices

    plt.figure(figsize=(8,6))
    plt.plot(stock_df[:train_end+look_back+1], color = 'b', label = 'Training Data')
    plt.plot(dev_set, color = 'g', label = 'Test Data')
    plt.plot(lstm_predictions, color = 'orange', label = 'Predicted Data')
    plt.ylabel('Close price in $')
    plt.title('Applying LSTM on ' + SYMBOL + ' stock prices obtained from Alpha Vantage API')
    plt.legend()
    plt.show()


    #### Net Returns on the Predicted Data Based on Strategy #####


    mac = cs(SYMBOL, lstm_predictions, short_window, long_window)
    signals = mac.generate_signals()

    # Plot of Test Prices and Predicted Prices along with Markers for Trading
    # Strategy

    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111, ylabel='Price in $')
    dev_set[SYMBOL].plot(ax=ax1, color = 'k', lw=2, label='actual data')
    lstm_predictions[SYMBOL].plot(ax=ax1, color = 'm', lw=2, label='predicted data')
    signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2)
    plt.legend()

    #buy signal
    ax1.plot(signals.loc[signals.positions == 1.0].index, signals.short_mavg[signals.positions == 1.0], '^', markersize=10, color= 'g')

    #sell signal
    ax1.plot(signals.loc[signals.positions == -1.0].index, signals.short_mavg[signals.positions == -1.0], 'v', markersize=10, color = 'r')
    plt.title('Trading strategy on predicted data with green (buy) and red (sell) markers')
    plt.show()

    pft = pf(SYMBOL, lstm_predictions, signals, initial_capital)
    portfolio = pft.backtest_portfolio()


    fig = plt.figure(figsize=(8,6))

    # Plot equity curve in dollars

    ax1 = fig.add_subplot(111, ylabel='Portfolio Value in $')
    portfolio['total'].plot(ax=ax1, lw=2, label = 'Total Cash = Cash + Holdings')
    ax1.plot(portfolio.loc[signals.positions == 1.0].index, portfolio.total[signals.positions == 1.0], '^', markersize=10, color='g')
    ax1.plot(portfolio.loc[signals.positions == -1.0].index, portfolio.total[signals.positions == -1.0], 'v', markersize=10, color='r')
    plt.title('Plot of equity curve with buy (green) and sell (red) markers')
    plt.show()

    print('Net Profit % based on Keras algorithm:', 100*(portfolio['total'][-1]-portfolio['total'][0])/portfolio['total'][0])


    ####  Analysis of LSTM and ARIMA models ### 

    # Test and Predicted Stock Prices in last 100 days - ARIMA
    
    plt.figure(figsize=(8,6))
    plt.plot(stock_df[-100:], color = 'g', label = 'Test Data -- Last 100 Days')
    plt.plot(arima_predict.shift(periods=-1, fill_value=0)[-100:], color='orange', label = 'predicted data (ARIMA) -- Last 100 Days')
    plt.plot(lstm_predictions.shift(periods=-2, fill_value=0)[-100:], color = 'r', label = 'predicted data (LSTM) -- Last 100 Days ')
    plt.legend()
    plt.ylabel('Price in $')
    plt.title('Compare ARIMA and LSTM model predictions in last 100 days')
    plt.show()
    
    

    if mse_arima < mse_lstm:
        print('#########       ARIMA model predictions are better than LSTM           #########')
    else:
        print('#########       LSTM model predictions are better than ARIMA           #########')

