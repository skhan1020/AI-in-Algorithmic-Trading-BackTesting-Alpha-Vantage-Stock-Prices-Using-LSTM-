import urllib.request, urllib.parse, urllib.error
import config
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

class MovingAverage:

    def __init__(self, symbol, data, short_window, long_window):
        self.symbol = symbol
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_mavg(self):
        
        # Moving Averages
        aapl_moving_avg_short = self.data[self.symbol].rolling(window=self.short_window).mean()
        aapl_moving_avg_long = self.data[self.symbol].rolling(window=self.long_window).mean()

        plt.figure(figsize=(8,6))
        self.data[self.symbol].plot(label='close price')
        aapl_moving_avg_short.plot(label='moving avg -- ' + str(self.short_window) + ' time periods')
        aapl_moving_avg_long.plot(label='moving avg -- ' + str(self.long_window) + ' time periods')
        plt.legend()
        plt.title('Short and Long Moving Averages of ' + self.symbol + ' stocks along with daily close prices')
        plt.show()

    
class Volatility:
    
    def __init__(self, symbol, data, min_periods):
        self.symbol = symbol
        self.data = data
        self.min_periods = min_periods
        self.daily_pct_change = self.percent_change()

    def percent_change(self):

        # Perecent Change
        daily_pct_change = self.data.pct_change().dropna(axis=0)
        # print(daily_pct_change)

        daily_pct_change.hist(bins=50, range=[-0.15, 0.15], sharex=True, figsize=(8,6))
        plt.title('Histogram of percent change in stock prices (close) of ' + self.symbol)
        plt.show()

        return daily_pct_change

    def volatility(self):
    
        # Volatility 
        vol = self.daily_pct_change.rolling(self.min_periods).std()*np.sqrt(self.min_periods)
        vol.plot(figsize=(8,6), label=str(self.min_periods) + ' time periods')
        plt.ylabel('Volatility')
        plt.title('Volatility in the price of ' + SYMBOL + ' stocks')
        plt.show()


class MovingAverageCrossStrategy:

    def __init__(self, symbol, data, short_window=20, long_window=100):
        self.symbol = symbol
        self.data = data
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):

        # Strategy

        signals = pd.DataFrame(index=self.data.index,columns=['signal'])
        signals['short_mavg'] = self.data[self.symbol].rolling(window=self.short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = self.data[self.symbol].rolling(window=self.long_window, min_periods=1, center=False).mean()


        pd.set_option('mode.chained_assignment', None)
        signals['signal'][self.short_window:] = np.where(signals['short_mavg'][self.short_window:] > signals['long_mavg'][self.short_window:], 1, 0)

        signals['positions'] = signals['signal'].diff()

        # print(signals[:100])


        return signals


class MarketOnClosePortfolio:

    def __init__(self, symbol, data, signals, initial_capital):
        self.symbol = symbol
        self.data = data
        self.signals = signals
        self.initial_capital = initial_capital
        self.positions = self.generate_positions()

    def generate_positions(self):
        
        # BackTesting
        positions = pd.DataFrame(index=self.signals.index,columns=[self.symbol])
        positions[self.symbol] = 0.0
        positions[self.symbol] = 100*self.signals['signal']
        return positions

    def backtest_portfolio(self):
    
        portfolio = self.positions.multiply(self.data[self.symbol], axis=0)
        pos_diff = self.positions.diff()
        portfolio['holdings'] = (self.positions.multiply(self.data[self.symbol], axis=0)).sum(axis=1)
        portfolio['cash'] = self.initial_capital - (pos_diff.multiply(self.data[self.symbol], axis=0)).sum(axis=1).cumsum()
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()

        return portfolio


class arima_model:
    
    def __init__(self, symbol, data, train_len, test_len):
        
        self.symbol = symbol
        self.data = data
        self.train_len = train_len
        self.test_len = test_len

    def evaluate(self):

        # Setting up Training and Test/Dev Sets

        val = self.data.values

        data_train, data_test = val[:self.train_len], val[self.train_len:self.test_len]
        history = [x for x in data_train]
        predictions = list()

        # Fitting the ARIMA model: parameters (p, d, q) = (2, 1, 0)

        for t in range(len(data_test)):
            model = ARIMA(history, order=(2,1,0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            predictions.append(output[0])

        # Error in ARIMA model

        error = mean_squared_error(data_test, predictions)
        print('Error in ARIMA model', error)
        
        arima_predict = pd.DataFrame(predictions, index = self.data[self.train_len:self.test_len].index, columns = [self.symbol])
        
        return arima_predict


class lstm_model: 

    def __init__(self, symbol, data, train_len1, train_len2, test_len1, test_len2):

        self.symbol = symbol
        self.data = data
        self.train_len1 = train_len1
        self.train_len2 = train_len2
        self.test_len1 = test_len1
        self.test_len2 = test_len2
    
    def evaluate(self):

        val = self.data.values

        # Setting up X_train, y_train, X_test, y_test
            
        data_train = val[np.arange(self.train_len1, self.train_len2)].reshape(-1, 1)
        data_test = val[np.arange(self.test_len1, self.test_len2)].reshape(-1, 1)

        # Normalizing Data

        scaler = MinMaxScaler()
        data_train_scaled = scaler.fit_transform(data_train)
        data_test_scaled = scaler.fit_transform(data_test)

        def create_dataset(dataset, look_back):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back -1):
                a = dataset[i: i+ look_back, 0]
                dataX.append(a)
                dataY.append(dataset[i+look_back, 0])
            return np.array(dataX), np.array(dataY)


        look_back = 3
        X_train, y_train = create_dataset(data_train_scaled, look_back)
        X_test, y_test = create_dataset(data_test_scaled, look_back)


        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_train.shape[1], 1))


        # Applying LSTM

        regressor = Sequential()
        regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(LSTM(units=50))

        regressor.add(Dense(units=1))

        regressor.compile(loss='mean_squared_error', optimizer='adam')
        history = regressor.fit(X_train, y_train, epochs=10, batch_size=32,
                validation_data=(X_test, y_test))
        # print(regressor.evaluate(X_test, y_test))

        predicted_price = regressor.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)
        test_price = scaler.inverse_transform(y_test.reshape(-1, 1))

        test_df = pd.DataFrame(data = test_price, index = self.data[self.train_len2+look_back+1:].index, columns = [self.symbol])
        predict_df = pd.DataFrame(data = predicted_price, index = self.data[self.train_len2+look_back+1:].index, columns = [self.symbol])
        
        return predict_df, test_df, look_back, history.history['loss'], history.history['val_loss']


if __name__ == '__main__':
    
    # Enter the Company Symbol, Short Window, Long Window and Initial Capital

    SYMBOL = input('Enter Company Symbol: ')
    short_window = int(input('Enter short window :'))
    long_window = int(input('Enter long window :'))
    initial_capital = float(input('Enter initial capital :'))


    # Enter Alpha Vantage API

    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + SYMBOL + '&outputsize=full&apikey='+ config.public_api_key

    # Extract JSON data 

    alpha_vantage_data = urllib.request.urlopen(url).read().decode()
    js = json.loads(alpha_vantage_data)


    close_price = list()
    dates = list()

    # Extract Close Prices 
    for item in js["Time Series (Daily)"]:
        dates.append(item)
        close_price.append(float(js['Time Series (Daily)'][item]['4. close']))

    # DataFrame of Close Prices

    stock_df = pd.DataFrame(data=close_price, index=dates)
    stock_df.index = pd.to_datetime(stock_df.index)
    stock_df = stock_df.sort_index(ascending=True)
    stock_df.columns = [SYMBOL]
    
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

    mavg = MovingAverage(SYMBOL, stock_df, short_window, long_window)
    mavg.generate_mavg()

    # Volatility Mesurement of Company's Stocks

    vol = Volatility(SYMBOL, stock_df, min_periods=75)
    vol.volatility()


    ###############     Test for Stationarity    ################

    # Setting Training and Testing Data -- 80% of Data (Train)
    
    train_start = 0
    train_end = int(np.floor(0.8*n))
    test_start = train_end
    test_end = n
    
    # Performing the Dickey-Fuller Test

    result = ts.adfuller(stock_df[SYMBOL])
    print('Results of the Dickey Fuller Test \n', result)
    print('\n')
    if result[0] < result[4]['1%'] or result[0] < result[4]['5%'] or result[0] < result[4]['10%']:
        print('#######    Null Hypothesis of Non-Stationarity can be rejected!    #######')
    else:
        print('#######    Null Hypothesis of Non-Stationarity cannot be rejected!    #######')
   

   
    #####  Implementing ARIMA Model to fit the test set ##### 

    ### Bug Present : Inaccurate Predictions -- Zero Variance in predicted data! 

    arm = arima_model(SYMBOL, stock_df, train_end, test_end)
    arm_predict = arm.evaluate()

    # Plot the ARIMA predicted and test data 

    plt.figure(figsize=(8,6))
    plt.plot(stock_df[test_start:test_end], label = 'test data')
    plt.plot(arm_predict,color='r', label = 'predicted data using ARIMA')
    plt.legend()
    plt.ylabel('Price in $')
    plt.show()
    
    #####   Implementing the LSTM Model to Predict Future Prices   #####

    lst = lstm_model(SYMBOL, stock_df, train_start, train_end, test_start, test_end)
    predictions, dev_set, look_back, train_loss, test_loss = lst.evaluate()

    # Plot of Train, Test Set Losses - Check for Overfitting/Underfitting

    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.legend()
    plt.title('Train and Test Set Loss vs Epochs')
    plt.show()


    # Plot of Training, Testing and Predicted Stock Prices

    plt.figure(figsize=(8,6))
    plt.plot(stock_df[:train_end+look_back+1], color = 'b', label = 'Training Data')
    plt.plot(stock_df[train_end+look_back+1:], color = 'g', label = 'Test Data')
    plt.plot(predictions, color = 'orange', label = 'Predicted Data')
    plt.ylabel('Close price in $')
    plt.title('Applying LSTM on ' + SYMBOL + ' stock prices obtained from Alpha Vantage API')
    plt.legend()
    plt.show()


    #### Net Returns on the Predicted Data Based on Strategy #####


    mac = MovingAverageCrossStrategy(SYMBOL, predictions, short_window, long_window)
    signals = mac.generate_signals()

    # Plot of Test Prices and Predicted Prices along with Markers for Trading
    # Strategy

    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111, ylabel='Price in $')
    dev_set[SYMBOL].plot(ax=ax1, color = 'k', lw=2, label='actual data')
    predictions[SYMBOL].plot(ax=ax1, color = 'm', lw=2, label='predicted data')
    signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2)
    plt.legend()

    #buy signal
    ax1.plot(signals.loc[signals.positions == 1.0].index, signals.short_mavg[signals.positions == 1.0], '^', markersize=10, color= 'g')

    #sell signal
    ax1.plot(signals.loc[signals.positions == -1.0].index, signals.short_mavg[signals.positions == -1.0], 'v', markersize=10, color = 'r')
    plt.title('Trading strategy on predicted data with green (buy) and red (sell) markers')
    plt.show()
    
    pft = MarketOnClosePortfolio(SYMBOL, predictions, signals, initial_capital)
    portfolio = pft.backtest_portfolio()


    fig = plt.figure(figsize=(8,6))

    #Plot equity curve in dollars

    ax1 = fig.add_subplot(111, ylabel='Portfolio Value in $')
    portfolio['total'].plot(ax=ax1, lw=2, label = 'Total Cash = Cash + Holdings')
    ax1.plot(portfolio.loc[signals.positions == 1.0].index, portfolio.total[signals.positions == 1.0], '^', markersize=10, color='g')
    ax1.plot(portfolio.loc[signals.positions == -1.0].index, portfolio.total[signals.positions == -1.0], 'v', markersize=10, color='r')
    plt.title('Plot of equity curve with buy (green) and sell (red) markers')
    plt.show()
    
    print('Net Profit % based on Keras algorithm:', 100*(portfolio['total'][-1]-portfolio['total'][0])/portfolio['total'][0])



