import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        plt.title('Volatility in the price of ' + self.symbol + ' stocks')
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



