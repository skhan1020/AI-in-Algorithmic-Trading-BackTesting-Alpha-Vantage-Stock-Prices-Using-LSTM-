import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import lag_plot


class AcfPacf:

    def __init__(self, data, symbol):

        self.data = data
        self.symbol = symbol

    def plot(self):

        # Autocorrelation vs Lag  --- ACF tails off -> AR model ; ACF cuts off after
        # lag q -> MA model

        plot_acf(self.data, lags=np.floor(len(self.data)*0.8))
        plt.xlabel('Lags')
        plt.ylabel('Autocorrelation (ACF) plot for ' + self.symbol + ' stock prices')
        plt.title(self.symbol + ' - ACF vs Lags')
        plt.show()

        # Partial Autocorrealtion vs Lag -- PACF cuts off after lag p -> AR model; PACF
        # tails off -> MA model

        plot_pacf(self.data, lags=np.floor(len(self.data)*0.008))
        plt.xlabel('Lags')
        plt.ylabel('Partial Autocorrelation (PACF) plot for ' + self.symbol + ' stock prices')
        plt.title(self.symbol + ' - PACF vs Lags')
        plt.show()

        # Choosing AR/MA models for forecasting -- Autocorrelation plots for different
        # lags show a linear relationship -- autoregressive models

        fig, axes = plt.subplots(2,2, figsize=(6, 8))
        plt.title(self.symbol + ' Autocorrelation plot ')

        ax_idcs = [(0,0), (0, 1), (1, 0), (1, 1)]

        for lag, ax_coords in enumerate(ax_idcs, 1):
            ax_row, ax_col = ax_coords
            axis = axes[ax_row][ax_col]
            lag_plot(self.data, lag=lag, ax=axis)
            axis.set_title('Lag = ' + str(lag))

        plt.subplots_adjust(hspace=0.5)
        plt.show()

    def acf_pacf_diff(self):

        # AutoCorrelation after 1st Order Differencing

        plot_acf(self.data.diff().dropna(), lags=np.floor(len(self.data)*0.8))
        plt.xlabel('Lags')
        plt.ylabel('ACF after 1st order Differencing')
        plt.title(self.symbol + ' - ACF (d=1) vs Lags')
        plt.show()
        

        # Partial AutoCorrelation after 1st Order Differencing

        plot_pacf(self.data.diff().dropna(), lags=np.floor(len(self.data)*0.008))
        plt.xlabel('Lags')
        plt.ylabel('Partial ACF after 1st order Differencing')
        plt.title(self.symbol + ' - PACF (d=1) vs Lags')
        plt.show()

    def check_stationarity(self):

        # Checking the Stationarity of Time Series -- Augmented Dickey Fuller Test

        result = adfuller(self.data[self.symbol], autolag='AIC')
        print('Results of the Dickey Fuller Test \n', result)
        print('ADF Statistic: ', result[0])
        print('p-value: ', result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print(key, value)
        print('\n')
        if result[0] < result[4]['1%'] or result[0] < result[4]['5%'] or result[0] < result[4]['10%']:
            print('#######    Null Hypothesis of Non-Stationarity can be rejected!    #######')
            print('\n')
            return 1
        else:
            print('#######    Null Hypothesis of Non-Stationarity cannot be rejected!    #######')
            print('\n')
            return 0


