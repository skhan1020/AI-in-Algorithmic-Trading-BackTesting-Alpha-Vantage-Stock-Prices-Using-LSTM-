import urllib.request, urllib.parse, urllib.error
import config
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error

# Enter the Company Symbol


SYMBOL = input('Enter Company Symbol: ')

# Enter url Alpha Vantage API

url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + SYMBOL + '&outputsize=full&apikey='+ config.public_api_key

# Extract JSON data

data = urllib.request.urlopen(url).read().decode()
js = json.loads(data)


close_price = list()
dates = list()

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


# Setting Training and Testing Data -- 80% of Data (Train)

train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end
test_end = n



# Autocorrelation vs Lag  --- ACF tails off -> AR model ; ACF cuts off after
# lag p -> MA model

plot_acf(stock_df, lags=5000)
plt.xlabel('Lags')
plt.ylabel('Autocorrelation (ACF) plot for ' + SYMBOL + ' stock prices')
plt.title(SYMBOL + ' - ACF vs Lags')
plt.show()

# Partial Autocorrealtion vs Lag -- PACF cuts off after lag q -> AR model; PACF
# tails off -> MA model

plot_pacf(stock_df, lags=50)
plt.xlabel('Lags')
plt.ylabel('Partial Autocorrelation (PACF) plot for ' + SYMBOL + ' stock prices')
plt.title(SYMBOL + ' - PACF vs Lags')
plt.show()

# Choosing AR/MA models for forecasting -- Autocorrelation plots for different
# lags show a linear relationship -- autoregressive models

fig, axes = plt.subplots(2,2, figsize=(6, 8))
plt.title(SYMBOL + ' Autocorrelation plot ')

ax_idcs = [(0,0), (0, 1), (1, 0), (1, 1)]

for lag, ax_coords in enumerate(ax_idcs, 1):
    ax_row, ax_col = ax_coords
    axis = axes[ax_row][ax_col]
    lag_plot(stock_df, lag=lag, ax=axis)
    axis.set_title('Lag = ' + str(lag))

plt.subplots_adjust(hspace=0.5)
plt.show()

# AutoCorrelation after 1st Order Differencing -- Order p determined from
# cut-off lag value

plot_acf(stock_df.diff().dropna(), lags=5000)
plt.xlabel('Lags')
plt.ylabel('ACF after 1st order Differencing')
plt.title(SYMBOL + ' - ACF (d=1) vs Lags')
plt.show()

# Forecasting using ARIMA Model -- AR(2) with 1st Order Differencing 

train_ar = stock_df[:train_end].values
test_ar = stock_df[train_end:].values

history = [x for x in train_ar]
predictions = list()
for t in range(len(test_ar)):
    model = ARIMA(history, order=(2,1,0))
    model_fit = model.fit(disp=0)
    output=model_fit.forecast()
    yhat=output[0]
    predictions.append(yhat)
    obs=test_ar[t]
    history.append(obs)

error = mean_squared_error(test_ar, predictions)
print('Error in ARIMA model', error)

arima_predict = pd.DataFrame(predictions, index = stock_df[train_end:test_end].index, columns = [SYMBOL])

# Plot ARIMA predictions

plt.figure()
# plt.plot(stock_df[:train_end], color = 'b', label = 'Training Data')
plt.plot(stock_df[train_end:], color = 'g', label = 'Test Data')
plt.plot(arima_predict, color='orange', label = 'predicted data using ARIMA')
plt.legend()
plt.ylabel('Price in $')
plt.title('ARIMA model predictions for '+SYMBOL)
plt.show()


