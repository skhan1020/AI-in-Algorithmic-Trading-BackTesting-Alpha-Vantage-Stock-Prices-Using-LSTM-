import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

class arima_model:

    def __init__(self, data, symbol, train_len):

        self.data = data
        self.symbol = symbol
        self.train_len = train_len

    def evaluate(self):

        # Forecasting using ARIMA Model -- AR(2) with 1st Order Differencing

        train_ar = self.data[:self.train_len].values
        test_ar = self.data[self.train_len:].values
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
        print('Mean Squared Error in ARIMA model', error)

        return predictions, error

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
        history = regressor.fit(X_train, y_train, epochs=50, batch_size=32,
                validation_data=(X_test, y_test))
        

        predicted_price = regressor.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)
        test_price = scaler.inverse_transform(y_test.reshape(-1, 1))

        test_df = pd.DataFrame(data = test_price, index = self.data[self.train_len2+look_back+1:].index, columns = [self.symbol])
        predict_df = pd.DataFrame(data = predicted_price, index = self.data[self.train_len2+look_back+1:].index, columns = [self.symbol])

        error = mean_squared_error(test_price, predicted_price)
        print('Mean Squared Error in LSTM model', error)

        return predict_df, test_df, look_back, history.history['loss'], history.history['val_loss'], error
