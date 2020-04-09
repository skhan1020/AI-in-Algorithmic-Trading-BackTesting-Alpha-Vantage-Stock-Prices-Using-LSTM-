import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.regularizers import L1L2
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import pmdarima as pm

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
            model = ARIMA(history, order=(0,1,0))
            model_fit = model.fit(disp=0)
            output=model_fit.forecast()
            yhat=output[0]
            predictions.append(yhat)
            obs=test_ar[t]
            history.append(obs)
        
        print(model_fit.summary())

        error = mean_squared_error(test_ar, predictions)
        print('Mean Squared Error in ARIMA model', error)
        
        forecast = np.array(predictions)
        actual = np.array(test_ar)
        MAPE = np.mean(np.abs(forecast-actual)/np.abs(actual))

        print('Mean Absolute Percentage Error in ARIMA model', MAPE)

        return predictions, error

class AutomatedArima:

    def __init__(self, data, symbol, train_len):

        self.data = data
        self.symbol = symbol
        self.train_len = train_len

    def evaluate(self):
        
        train_ar = self.data[:self.train_len].values
        test_ar = self.data[self.train_len:].values

        model = pm.auto_arima(train_ar, start_p=1, start_q=1, test='adf', max_p=3,
                max_q=3, m=1, d=None, seasonal=False, start_P=0, D=0,
                trace=True, error_action='ignore',
                suppress_warnings=True,stepwise=True)

        print(model.summary())

        model.plot_diagnostics(figsize=(8,8))
        plt.show()
    
        def forecast_one_step():
            fc, conf_int = model.predict(n_periods=1, return_conf_int=True)
            return (fc.tolist()[0], np.asarray(conf_int).tolist()[0]) 

        forecasts, confidence_intervals = [], []

        for new_ob in test_ar:
            fc, conf = forecast_one_step()
            forecasts.append(fc)
            confidence_intervals.append(conf)
            model.update(new_ob)
        
        error = mean_squared_error(test_ar, forecasts)

        print('Mean Squared Error from Auto ARIMA model', error)

        return forecasts, error

class ANNModel:

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

        data_train = val[np.arange(self.train_len1, self.train_len2), :]
        data_test = val[np.arange(self.test_len1, self.test_len2), :]

        # Normalizing the Data

        scaler = MinMaxScaler()
        data_train_scaled = scaler.fit_transform(data_train)
        data_test_scaled = scaler.fit_transform(data_test)

        # Create Sliding Window -- " look_back " variable

        def create_dataset(dataset, look_back):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back -1):
                a = dataset[i: i+ look_back, 0]
                dataX.append(a)
                dataY.append(dataset[i+look_back, 0])
            return np.array(dataX), np.array(dataY)


        look_back = 240  # 3 input features

        X_train, y_train = create_dataset(data_train_scaled, look_back)
        X_test, y_test = create_dataset(data_test_scaled, look_back)

        # Placeholders for the input and output vectors

        X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, look_back])
        Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])

        # Initialization

        sigma = 1
        weight_initializer = tf.variance_scaling_initializer(mode="fan_avg",distribution="uniform",scale=sigma)
        bias_initializer = tf.zeros_initializer()

        # Desiging the Aritifical Neural Network (ANN) 

        n_neurons_1 = 512  # First layer has 512 units
        n_target = 1  # Output

        # First Hidden layer and Bias
        
        W_hidden_1 = tf.Variable(weight_initializer([look_back, n_neurons_1]))
        bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

        # Output and Bias associated with it

        W_out = tf.Variable(weight_initializer([n_neurons_1, n_target]))
        bias_out = tf.Variable(bias_initializer([n_target]))

        # Activation function for 1st hidden layer

        hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))

        # Computation of the output (Matrix Multiplication)

        out = tf.transpose(tf.add(tf.matmul(hidden_1, W_out), bias_out))

        # Loss Function

        mse = tf.reduce_mean(tf.math.squared_difference(out, Y))
        
        # Optimization (Adam)

        opt = tf.compat.v1.train.AdamOptimizer().minimize(mse)

        # Setting up a Session

        net = tf.compat.v1.Session()

        net.run(tf.compat.v1.global_variables_initializer())

        epochs = 50
        batch_size = 500


        for e in range(epochs):

            shuffle_indices = np.random.permutation(np.arange(len(y_train)))
            X_train = X_train[shuffle_indices]
            y_train = y_train[shuffle_indices]


            for i in range(0, len(y_train) // batch_size):
                start = i*batch_size
                batch_x = X_train[start:start + batch_size]
                batch_y = y_train[start:start + batch_size]

                net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        predicted_price = net.run(out, feed_dict={X: X_test})
        predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))
        test_price = scaler.inverse_transform(y_test.reshape(-1, 1))

        test_df = pd.DataFrame(data = test_price, index = self.data[self.train_len2+look_back+1:].index, columns=[self.symbol])
        predict_df = pd.DataFrame(data = predicted_price, index = self.data[self.train_len2+look_back+1:].index, columns = [self.symbol])

        mse_final = net.run(mse, feed_dict ={X: X_test, Y: y_test})

        error = mean_squared_error(test_price, predicted_price)
        print("Mean Squared Error in ANN model", error)

        forecast = np.array(predicted_price)
        actual = np.array(test_price)
        MAPE = np.mean(np.abs(forecast-actual)/np.abs(actual))

        print('Mean Absolute Percentage Error in ANN model', MAPE)

        return predict_df, test_df, look_back, error


class LSTMModel: 

    def __init__(self, symbol, data, train_len1, train_len2, test_len1, test_len2, reg):

        self.symbol = symbol
        self.data = data
        self.train_len1 = train_len1
        self.train_len2 = train_len2
        self.test_len1 = test_len1
        self.test_len2 = test_len2
        self.reg = reg

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


        look_back = 240
        X_train, y_train = create_dataset(data_train_scaled, look_back)
        X_test, y_test = create_dataset(data_test_scaled, look_back)


        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_train.shape[1], 1))


        # Applying LSTM

        regressor = Sequential()
        regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1), bias_regularizer = L1L2(l1=self.reg[0], l2=self.reg[1])))
        regressor.add(LSTM(units=50))

        regressor.add(Dense(units=1))

        regressor.compile(loss='mean_squared_error', optimizer='adam')
        history = regressor.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
        

        predicted_price = regressor.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)
        test_price = scaler.inverse_transform(y_test.reshape(-1, 1))

        test_df = pd.DataFrame(data = test_price, index = self.data[self.train_len2+look_back+1:].index, columns = [self.symbol])
        predict_df = pd.DataFrame(data = predicted_price, index = self.data[self.train_len2+look_back+1:].index, columns = [self.symbol])

        error = mean_squared_error(test_price, predicted_price)
        print('Mean Squared Error in LSTM model', error)

        forecast = np.array(predicted_price)
        actual = np.array(test_price)
        MAPE = np.mean(np.abs(forecast-actual)/np.abs(actual))

        print('Mean Absolute Percentage Error in LSTM model', MAPE)
        
        return predict_df, test_df, look_back, history.history['loss'], history.history['val_loss'], error

