import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import sys
sys.path.insert(1, '../')
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
    # plt.show()

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
    # plt.show()
   
    return create_io_obj()


def lstm_plots(data, symbol, train_start, train_end, test_start, test_end):

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

    return bytes_image1, bytes_image2

