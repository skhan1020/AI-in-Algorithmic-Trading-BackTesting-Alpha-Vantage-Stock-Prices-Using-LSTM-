import urllib.request, urllib.parse, urllib.error
import json
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
import config

def get_data(SYMBOL):


    # Enter Alpha Vantage API URL

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

    return stock_df, SYMBOL
