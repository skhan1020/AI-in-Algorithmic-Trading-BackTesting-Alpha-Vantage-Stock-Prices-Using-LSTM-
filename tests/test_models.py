import unittest
import numpy as np
import  sys
sys.path.insert(1,'../')
import stock_data
from models import arima_model, lstm_model

class TestData(unittest.TestCase):

    def setUp(self):

        self.stock_price = stock_data
        self.data, self.company = self.stock_price.get_data()
        self.train_len = int(np.floor(0.8*len(self.data)))

    def test_data(self):

        self.assertIsNotNone(self.data)
        self.assertEqual(self.company, 'AAPL')


    def test_arima(self):

        arm = arima_model(self.data, self.company, self.train_len)
        arima_predictions, arima_error = arm.evaluate()

        self.assertIsNotNone(arima_predictions)

    def test_lstm(self):

        lstm = lstm_model(self.company, self.data, 0, self.train_len, self.train_len, len(self.data))
        lstm_predictions = lstm.evaluate()

        self.assertIsNotNone(lstm_predictions)


if __name__ == '__main__':
    unittest.main()
