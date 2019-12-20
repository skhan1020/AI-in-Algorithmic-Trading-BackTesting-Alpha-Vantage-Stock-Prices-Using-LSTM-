import unittest
import  sys
sys.path.insert(1,'../')
import stock_data
from tradingstrategy import MovingAverageCrossStrategy as mac
from tradingstrategy import MarketOnClosePortfolio as pft


class TestData(unittest.TestCase):

    def setUp(self):

        self.stock_price = stock_data
        self.data, self.company = self.stock_price.get_data()
        self.short_window = 20
        self.long_window = 100
        self.initial_capital = 1000


    def test_tradingstrategy(self):

        strategy = mac(self.company, self.data, self.short_window, self.long_window)
        signals = strategy.generate_signals()

        pf = pft(self.company, self.data, signals, self.initial_capital)
        positions = pf.generate_positions()

        portfolio = pf.backtest_portfolio()

        self.assertIsNotNone(signals)
        self.assertIsNotNone(positions)
        self.assertIsNotNone(portfolio)


if __name__ == '__main__':
    unittest.main()
