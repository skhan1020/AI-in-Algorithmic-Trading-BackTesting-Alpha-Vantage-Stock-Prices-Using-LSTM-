import numpy as np
import urllib
import unittest
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(1,'../')
import config

class TestURL(unittest.TestCase):


    def test_url(self):

        SYMBOL =  'AAPL'

        with patch('urllib.request.urlopen') as urlopen_mock:
            mock_response  = MagicMock()
            mock_response.read.return_value = 'Success'
            urlopen_mock.return_value = mock_response

            response = urllib.request.urlopen('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + SYMBOL + '&outputsize=full&apikey='+ config.public_api_key)
            
            self.assertEqual(response.read(), 'Success')


if __name__ == '__main__':
    unittest.main()


