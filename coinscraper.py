import coinmarketcap as cmc
import pandas as pd
import numpy as np
import time
from logger import Logging

class BitReader(object):

    def __init__(self, filename, convert_time=False, convert_time_col=None):
        self.filename = filename
        self.logger = Logging()
        self.logger.info('begin reading file: %s'%(filename))
        self.read_file()
        self.logger.info('successful file read: %s'%(filename))
        if convert_time and convert_time_col:
            self.set_date_col(convert_time_col)
            self.convert_time()
            self.logger.info('converted time column: %s'%(convert_time_col))

    def get_data(self):
        return self.data
    
    def set_date_col(self, col):
        self.date_col = col

    def read_file(self):
        #use pandas to read in data, drop na values
        self.data = pd.read_csv(self.filename).dropna()

    def convert_time(self, unit='s'):
        # convert unix time to human readable
        self.data[self.date_col] = pd.to_datetime(self.data[self.date_col],unit=unit)

    def cols(self, cols):
        return self.data[cols]

    def rows(self, start, n=0):
        if n < 1:
            return self.data.iloc[start:].as_matrix(cols)
        return self.data.iloc[start:start+n]

    # return a np array of [start:start+n] rows and cols columns
    def to_numpy_rows_cols(self, cols, start, n=0):
        if n < 1:
            return self.data.iloc[start:].as_matrix(cols)
        return self.data.iloc[start:start+n].as_matrix(cols)


# ancillary functions to return dataframes of specific live data alt coins
def collect_data(coin='', df=pd.DataFrame()):
       obj = cmc.Market()
       data = obj.ticker(coin)
       return df.append(pd.DataFrame(data))

# retrieve the call id's for all alt coins from coinmarketcap live feed
def coin_names():
       obj = cmc.Market()
       data = obj.ticker()
       return [coin['id'] for coin in data]
        
if __name__ == '__main__':
    #all_coins = coin_names() 
    #my_coins = ['tenx', 'adx-net','Bitcoin', 'NEO', 'OmiseGo','Legends Room']
    #for coin in all_coins:
    #pay = pd.DataFrame()
    #coin = 'tenx'
    #for i in range(5):
        #pay = collect_data(coin, pay)
        #print len(pay), pay.shape
            #
        #time.sleep(30)
    #print pay
    filename = '../store/crypto/coinbaseUSD_1-min_data_2014-12-01_to_2017-05-31.csv'
    print('reading file: %s' % (filename))
    bit = BitReader(filename, True, 'Timestamp')
    arr = bit.to_numpy_rows_cols(['Close', 'Volume_(BTC)'], -10)
    print arr
