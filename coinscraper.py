import coinmarketcap as cmc
import pandas as pd
import numpy as np
import time
import csv
import os
from logger import Logging


# ancillary functions to return dataframes of specific live data alt coins
def collect_data(n=10):
       obj = cmc.Market()
       data = obj.ticker(limit=10)
       return data

# retrieve the call id's for all alt coins from coinmarketcap live feed
def coin_names():
       obj = cmc.Market()
       data = obj.ticker()
       return [coin['id'] for coin in data]
       
def output_data(list_df):
    for symbol in list_df:
        filename = 'data/' + symbol['symbol'] + '_scrape.csv'
        if os.path.exists(filename):
            f = open(filename, 'a')
            writer = csv.writer(f)
            writer.writerow(symbol.values())
        else:
            f = open(filename, 'w')
            writer = csv.writer(f)
            writer.writerow(symbol.keys())



if __name__ == '__main__':
    #all_coins = coin_names() 
    #my_coins = ['tenx', 'adx-net','Bitcoin', 'NEO', 'OmiseGo','Legends Room']
    minutes = 0
    while(True):
        data = collect_data()
        if minutes % 60 == 0:
            print 'collecting data for \n' + ' '.join([item['symbol'] for item in data])
        output_data(data)
        time.sleep(60)
        minutes +=1
        print 'scraping for % hrs %s min'%(np.divmod(minutes,60))

    
