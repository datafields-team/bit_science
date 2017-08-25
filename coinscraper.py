import coinmarketcap as cmc
import pandas as pd
import numpy as np
import time

def collect_data(coin='', df=pd.DataFrame()):
       obj = cmc.Market()
       data = obj.ticker(coin)
       #print df.append(pd.DataFrame(data))
       return df.append(pd.DataFrame(data))

def coin_names():
       obj = cmc.Market()
       data = obj.ticker()
       return [coin['id'] for coin in data]
        
if __name__ == '__main__':
    all_coins = coin_names() 
    my_coins = ['tenx', 'adx-net','Bitcoin', 'NEO', 'OmiseGo','Legends Room']
    #for coin in all_coins:
    pay = pd.DataFrame()
    coin = 'tenx'
    for i in range(5):
        pay = collect_data(coin, pay)
        print len(pay), pay.shape
            
        time.sleep(30)
    print pay
