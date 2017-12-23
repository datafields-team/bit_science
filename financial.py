import os
import PIL
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg
import time
import csv
import matplotlib.dates as mdates
from matplotlib.dates import date2num
import matplotlib.ticker as mticker
from matplotlib.finance import candlestick_ohlc


from logger import Logging

#TODO variable store filenames, verify everything is stored to self, have read in function to store data as pd/np

class Finance(object):
    def __init__(self, infile, name,outfile='output/'):
        self.data = pd.DataFrame()
        self.name = name
        #dictionary of filename jpgs starting on the ith day value is y val on the i+n+1 th day
        self.y_vals = {}
        self.infile = 'data/'+infile
        self.outfile = outfile
        self.perc_change = []
        self.perc_close = []
        self.yvals = []
        self.np_images = []
        self.fig = plt.figure()

    def process_files(self, n=30, lookahead=0, max_files=70000):
        start_time = time.time()
        self.data = pd.read_csv(self.infile)
        self.data.dropna(axis=0, how='any', inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        print 'read data', len(self.data), self.data.columns
        try:
            self.data = self.data[['Date', 'Open', 'High', 'Low', 'Close']]
        except:
            self.data = self.data[['Timestamp', 'Open', 'High', 'Low', 'Close']]
        print self.data.columns
        self.data.columns = ['Date', 'Open', 'High', 'Low', 'Close']
        print self.data.columns
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data['Date'] = self.data['Date'].map(mdates.date2num)
        self.data = self.data.head(max_files)
        #self.ax.set_ylim((np.max(self.data.High), np.min(self.data.Low)))
        self.convert_ndays_to_image(n, lookahead)
        end_time = time.time()
        try:
            os.remove('tmp.png')
            print 'deleting temp.png'
        except:
            pass
        print 'writing out y_val to y_val.csv'
        #self.write_csv(self.yvals, 'yvals.csv')
        #self.write_csv(self.change,'change.csv')
        #self.write_csv(self.y_vals, 'old_y_vals.csv')
        #print self.np_images[0].shape
        with open(self.outfile + self.name + 'images.pkl','wb') as p:
            pickle.dump(self.np_images, p)
        #np.savetxt('images.csv', self.np_images, delimiter=',')
        np.savetxt(self.outfile + self.name + 'delta_high.csv', np.array(self.perc_change), delimiter=',')
        np.savetxt(self.outfile + self.name + 'delta_close.csv', np.array(self.perc_close), delimiter=',')
        np.savetxt(self.outfile + self.name +'delta_next.csv', np.array(self.yvals), delimiter=',')
        print 'process time: %f'%(end_time - start_time)
        #img = mpimg.imread('temp.jpg')
        #plt.imshow(img)
        #plt.show()
        
    def convert_ndays_to_image(self,n, lookahead=0):
        #calulcate the full dataframe values for rolling, std, bollinger bands upfront then parse for imaging
        #self.rolling = self.data['Adj_Close'].rolling(n).mean()
        #self.std = self.data['Adj_Close'].rolling(n).std()
        #self.adj = self.data['Adj_Close']
        #self.upper_bollinger = self.rolling + 2* self.std
        #self.lower_bollinger = self.rolling - 2* self.std
        #how many batches can be y value labeled (-1) for last
        num_batches = len(self.data) - n - 1
        imsize = 32
        print('creating %s by %s images - can be changed in financial.py :84'%(imsize, imsize))
        #starting at nth day (first day with rolling value), ending at len(data)-1 th day for last day with y val
        for i in range(0, len(self.data)-n-lookahead-1):
            print 'processing days %s-%s of %s'%(i, n+i,len(self.data)-1)
            self.write_image(n,i,n+i, imsize)
            self.yvals.append(self.data['Close'][n+i])
            max_close = np.max(self.data['Close'][n+i:n+i+lookahead])
            max_high = np.max(self.data['High'][n+i:n+i+lookahead])
            high_change = (max_high - self.data['Close'][n+i-1]) / self.data['Close'][n+i-1]
            close_change = (max_close - self.data['Close'][n+i-1]) / self.data['Close'][n+i-1]
            self.perc_change.append(high_change)
            self.perc_close.append(close_change)
            self.y_vals[i] = self.data['Close'][i+n]

    def write_image(self, n, begin, end, imsize=32):
            #fig = plt.figure()
            #ax = plt.subplot2grid((6,1),(0,0),rowspan=6, colspan=1)
            self.ax = plt.subplot()
            plt.tight_layout()
            self.ax.axis('off')
            candlestick_ohlc(self.ax, self.data.values[begin:end], width=0.6, colorup='g', colordown='r')
            plt.savefig('tmp.png')
            I = np.asarray(Image.open('tmp.png').convert('RGB').resize((imsize, imsize),Image.ANTIALIAS),np.uint8)
            #img = PIL.Image.open("tmp.png").convert("RGB")
            #imgarr = np.array(img) 
            self.np_images.append(I)
            #os.remove('tmp.png')
            plt.cla()
            #n+1 days in range
            #t = np.arange(0,n+1,1)
            #plt.plot(t,self.adj.loc[begin:end],t,self.rolling.loc[begin:end],t,self.up_boll.loc[begin:end], t,self.low_boll.loc[begin:end])
            #plt.savefig('/home/ubuntu/store/fin_data/temp')
            #just in case - clf
            #Image.open('/home/ubuntu/store/fin_data/temp.png').save('/home/ubuntu/store/fin_data/%s.jpg'%(begin),'JPEG')
            #plt.close(self.fig)

    def write_csv(self,d, filename):
        if filename is None:
            filename = 'output/'
        with open(filename, 'w') as f:
            w = csv.writer(f)
            if type(d) == list:
                w.writerows(d)
            else:
                w.writerows(d.items())

if __name__ == '__main__':
    f = Finance('AAPL.csv', 'AAPL')
    f.process_files(60,5)

