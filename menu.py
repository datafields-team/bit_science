import sys
import os
import numpy as np
import tensorflow as tf
import time

import logger
import financial
import fin_modelbuilder as fmb
import coinscraper
import pickle


def welcome():
    print
    print '****************************************'
    print '*                                      *'
    print '*  F.INANCIAL I.MAGERY S.CIENCE T.OOL  *'
    print '*                                      *'
    print '****************************************'
    print
    print ' welcome to the F.I.S.T. - please enjoy '
    print

def usage(command):
    print 'valid commands include %s'%command

def save():
    print 'saving program state...'
    sys.exit()

def get_main_option():
    option = raw_input('what can i do for you (to quit enter Q): ')
    if option == 'Q':
        save()
    return option

def imagify_data():
    valid_data_types = ['Q', 'stock', 'crypto']
    data_type = raw_input('what type of data can i imagify for you: ')
    if data_type == 'Q':
        save()
    while data_type not in valid_data_types:
        usage(valid_data_types)
        imagify_data()
    if data_type == 'stock':
        return imagify_stock_data()
    if data_type == 'crypto':
        return imagify_crypto_data()

def imagify_stock_data():
    symbol = raw_input('which symbol can i imagify for you: ')
    if symbol == 'Q':
        save()
    filename = symbol.upper() + '.csv'
    while not os.path.exists('data/' + filename):
        print filename
        usage(os.listdir('data'))
        imagify_stock_data()
    return financial.Finance(filename, symbol.upper())    

def imagify_crypto_data():
    symbol = raw_input('which symbol can i imagify for you: ')
    options = os.listdir('data')
    options.append('other')
    if symbol == 'Q':
        save()
    elif symbol == 'other':
        filename = raw_input('which file would you like to read: ')
    filename = symbol.upper() + '_scrape.csv'
    while not os.path.exists('data/' + filename):
        usage(options)
        imagify_stock_data()
    return financial.Finance(filename, symbol.upper())    

def image_neural_net():
    files = []
    options = os.listdir('input')
    options.append('done')
    symbol = raw_input('which symbol would you like to load (BTC): ').upper()
    while symbol != 'DONE':
        while not os.path.exists('input/' + symbol + 'images.pkl') and symbol != 'DONE':
            usage(options)
            symbol = ask_question('which symbol would you like to load (BTC): ').upper()
        if symbol == 'DONE':
            pass
        else:
            print('adding %s to files'%symbol)
            files.append(symbol)
            symbol = raw_input('which symbol would you like to add (or done to continue): ').upper()
    print('processing files %s'%files)
    while 'DONE' in files:
        files.remove('DONE')
    if len(files) < 1:
        return -1
    with open('input/' + files[0]+'images.pkl') as f:
        images = pickle.load(f) 
    images = np.array(images)
    close = raw_input('would you like to use close or high: ')
    if close == 'close':
        ext = 'delta_close.csv'
    else:
        ext = 'delta_high.csv'
    try:
        yvals = np.genfromtxt('input/' +files[0]+ext, delimiter=',')
    except:
        print(symbol, ext)
        sys.exit(1)
    if len(files) > 1:
        for sym in files[1:]:
        #open file 1 from pickle
            with open('input/' + sym+'images.pkl') as f:
                images2 = pickle.load(f) 
                #may want more than one type of file say tech vs fin to help train
                images = np.concatenate((images, np.array(images2)), axis=0)
            yvals2 = np.genfromtxt(sym+ext, delimiter=',')
            yvals = np.concatenate((yvals, yvals2), axis=0)
    #normalize images
    images = images / 255.0
    perc_split = ask_question('what test split would you like (default 10): ')
    while(True):
        try:
            perc_split = float(perc_split)
            break
        except:
            print 'sorry could not convert that to a float'
            n = ask_question('what test split would you like (default 10): ')
    num_buckets = ask_question('how many classification buckets would you like: ')
    while(True):
        try:
            num_buckets = int(num_buckets)
            break
        except:
            print 'sorry could not convert that to an int'
            num_buckets = ask_question('how many classification buckets would you like: ')
    min_val = ask_question('what is the min value for buckets: ')
    while(True):
        try:
            min_val = float(min_val)
            break
        except:
            print 'sorry could not convert that to a float'
            min_val = ask_question('what is the min value for buckets: ')
    max_val = ask_question('what is the max value for buckets: ')
    while(True):
        try:
            max_val = float(max_val)
            break
        except:
            print 'sorry could not convert that to a float'
            max_val = ask_question('what is the max value for buckets: ')
    steps = ask_question('how many training steps: ')
    while(True):
        try:
            steps = int(steps)
            break
        except:
            print 'sorry could not convert that to an int'
            steps = ask_question('how many traiing steps: ')
    #use parameter to set number of training vs test images
    perc_split = int((perc_split/100.00) * len(images))
    #sanity
    assert not np.any(np.isnan(images))
    assert not np.any(np.isnan(yvals))
    #split train test
    train_xvals, test_xvals = images[:perc_split], images[perc_split:]
    #if second file, format y labels for that too
    #yvals is a decimal percent
    yvals = yvals * 100
    print(images.shape, yvals.shape)
    #yvals = yvals.reshape(-1,1)
    if num_buckets < 3:
        buckets = np.array([min_val, max_val])
    else:
        buckets = np.linspace(min_val, max_val, num_buckets-1)
    print(np.max(yvals), np.min(yvals))
    print(buckets)
    #drop each into bucket
    labels = np.digitize(yvals, buckets)
    
    print(np.max(labels), np.min(labels))
    #labels = np.zeros_like(yvals)
    #labels[yvals>0.0] = 1
    name = str(num_buckets) + ext.split('.')[0]
    train_yvals, test_yvals = labels[:perc_split], labels[perc_split:]
    mdl = fmb.ModelBuilder(train_xvals, train_yvals, test_xvals, test_yvals, len(buckets))
    classifier = tf.estimator.Estimator(model_fn=mdl.cnn_model_fn, model_dir=name +'_convnet_model')
    tensors_to_log = {'probabilities' : 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=500)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':train_xvals}, y=train_yvals, batch_size=80, num_epochs=None, shuffle=False)
    classifier.train(input_fn=train_input_fn, steps=steps, hooks=[logging_hook])
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':test_xvals}, y=test_yvals, num_epochs=1, shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(images.shape, yvals.shape)
    print(np.max(labels), np.min(labels))
    print(buckets)
    print(np.max(labels), np.min(labels))
    print(np.max(yvals), np.min(yvals))
    return eval_results, mdl



def scrape_data():
    num_syms = ask_question('how many top symbols shall i scrape (10): ')
    while(True):
        try:
            num_syms = int(num_syms)
            break
        except:
            print 'sorry could not convert that to an integer'
            num_syms = ask_question('how many top symbols shall i scrape: ')
    days = ask_question('how many days shall i scrape for: ')
    while(True):
        try:
            days = int(days)
            break
        except:
            print 'sorry could not convert that to an integer'
            days = ask_question('how many days shall i scrape for: ')
    hours = ask_question('how many hours shall i scrape for: ')
    while(True):
        try:
            hours = int(hours)
            break
        except:
            print 'sorry could not convert that to an integer'
            hours = ask_question('how many hours shall i scrape for: ')
    minutes = ask_question('how many minutes shall i scrape for: ')
    while(True):
        try:
            minutes = int(minutes)
            break
        except:
            print 'sorry could not convert that to an integer'
            minutes = ask_question('how many minutes shall i scrape for: ')
    total_time = days * 24 * 60 * 60 + hours * 60 * 60 + minutes
    elapsed_time= 0
    while(True):
        data = cmc.collect_data(num_syms)
        if elapsed_time%60 == 0:
            print('collecting data for \n' + ' '.join([item['symbol'] for item in data]))
            cmc.output_data(data)
            time.sleep(60)
        elapsed_time += 1
        if elapsed_time > total_time:
            break
            return

def ask_question(msg):
    return raw_input(msg)


if __name__ == '__main__':
    welcome()
    valid_options = ['Q','imagify', 'scrape', 'neuralnet']
    option = get_main_option()
    while option != 'Q':
        if option not in valid_options:
            usage(valid_options)
            option = get_main_option()
            continue
        else:
            #instructions for imagifying option ********************
            if option == 'imagify':
                finance_obj = imagify_data()
                n = ask_question('what size rolling window would you like: ')
                m = ask_question('what lookahead would you like: ')
                while(True):
                    try:
                        n = int(n)
                        m = int(m)
                        break
                    except:
                        print 'sorry could not convert those to an integer'
                        n = ask_question('what size rolling window would you like (default 30): ')
                        m = ask_question('what lookahead would you like: ')
                finance_obj.process_files(n,m)
            #*******************************************************
            
            #instructions for neural net option ********************
            elif option == 'neuralnet':
                image_neural_net()

            #*******************************************************

            #instructions for scraping option ********************
            elif option == 'scrape':
                scrape_data()
            #*******************************************************
            
            else:
                usage(valid_options)
            
            option = get_main_option()
    save()
    
