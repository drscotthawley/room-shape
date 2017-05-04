#! /usr/bin/env python
# Author: Scott Hawley

# Can the system learn to "invert" the Rayleigh equation?
# Given a subset of room mode frequencies, can we learn the dimensions of the room?
# (Also we may give it the volume of the room, since volume is proportional to reverb time)

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import ProgbarLogger, ModelCheckpoint, ReduceLROnPlateau, Callback, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU
from os.path import isfile
import random


mydtype = np.float32


def gen_data(N,subset_size=20, invert_dims=False):
    # Inputs:
    #    N  = number of "rooms" to generate
    #    subset_size = how many frequencies for each room to use
    max_nx = 5                      # highest harmonic number to use
    vs = 1130                       # speed of sound in ft/s
    minsize, maxsize = 5, 50        # size range of rooms to generate
    subset_size = (max_nx)**3                # grab this many modes, currently we take all of them
    X = np.zeros([N,subset_size],dtype=mydtype)   # array to hold mode freqs as inputs to NN
    Y = np.zeros([N,3],dtype=mydtype)               # target room dimensions

    indexes = np.array(list(range( (max_nx+1)**3)), dtype=mydtype)
    nx = np.floor( indexes / ((max_nx+1)**2))           # [0,0,0,0,1,1,1,1]
    ny = np.floor(indexes / (max_nx+1)) % (max_nx+1)    # [0,0,1,1,0,0,1,1]
    nz = indexes % (max_nx+1)                           # [0,1,0,1,0,1,0,1]

    for i in range(N):              # create (partial) list of mode frequencies for N rooms
        dims = np.random.uniform(low=minsize,high=maxsize,size=3)
        dims.sort()                 # just to try to avoid redunancy/confusion in training
        
        ''' The following is more legible, but slower:
        #freqs = []
        #for nx in range(max_nx+1):
        #    for ny in range(max_nx+1):
        #        for nz in range(max_nx+1):  
        #            if (nx+ny+nz > 0):          # zero frequency is too easy ;-)
        #                f = vs/2*np.sqrt( (nx/dims[0])**2 + (ny/dims[1])**2 + (nz/dims[2])**2)
        #                freqs.append(f)
        '''
        # The following is faster that the above:
        freqs = vs/2*np.sqrt( (nx/dims[0])**2 + (ny/dims[1])**2 + (nz/dims[2])**2)


        rand_sample = [ freqs[i] for i in sorted(random.sample(range(len(freqs)), subset_size)) ]
        #rand_sample.sort()        # doesn't matter / doesn't really help # maybe this is cheating, but let's help the network
        rand_sample = np.array(rand_sample,dtype=mydtype)
        #rand_sample = vs/(rand_sample+1e-6)

        X[i,:] = rand_sample

        #X[i,-1] =  dims[0]*dims[1]*dims[2]    # Make it easier: give it the room volume too

        if (invert_dims):
            Y[i,:] = 1/np.array(dims,dtype=mydtype)   # give it 1/lengths
        else:
            Y[i,:] = np.array(dims,dtype=mydtype)   

    return X, Y
       

def make_model(X, Y, n_hidden, weights_file, n_layers=7, dropout_fac=0.2):

    if ( isfile(weights_file) ):
        print ('Weights file detected. Loading from ',weights_file)
        model = load_model(weights_file)
    else:
        print('No weights file detected, so starting from scratch.')

        # create and fit Multilayer Perceptron model
        model = Sequential()
        model.add(Dense(n_hidden, input_shape=(X.shape[1],), kernel_initializer="he_uniform"))
        model.add(BatchNormalization(axis=1))
        model.add(ELU(alpha=1.0))  
        model.add(Dropout(dropout_fac))

        for l in range(n_layers-1):
            model.add(Dense( int(n_hidden/(2**(l))) ))
            model.add(BatchNormalization(axis=1))
            model.add(ELU(alpha=1.0))  
            model.add(Dropout(dropout_fac))

        model.add(Dense(Y.shape[1]))
        model.compile(loss='mse', optimizer='nadam') #, metrics=['accuracy'])
    model.summary()
    return model

def calc_mse(Y_test,Y_pred):
    return ((Y_test - Y_pred)**2).mean()


def test_predict(model,X_test,Y_test,invert_dims=False):
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"

    print("\n Predicting....  Sample results:   (invert_dims =",invert_dims,")")

    Y_pred = model.predict(X_test)

    if (invert_dims):               # invert back
        test_data = 1/Y_test
        pred_data = 1/Y_pred
    else:
        test_data = Y_test
        pred_data = Y_pred

   
    for i in range(5):#     Y_pred.shape[0]):
        print("   test_data[",i,"] = ",test_data[i],", pred_data[",i,"] = ",pred_data[i],sep="")
        #print(GREEN,"          1/test_data[",i,"] = ",1/test_data[i],", 1/pred_data[",i,"] = ",
        #    1/pred_data[i],RESET,sep="")
        #score = model.evaluate(X_test, Y_test, verbose=False) 
        score = calc_mse(test_data,pred_data)

    print('Overall test score: mse loss: ',score)    #Loss on test
    #print('Test accuracy: ', score[1])
    print("")
    return



def main():
    np.random.seed(2)

    # parameters for 'size' of run
    n_hidden = 200
    n_layers=7
    dropout_fac = 0
    batch_size = 100
    n_train = 300000 
    n_val = 20000
    n_test =10000
    grab_modes = 100    # take a subsample of this many modes from list of frequencies
    invert_dims = False   # 22.1958 with MAPE and True,  13.1486 False

    print("Generating Data...")
    print("   ...Testing")
    X_test, Y_test = gen_data(n_test,subset_size=grab_modes, invert_dims=invert_dims)
    print("   ...Validation")
    X_val, Y_val = gen_data(n_val,subset_size=grab_modes, invert_dims=invert_dims)
    print("   ...Training")
    X_train, Y_train = gen_data(n_train,subset_size=grab_modes, invert_dims=invert_dims)

    weights_file = "weights.hdf5"
    model = make_model(X_train, Y_train, n_hidden, weights_file, n_layers=n_layers, dropout_fac=dropout_fac)

    # callbacks
    checkpoint = ModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.2, patience=4, min_lr=0.0001)
    earlystop = EarlyStopping(patience=6)
    class testing_callback(Callback): 
        def on_epoch_end(self, epoch, logs={}):
            test_predict(model,X_test,Y_test,invert_dims=invert_dims)
            return
    testing_cb = testing_callback()

    # Training Loop
    n_epochs= 1000
    model.fit(X_train, Y_train, epochs=n_epochs, batch_size=batch_size, verbose=1, validation_data=(X_val, Y_val), 
            callbacks =[checkpoint, earlystop, testing_cb, reduce_lr, ProgbarLogger()])


if __name__ == '__main__':
    main()
