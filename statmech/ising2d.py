import argparse
import numpy as np
import os
import pickle
import sys
import tensorflow as atf

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import dtypes
from urllib.request import urlopen 

"""
Identifying Phases in the 2D Ising Model with TensorFlow
"""

tf.set_random_seed(seed)
class DataSet(object):
    def __init__(self, data_X, data_Y, dtype=dtypes.float32):
        """
        Checks data and casts it into correct data type. 
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError("Invalid dtype %r, expected uint8 or float32" % dtype)

        assert data_X.shape[0] == data_Y.shape[0], ("data_X.shape: %s data_Y.shape: %s" % (data_X.shape, data_Y.shape))
        self.num_examples = data_X.shape[0]

        if dtype == dtypes.float32:
            data_X = data_X.astype(np.float32)
        self.data_X = data_X
        self.data_Y = data_Y 

        self.epochs_completed = 0
        self.index_in_epoch = 0

    def next_batch(self, batch_size, seed=None):
        """
        Return the next `batch_size` examples from this data set.
        """
        if seed:
            np.random.seed(seed)

        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.data_X = self.data_X[perm]
            self.data_Y = self.data_Y[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch

        return self.data_X[start:end], self.data_Y[start:end]

def load_data():

    # path to data directory (for testing)

    url_main = "https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/"

    ######### LOAD DATA
    # The data consists of 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25):
    data_file_name = "Ising2DFM_reSample_L40_T=All.pkl" 
    # The labels are obtained from the following file:
    label_file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl"

    data = pickle.load(urlopen(url_main + data_file_name)) # pickle reads the file and returns the Python object (1D array, compressed bits)
    data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
    data=data.astype('int')
    data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)

    # labels (convention is 1 for ordered states and 0 for disordered states)
    labels = pickle.load(urlopen(url_main + label_file_name)) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)
    
    print("Finished loading data")
    return data, labels
