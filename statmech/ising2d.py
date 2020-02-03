import argparse
import numpy as np
import os
import sys
import tensorflow as atf

from tensorflow.python.framework import dtypes

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
