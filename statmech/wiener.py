import math
import numpy as np
import statsmodels.api as sm
import xgboost as xgb # For xgboost

from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, GRU, Activation, Dropout
from numpy.linalg import inv as inv # Used in kalman filter
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.stats import norm
from scipy.spatial.distance import cdist
from sklearn import linear_model #For Wiener Filter and Wiener Cascade
from sklearn.svm import SVR #For support vector regression (SVR)

"""
Wiener filter used in signal processing for decoding.
"""

class WienerFilterDecoder(object):
    """
    Class for the Wiener Filter Decoder.
    """
    def __init__(self):
        return

    def fit(self,X_flat_train,y_train):
        """
        Train Wiener Filter Decoder.
        X_flat_train is a numpy 2d array of shape [n_samples,n_features] of the neural data.
        y_train is a numpy 2d array of shape [n_samples, n_outputs] that are the outputs to
        predict.
        """
        self.model=linear_model.LinearRegression() # Initialize linear regression model
        self.model.fit(X_flat_train, y_train) # Train the model

    def predict(self,X_flat_test):
        """
        Predict outcomes using trained Wiener Cascade Decoder.
        X_flat_test is a numpy 2d array of shape [n_samples,n_features] of data used to predict.
        Return y_test_predicted is a numpy 2d array of shape [n_samples,n_outputs] of the 
        predicted output.
        """
        y_test_predicted=self.model.predict(X_flat_test) # Make predictions
        return y_test_predicted
