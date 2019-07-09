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

class WienerCascadeDecoder(object):
    """
    Class for the Wiener Cascade Decoder.
    degree is the polynomial degree used for the static nonlinearity
    """
    def __init__(self,degree=3):
         self.degree=degree

    def fit(self,X_flat_train,y_train):
        """
        Train Wiener Cascade Decoder.
        """
        num_outputs=y_train.shape[1] # Number of outputs
        models=[] #Initialize list of models (there will be a separate model for each output)
        for i in range(num_outputs): # Loop through outputs
            #Fit linear portion of model
            regr = linear_model.LinearRegression() # Call the linear portion of the model "regr"
            regr.fit(X_flat_train, y_train[:,i]) # Fit linear
            y_train_predicted_linear=regr.predict(X_flat_train) # Get outputs of linear portion of model
            #Fit nonlinear portion of model
            p=np.polyfit(y_train_predicted_linear,y_train[:,i],self.degree)
            #Add model for this output (both linear and nonlinear parts) to the list "models"
            models.append([regr,p])
        self.model=models

    def predict(self,X_flat_test):
        """
        Predict outcomes using trained Wiener Cascade Decoder.
        """
        num_outputs=len(self.model) # Number of outputs being predicted. Recall from the "fit" function that self.model is a list of models
        y_test_predicted=np.empty([X_flat_test.shape[0],num_outputs]) # Initialize matrix that contains predicted outputs
        for i in range(num_outputs): # Loop through outputs
            [regr,p]=self.model[i] # Get the linear (regr) and nonlinear (p) portions of the trained model
            # Predictions on test set
            y_test_predicted_linear=regr.predict(X_flat_test) # Get predictions on the linear portion of the model
            y_test_predicted[:,i]=np.polyval(p,y_test_predicted_linear) # Run the linear predictions through the nonlinearity to get the final predictions
        return y_test_predicted

