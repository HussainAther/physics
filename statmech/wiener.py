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
        self.model = linear_model.LinearRegression() # Initialize linear regression model
        self.model.fit(X_flat_train, y_train) # Train the model

    def predict(self,X_flat_test):
        """
        Predict outcomes using trained Wiener Cascade Decoder.
        X_flat_test is a numpy 2d array of shape [n_samples,n_features] of data used to predict.
        Return y_test_predicted is a numpy 2d array of shape [n_samples,n_outputs] of the 
        predicted output.
        """
        y_test_predicted = self.model.predict(X_flat_test) # Make predictions
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
        num_outputs = y_train.shape[1] # Number of outputs
        models = [] #Initialize list of models (there will be a separate model for each output)
        for i in range(num_outputs): # Loop through outputs
            #Fit linear portion of model
            regr = linear_model.LinearRegression() # Call the linear portion of the model "regr"
            regr.fit(X_flat_train, y_train[:,i]) # Fit linear
            y_train_predicted_linear=regr.predict(X_flat_train) # Get outputs of linear portion of model
            #Fit nonlinear portion of model
            p = np.polyfit(y_train_predicted_linear,y_train[:,i],self.degree)
            #Add model for this output (both linear and nonlinear parts) to the list "models"
            models.append([regr,p])
        self.model = models

    def predict(self,X_flat_test):
        """
        Predict outcomes using trained Wiener Cascade Decoder.
        """
        num_outputs = len(self.model) # Number of outputs being predicted. Recall from the "fit" function that self.model is a list of models
        y_test_predicted=np.empty([X_flat_test.shape[0],num_outputs]) # Initialize matrix that contains predicted outputs
        for i in range(num_outputs): # Loop through outputs
            [regr,p]=self.model[i] # Get the linear (regr) and nonlinear (p) portions of the trained model
            # Predictions on test set
            y_test_predicted_linear = regr.predict(X_flat_test) # Get predictions on the linear portion of the model
            y_test_predicted[:,i] = np.polyval(p,y_test_predicted_linear) # Run the linear predictions through the nonlinearity to get the final predictions
        return y_test_predicted

class KalmanFilterDecoder(object):
    """
    Class for the Kalman Filter Decoder. C is an optional float parameter to scale hte noise matrix
    of hte transition in kinematic sstates. 
    """

    def __init__(self,C=1):
        self.C=C


    def fit(self,X_kf_train,y_train):
        """
        Train Kalman Filter Decoder.
        """
        # First we'll rename and reformat the variables to be in a more standard kalman filter nomenclature.
        # xs are the state (here, the variable we're predicting, i.e. y_train)
        # zs are the observed variable (neural data here, i.e. X_kf_train)
        X = np.matrix(y_train.T)
        Z = np.matrix(X_kf_train.T)
        # Number of time bins
        nt=X.shape[1]
        # Calculate the transition matrix (from x_t to x_t+1) using least-squares, and compute its covariance
        # In our case, this is the transition from one kinematic state to the next
        X2 = X[:,1:]
        X1 = X[:,0:nt-1]
        A=X2*X1.T*inv(X1*X1.T) # Transition matrix
        W=(X2-A*X1)*(X2-A*X1).T/(nt-1)/self.C #C ovariance of transition matrix. Note we divide by nt-1 since only nt-1 points were used in the computation (that's the length of X1 and X2). We also introduce the extra parameter C here.
        # Calculate the measurement matrix (from x_t to z_t) using least-squares, and compute its covariance
        # In our case, this is the transformation from kinematics to spikes
        H = Z*X.T*(inv(X*X.T)) # Measurement matrix
        Q = ((Z - H*X)*((Z - H*X).T)) / nt # Covariance of measurement matrix
        params=[A,W,H,Q]
        self.model=params

    def predict(self,X_kf_test,y_test):
        """
        Predict outcomes using trained Kalman Filter Decoder.
        """
        # Extract parameters
        A,W,H,Q = self.model
        X = np.matrix(y_test.T)
        Z = np.matrix(X_kf_test.T)
        num_states = X.shape[0] # Dimensionality of the state
        states = np.empty(X.shape) # Keep track of states over time (states is what will be returned as y_test_predicted)
        P_m = np.matrix(np.zeros([num_states,num_states]))
        P = np.matrix(np.zeros([num_states,num_states]))
        state=X[:,0] # Initial state
        states[:,0] = np.copy(np.squeeze(state))
        # Get predicted state for every time bin
        for t in range(X.shape[1]-1):
            P_m = A*P*A.T+W
            state_m = A*state
            K = P_m*H.T*inv(H*P_m*H.T+Q) # Calculate Kalman gain
            P = (np.matrix(np.eye(num_states))-K*H)*P_m
            state = state_m+K*(Z[:,t+1]-H*state_m)
            states[:,t+1] = np.squeeze(state) # Record state at the timestep
        y_test_predicted = states.T
        return y_test_predicted

class DenseNNDecoder(object):
    """
    Class for the dense (fully-connected) neural network decoder.
    units is the number of hidden units in each layer.
    dropout is the proportion of units that get dropped out.
    num_epochs is the number of epochs used for training.
    verbose is whether to show progress of the fit after each epoch.
    """
    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=0):
         self.dropout = dropout
         self.num_epochs = num_epochs
         self.verbose = verbose
         # If "units" is an integer, put it in the form of a vector
         try: #Check if it's a vector
             units[0]
         except: #If it's not a vector, create a vector of the number of units for each layer
             units = [units]
         self.units = units
         # Determine the number of hidden layers (based on "units" that the user entered)
         self.num_layers = len(units)

    def fit(self,X_flat_train,y_train):
        """
        Train DenseNN Decoder.
        """
        model = Sequential() # Declare model
        # Add first hidden layer
        model.add(Dense(self.units[0],input_dim = X_flat_train.shape[1])) # Add dense layer
        model.add(Activation("Relu"))) # Add nonlinear (tanh) activation
        # if self.dropout!=0:
        if self.dropout! = 0: model.add(Dropout(self.dropout))  # Dropout some units if proportion of dropout != 0
        # Add any additional hidden layers (beyond the 1st)
        for layer in range(self.num_layers-1): # Loop through additional layers
            model.add(Dense(self.units[layer+1])) # Add dense layer
            model.add(Activation("Relu")) # Add nonlinear (tanh) activation
            if self.dropout! = 0: model.add(Dropout(self.dropout)) # Dropout some units if proportion of dropout != 0
        # Add dense connections to all outputs
        model.add(Dense(y_train.shape[1])) # Add final dense layer (connected to outputs)
        # Fit model (and set fitting parameters)
        model.compile(loss="mse", optimizer="adam", metrics=["accuracy"]) # Set loss function and optimizer
        model.fit(X_flat_train, y_train, nb_epoch=self.num_epochs, verbose=self.verbose) #F it the model
        self.model = model

    def predict(self,X_flat_test):
        """
        Predict outcomes using trained DenseNN Decoder.
        """
        y_test_predicted = self.model.predict(X_flat_test) # Make predictions
        return y_test_predicted
