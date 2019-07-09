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
        model.add(Activation("relu"))) # Add nonlinear (tanh) activation
        # if self.dropout!=0:
        if self.dropout! = 0: model.add(Dropout(self.dropout))  # Dropout some units if proportion of dropout != 0
        # Add any additional hidden layers (beyond the 1st)
        for layer in range(self.num_layers-1): # Loop through additional layers
            model.add(Dense(self.units[layer+1])) # Add dense layer
            model.add(Activation("relu")) # Add nonlinear (tanh) activation
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

class SimpleRNNDecoder(object):
    """
    Simple recurrent neural network decoder.
    """
    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=0):
         self.units = units
         self.dropout = dropout
         self.num_epochs = num_epochs
         self.verbose = verbose

    def fit(self,X_train,y_train):
        """
        Train SimpleRNN Decoder.
        """
        model = Sequential() #Declare model
        # Add recurrent layer
        model.add(SimpleRNN(self.units,input_shape=(X_train.shape[1],X_train.shape[2]), dropout_W=self.dropout, dropout_U=self.dropout,activation="relu")) # Within recurrent layer, include dropout
        if self.dropout!=0: model.add(Dropout(self.dropout)) # Dropout some units (recurrent layer output units)
        # Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))
        # Fit model (and set fitting parameters)
        model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"]) # Set loss function and optimizer
        model.fit(X_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose) # Fit the model
        self.model=model

    def predict(self,X_test):

        """
        Predict outcomes using trained SimpleRNN Decoder/
        """

        y_test_predicted = self.model.predict(X_test) #Make predictions
        return y_test_predicted
    
class GRUDecoder(object):
    """
    Class for the gated recurrent unit (GRU) decoder.
    """
    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=0):
         self.units = units
         self.dropout = dropout
         self.num_epochs = num_epochs
         self.verbose = verbose

    def fit(self,X_train,y_train):
        """
        Train GRU Decoder.
        """
        model = Sequential() # Declare model
        # Add recurrent layer
        model.add(GRU(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout_W=self.dropout,dropout_U=self.dropout)) #W ithin recurrent layer, include dropout
        if self.dropout != 0: model.add(Dropout(self.dropout)) # Dropout some units (recurrent layer output units)
        # Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))
        # Fit model (and set fitting parameters)
        model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"]) # Set loss function and optimizer
        model.fit(X_train, y_train, nb_epoch=self.num_epochs, verbose=self.verbose) # Fit the model
        self.model = model

    def predict(self,X_test):
        """
        Predict outcomes using trained GRU Decoder.
        """
        y_test_predicted = self.model.predict(X_test) # Make predictions

class LSTMDecoder(object):

    """
    Class for the gated recurrent unit (GRU) decoder. Long short term memory (LSTM).
    """

    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=0):
         self.units = units
         self.dropout = dropout
         self.num_epochs = num_epochs
         self.verbose = verbose

    def fit(self,X_train,y_train):
        """
        Train LSTM Decoder.
        """
        model = Sequential() #Declare model
        #Add recurrent layer
        model.add(LSTM(self.units,input_shape=(X_train.shape[1], X_train.shape[2]), dropout_W=self.dropout, dropout_U=self.dropout)) #Within recurrent layer, include dropout
        if self.dropout != 0: model.add(Dropout(self.dropout)) #Dropout some units (recurrent layer output units)
        # Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))
        # Fit model (and set fitting parameters)
        model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"]) # Set loss function and optimizer
        model.fit(X_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose) # Fit the model
        self.model=model

    def predict(self,X_test):
        """
        Predict outcomes using trained LSTM Decoder.
        """
        y_test_predicted = self.model.predict(X_test) # Make predictions
        return y_test_predicted

class XGBoostDecoder(object):
    """
    Class for the XGBoost Decoder.
    """
    def __init__(self,max_depth=3,num_round=300,eta=0.3,gpu=-1):
        self.max_depth = max_depth
        self.num_round = num_round
        self.eta = eta
        self.gpu = gpu

    def fit(self,X_flat_train,y_train):
        """
        Train XGBoost Decoder.
        """
        num_outputs = y_train.shape[1] # Number of outputs
        # Set parameters for XGBoost
        param = {"objective": "reg:linear", # for linear output
            "eval_metric": "logloss", # loglikelihood loss
            "max_depth": self.max_depth, # this is the only parameter we have set, it's one of the way or regularizing
            "eta": self.eta,
            "seed": 2925, # for reproducibility
            "silent": 1}
        if self.gpu<0:
            param["nthread"] = -1 # with -1 it will use all available threads
        else:
            param["gpu_id"] = self.gpu
            param["updater"] = "grow_gpu"
        models=[] # Initialize list of models (there will be a separate model for each output)
        for y_idx in range(num_outputs): #Loop through outputs
            dtrain = xgb.DMatrix(X_flat_train, label=y_train[:,y_idx]) # Put in correct format for XGB
            bst = xgb.train(param, dtrain, self.num_round) # Train model
            models.append(bst) # Add fit model to list of models
        self.model=models

    def predict(self,X_flat_test):
        """
        Predict outcomes using trained XGBoost Decoder.
        """
        dtest = xgb.DMatrix(X_flat_test) # Put in XGB format
        num_outputs=len(self.model) # Number of outputs
        y_test_predicted=np.empty([X_flat_test.shape[0],num_outputs])# Initialize matrix of predicted outputs
        for y_idx in range(num_outputs): # Loop through outputs
            bst=self.model[y_idx] # Get fit model for this output
            y_test_predicted[:,y_idx] = bst.predict(dtest) # Make prediction
        return y_test_predicted

class SVRDecoder(object):
    """
    Class for the Support Vector Regression (SVR) Decoder.
    """
    def __init__(self,max_iter=-1,C=3.0):
        self.max_iter = max_iter
        self.C = C
        return

    def fit(self,X_flat_train,y_train):
        """
        Train SVR Decoder.
        """
        num_outputs = y_train.shape[1] # Number of outputs
        models=[] # Initialize list of models (there will be a separate model for each output)
        for y_idx in range(num_outputs): # Loop through outputs
            model = SVR(C=self.C, max_iter=self.max_iter) # Initialize SVR model
            model.fit(X_flat_train, y_train[:,y_idx]) # Train the model
            models.append(model) # Add fit model to list of models
        self.model = models

    def predict(self,X_flat_test):
        """
        Predict outcomes using trained Wiener Cascade Decoder.
        """
        num_outputs = len(self.model) # Number of outputs
        y_test_predicted = np.empty([X_flat_test.shape[0],num_outputs]) # Initialize matrix of predicted outputs
        for y_idx in range(num_outputs): # Loop through outputs
            model = self.model[y_idx] # Get fit model for that output
            y_test_predicted[:,y_idx] = model.predict(X_flat_test) # Make predictions
        return y_test_predicted

def glm_run(Xr, Yr, X_range):
    """
    GLM helper function for the NaiveBayesDecoder.
    """
    X2 = sm.add_constant(Xr)
    poiss_model = sm.GLM(Yr, X2, family=sm.families.Poisson())
    try:
        glm_results = poiss_model.fit()
        Y_range= glm_results.predict(sm.add_constant(X_range))
    except np.linalg.LinAlgError:
        print("\nWARNING: LinAlgError")
        Y_range = np.mean(Yr)*np.ones([X_range.shape[0],1])
    return Y_range

class NaiveBayesDecoder(object):

    """
    Class for the Naive Bayes Decoder.
    encoding_model is a string, default='quadratic' of which encoding model to use.
    res is the number of bins to divide the outputs into (going from minimum to maximum).
    """
    def __init__(self,encoding_model='quadratic',res=100):
        self.encoding_model = encoding_model
        self.res = res
        return

    def fit(self,X_b_train,y_train):
        """
        Train Naive Bayes Decoder.
        """
        # Fit tuning curve
        input_x_range=np.arange(np.min(y_train[:,0]),np.max(y_train[:,0])+.01,np.round((np.max(y_train[:,0])-np.min(y_train[:,0]))/self.res))
        input_y_range=np.arange(np.min(y_train[:,1]),np.max(y_train[:,1])+.01,np.round((np.max(y_train[:,1])-np.min(y_train[:,1]))/self.res))
        # Get all combinations of x/y values
        input_mat = np.meshgrid(input_x_range,input_y_range)
        xs = np.reshape(input_mat[0],[input_x_range.shape[0]*input_y_range.shape[0],1])
        ys = np.reshape(input_mat[1],[input_x_range.shape[0]*input_y_range.shape[0],1])
        input_xy=np.concatenate((xs,ys),axis=1)
        if self.encoding_model=='quadratic':
            input_xy_modified = np.empty([input_xy.shape[0],5])
            input_xy_modified[:,0] = input_xy[:,0]**2
            input_xy_modified[:,1] = input_xy[:,0]
            input_xy_modified[:,2] = input_xy[:,1]**2
            input_xy_modified[:,3] = input_xy[:,1]
            input_xy_modified[:,4] = input_xy[:,0]*input_xy[:,1]
            y_train_modified = np.empty([y_train.shape[0],5])
            y_train_modified[:,0] = y_train[:,0]**2
            y_train_modified[:,1] = y_train[:,0]
            y_train_modified[:,2] = y_train[:,1]**2
            y_train_modified[:,3] = y_train[:,1]
            y_train_modified[:,4] = y_train[:,0]*y_train[:,1]
        # Create tuning curves
        num_nrns=X_b_train.shape[1] # Number of neurons to fit tuning curves for
        tuning_all=np.zeros([num_nrns,input_xy.shape[0]]) # Matrix that stores tuning curves for all neurons
        # Loop through neurons and fit tuning curves
        for j in range(num_nrns): #Neuron number
            if self.encoding_model == "linear":
                tuning=glm_run(y_train,X_b_train[:,j:j+1],input_xy)
            if self.encoding_model == "quadratic":
                tuning=glm_run(y_train_modified,X_b_train[:,j:j+1],input_xy_modified)
            tuning_all[j,:] = np.squeeze(tuning)
        # Save tuning curves to be used in "predict" function
        self.tuning_all = tuning_all
        self.input_xy = input_xy
        n = y_train.shape[0]
        dx = np.zeros([n-1,1])
        for i in range(n-1):
            dx[i] = np.sqrt((y_train[i+1,0]-y_train[i,0])**2+(y_train[i+1,1]-y_train[i,1])**2) # Change in state across time steps
        std = np.sqrt(np.mean(dx**2)) # dx is only positive. this gets approximate stdev of distribution (if it was positive and negative)
        self.std = std #Save for use in "predict" function

    def predict(self,X_b_test,y_test):
        """
        Predict outcomes using trained tuning curves.
        """
        tuning_all=self.tuning_all
        input_xy=self.input_xy
        std=self.std
        dists = squareform(pdist(input_xy, "euclidean")) 
        prob_dists = norm.pdf(dists,0,std)
        loc_idx = np.argmin(cdist(y_test[0:1,:],input_xy)) 
        num_nrns = tuning_all.shape[0]
        y_test_predicted = np.empty([X_b_test.shape[0],2]) 
        num_ts = X_b_test.shape[0] 
        for t in range(num_ts):
            rs=X_b_test[t,:] 
            probs_total=np.ones([tuning_all[0,:].shape[0]])
            for j in range(num_nrns):
                lam = np.copy(tuning_all[j,:]) 
                probs = np.exp(-lam)*lam**r/math.factorial(r) 
                probs_total = np.copy(probs_total*probs)
            prob_dists_vec = np.copy(prob_dists[loc_idx,:]) 
            probs_final = probs_total*prob_dists_vec 
            loc_idx = np.argmax(probs_final)
            y_test_predicted[t,:] = input_xy[loc_idx,:]
        return y_test_predicted 