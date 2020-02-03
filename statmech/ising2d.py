import argparse
import matplotlib.pyplot as plt
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
Deep neural network
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
    """
    Load the data.
    """
    url_main = "https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/"

    # The data consists of 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25):
    data_file_name = "Ising2DFM_reSample_L40_T=All.pkl" 
    # The labels are obtained from the following file:
    label_file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl"

    data = pickle.load(urlopen(url_main + data_file_name)) # pickle reads the file and returns the Python object (1D array, compressed bits)
    data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
    data=data.astype("int")
    data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)

    # labels (convention is 1 for ordered states and 0 for disordered states)
    labels = pickle.load(urlopen(url_main + label_file_name)) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)
    
    print("Finished loading data")
    return data, labels

def prepare_data(data, labels, dtype=dtypes.float32, test_size=0.2, validation_size=5000):
    
    L=40 # linear system size

    # divide data into ordered, critical and disordered
    X_ordered=data[:70000,:]
    Y_ordered=labels[:70000]

    X_critical=data[70000:100000,:]
    Y_critical=labels[70000:100000]

    X_disordered=data[100000:,:]
    Y_disordered=labels[100000:]

    # Define training and test data sets.
    X=np.concatenate((X_ordered,X_disordered)) #np.concatenate((X_ordered,X_critical,X_disordered))
    Y=np.concatenate((Y_ordered,Y_disordered)) #np.concatenate((Y_ordered,Y_critical,Y_disordered))

    # Pick random data points from ordered and disordered states to create the training and test sets.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, train_size=1.0-test_size)

    # Make data categorical (i.e [0,1] or [1,0]).
    Y_train=to_categorical(Y_train)
    Y_test=to_categorical(Y_test)
    Y_critical=to_categorical(Y_critical)

    if not 0 <= validation_size <= len(X_train):
        raise ValueError("Validation size should be between 0 and {}. Received: {}.".format(len(X_train), validation_size))

    X_validation = X_train[:validation_size]
    Y_validation = Y_train[:validation_size]
    X_train = X_train[validation_size:]
    Y_train = Y_train[validation_size:]

    # Create data sets.
    dataset = {
        "train" : DataSet(X_train, Y_train),
        "test" : DataSet(X_test, Y_test),
        "critical" : DataSet(X_critical, Y_critical),
        "validation" : DataSet(X_validation, Y_validation)
    }

    return dataset

def prepare_Ising_DNN():
    data, labels = load_data()
    return prepare_data(data, labels, test_size=0.2, validation_size=5000)

class model(object):
    def __init__(self, N_neurons, opt_kwargs):
        """
        Builds the TFlow graph for the DNN.

        N_neurons: number of neurons in the hidden layer
        opt_kwargs: optimizer"s arguments
        """ 
        # Define global step for checkpointing.
        self.global_step=tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

        self.L=40 # system linear size
        self.n_feats=self.L**2 # 40x40 square lattice
        self.n_categories=2 # 2 Ising phases: ordered and disordered

        # Create placeholders for input X and label Y.
        self.create_placeholders()
        # Create weight and bias, initialized to 0 and construct DNN to predict Y from X.
        self.deep_layer_neurons=N_neurons
        self.create_DNN()
        # Define loss function.
        self.create_loss()
        # Use gradient descent to minimize loss.
        self.create_optimiser(opt_kwargs)
        # Create accuracy.
        self.create_accuracy()

    def create_placeholders(self):
        with tf.name_scope("data"):
            # input layer
            self.X=tf.placeholder(tf.float32, shape=(None, self.n_feats), name="X_data")
            # target
            self.Y=tf.placeholder(tf.float32, shape=(None, self.n_categories), name="Y_data")
            # p
            self.dropout_keepprob=tf.placeholder(tf.float32, name="keep_probability")

    def _weight_variable(self, shape, name="", dtype=tf.float32):
        """
         Generate a weight variable of a given shape.
         """
        # weights are drawn from a normal distribution with std 0.1 and mean 0.
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, dtype=dtype, name=name)

    def _bias_variable(self, shape, name="", dtype=tf.float32):
        """
        Generate a bias variable of a given shape.
        """
        initial = tf.constant(0.1, shape=shape) 
        return tf.Variable(initial, dtype=dtype, name=name)

    def create_DNN(self):
        with tf.name_scope("DNN"):

            # Fully connected layer
            W_fc1 = self._weight_variable([self.n_feats, self.deep_layer_neurons],name="fc1",dtype=tf.float32)
            b_fc1 = self._bias_variable([self.deep_layer_neurons],name="fc1",dtype=tf.float32)

            a_fc1 = tf.nn.relu(tf.matmul(self.X, W_fc1) + b_fc1)

            # Softmax layer (see loss function)
            W_fc2 = self._weight_variable([self.deep_layer_neurons, self.n_categories],name="fc2",dtype=tf.float32)
            b_fc2 = self._bias_variable([self.n_categories],name="fc2",dtype=tf.float32)
        
            self.Y_predicted = tf.matmul(a_fc1, W_fc2) + b_fc2

    def create_loss(self):
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=self.Y_predicted)
                        )

    def create_optimiser(self,opt_kwargs):
        with tf.name_scope("optimiser"):
            self.optimizer = tf.train.GradientDescentOptimizer(**opt_kwargs).minimize(self.loss,global_step=self.global_step) 
    def create_accuracy(self):
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.Y_predicted, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float64) # change data type
            self.accuracy = tf.reduce_mean(correct_prediction)

def evaluate_model(neurons, lr, Ising_Data, verbose):
    """
    This function trains a DNN to solve the Ising classification problem

    neurons: number of hidden neurons
    lr: SGD learning rate
    Ising_Data: Ising data set
    verbose (bool): toggles output during the calculation 
    """
    training_epochs=100
    batch_size=100

    # SGD learning params
    opt_params=dict(learning_rate=lr)

    # create DNN
    DNN=model(neurons,opt_params)
    with tf.Session() as sess:

        # Initialize the necessary variables, in this case, w and b.
        sess.run(tf.global_variables_initializer())

        # Train the DNN.
        for epoch in range(training_epochs): 

            batch_X, batch_Y = Ising_Data["train"].next_batch(batch_size,seed=seed)

            loss_batch, _ = sess.run([DNN.loss,DNN.optimizer], 
                                    feed_dict={DNN.X: batch_X,
                                               DNN.Y: batch_Y, 
                                               DNN.dropout_keepprob: 0.5} )
            accuracy = sess.run(DNN.accuracy, 
                                feed_dict={DNN.X: batch_X,
                                           DNN.Y: batch_Y, 
                                           DNN.dropout_keepprob: 1.0} )
            # Count training step.
            step = sess.run(DNN.global_step)
        # Test DNN performance on entire train test and critical data sets.
        train_loss, train_accuracy = sess.run([DNN.loss, DNN.accuracy], 
                                                    feed_dict={DNN.X: Ising_Data["train"].data_X,
                                                               DNN.Y: Ising_Data["train"].data_Y,
                                                               DNN.dropout_keepprob: 0.5}
                                                                )
        if verbose: print("train loss/accuracy:", train_loss, train_accuracy)

        test_loss, test_accuracy = sess.run([DNN.loss, DNN.accuracy], 
                                                    feed_dict={DNN.X: Ising_Data["test"].data_X,
                                                               DNN.Y: Ising_Data["test"].data_Y,
                                                               DNN.dropout_keepprob: 1.0}
                                                               )

        if verbose: print("test loss/accuracy:", test_loss, test_accuracy)

        critical_loss, critical_accuracy = sess.run([DNN.loss, DNN.accuracy], 
                                                    feed_dict={DNN.X: Ising_Data["critical"].data_X,
                                                               DNN.Y: Ising_Data["critical"].data_Y,
                                                               DNN.dropout_keepprob: 1.0}
                                                               )
        if verbose: print("crtitical loss/accuracy:", critical_loss, critical_accuracy)

        return train_loss,train_accuracy,test_loss,test_accuracy,critical_loss,critical_accuracy 

def grid_search(verbose):
    """
    This function performs a grid search over a set of different learning rates 
    and a number of hidden layer neurons.
    """

    # Load Ising data.
    Ising_Data = prepare_Ising_DNN()

    # Perform grid search over learning rate and number of hidden neurons.
    N_neurons=np.logspace(0,3,4).astype("int") # Check number of neurons over multiple decades.
    learning_rates=np.logspace(-6,-1,6)

    # Pre-allocate variables to store accuracy and loss data.
    train_loss=np.zeros((len(N_neurons),len(learning_rates)),dtype=np.float64)
    train_accuracy=np.zeros_like(train_loss)
    test_loss=np.zeros_like(train_loss)
    test_accuracy=np.zeros_like(train_loss)
    critical_loss=np.zeros_like(train_loss)
    critical_accuracy=np.zeros_like(train_loss)

    # Grid search.
    for i, neurons in enumerate(N_neurons):
        for j, lr in enumerate(learning_rates):

            print("training DNN with %4d neurons and SGD lr=%0.6f." %(neurons,lr) )

            train_loss[i,j],train_accuracy[i,j],\
            test_loss[i,j],test_accuracy[i,j],\
            critical_loss[i,j],critical_accuracy[i,j] = evaluate_model(neurons,lr,Ising_Data,verbose)

    plot_data(learning_rates,N_neurons,train_accuracy, "training")
    plot_data(learning_rates,N_neurons,test_accuracy, "testing")
    plot_data(learning_rates,N_neurons,critical_accuracy, "critical")

def plot_data(x,y,data,title=None):

    # Plot results.
    fontsize=16


    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation="nearest", vmin=0, vmax=1)
    
    cbar=fig.colorbar(cax)
    cbar.ax.set_ylabel("accuracy (%)",rotation=90,fontsize=fontsize)
    cbar.set_ticks([0,.2,.4,0.6,0.8,1.0])
    cbar.set_ticklabels(["0%","20%","40%","60%","80%","100%"])

    # Put text on matrix elements.
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            c = "${0:.1f}\\%$".format( 100*data[j,i])  
            ax.text(x_val, y_val, c, va="center", ha="center")

    # Convert axis vaues to to string labels.
    x=[str(i) for i in x]
    y=[str(i) for i in y]

    ax.set_xticklabels([""]+x)
    ax.set_yticklabels([""]+y)

    ax.set_xlabel("$\\mathrm{learning\\ rate}$",fontsize=fontsize)
    ax.set_ylabel("$\\mathrm{hidden\\ neurons}$",fontsize=fontsize)
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    plt.show()

# Run it.
verbose=False
grid_search(verbose)

"""
Create convolutional neural network.
"""
class model(object):
    # Build the graph for the CNN.
    def __init__(self,opt_kwargs):

        # Define global step for checkpointing.
        self.global_step=tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

        self.L=40
        self.n_feats= self.L**2 
        self.n_categories=2

        # Create placeholders for input X and label Y.
        self.create_placeholders()
        # Create weight and bias, initialized to 0 and construct CNN to predict Y from X.
        self.create_CNN()
        # Define loss function.
        self.create_loss()
        # Use gradient descent to minimize loss.
        self.create_optimiser(opt_kwargs)
        # Create accuracy.
        self.create_accuracy()

        print("finished creating CNN")

    def create_placeholders(self):
        with tf.name_scope("data"):
            self.X=tf.placeholder(tf.float32, shape=(None,self.n_feats), name="X_data")
            self.Y=tf.placeholder(tf.float32, shape=(None,self.n_categories), name="Y_data")
            self.dropout_keepprob=tf.placeholder(tf.float32, name="keep_probability")

    def create_CNN(self, N_filters=10):
        with tf.name_scope("CNN"):
            # conv layer 1, 5x5 kernel, 1 input 10 output channels
            W_conv1 = self.weight_variable([5, 5, 1, N_filters],name="conv1",dtype=tf.float32) 
            b_conv1 = self.bias_variable([N_filters],name="conv1",dtype=tf.float32)
            X_reshaped = tf.reshape(self.X, [-1, self.L, self.L, 1])
            h_conv1 = tf.nn.relu(self.conv2d(X_reshaped, W_conv1, name="conv1") + b_conv1)

            # Pooling layer - downsamples by 2X.
            h_pool1 = self.max_pool_2x2(h_conv1,name="pool1")
            # conv layer 2, 5x5 kernel, 10 input 20 output channels
            W_conv2 = self.weight_variable([5, 5, 10, 20],name="conv2",dtype=tf.float32)
            b_conv2 = self.bias_variable([20],name="conv2",dtype=tf.float32)
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, name="conv2") + b_conv2)

            # Dropout controls the complexity of the CNN, prevents co-adaptation of features.
            h_conv2_drop = tf.nn.dropout(h_conv2, self.dropout_keepprob,name="conv2_dropout")

            # Second pooling layer
            h_pool2 = self.max_pool_2x2(h_conv2_drop,name="pool2")

            # Fully connected layer 1 -- after second round of downsampling, our 40x40 image
            # is down to 7x7x20 feature maps -- maps this to 50 features.
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*20])

            W_fc1 = self.weight_variable([7*7*20, 50],name="fc1",dtype=tf.float32)
            b_fc1 = self.bias_variable([50],name="fc1",dtype=tf.float32)

            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            # Dropout controls the complexity of the CNN, prevents co-adaptation of features.
            h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout_keepprob,name="fc1_dropout")

            # Map the 50 features to 2 classes, one for each phase.
            W_fc2 = self.weight_variable([50, self.n_categories],name="fc12",dtype=tf.float32)
            b_fc2 = self.bias_variable([self.n_categories],name="fc12",dtype=tf.float32)

            self.Y_predicted = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    def weight_variable(self, shape, name="", dtype=tf.float32):
        """
        Generate a weight variable of a given shape.
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial,dtype=dtype,name=name)

    def bias_variable(self, shape, name="", dtype=tf.float32):
        """
        Generate a bias variable of a given shape.
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial,dtype=dtype,name=name)

    def conv2d(self, x, W, name=""):
        """
        Return a 2d convolution layer with full stride.
        """
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID", name=name)

    def max_pool_2x2(self, x,name=""):
        """
        Downsample a feature map by 2X.
        """
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], 
                                padding="VALID",
                                name=name
                                )
