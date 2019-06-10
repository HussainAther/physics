import numpy as np

"""
Cross entropy as a measure of optimizing a convolutional neural network.
"""

def softmax(X):
    """
    Softmax function takes an N-dimensional vector of real numbers and transforms 
    it into a vector of real number in range (0,1) which add upto 1.
    """
    exps = np.exp(X)
    return exps / np.sum(exps)

def cross_entropy(X,y):
    """
    Return cross entropy, distance between what the model believes a distribution
    is and what the distribution really is. X is the output from fully connected 
    layer (num_examples x num_classes). y is labels (num_examples x 1)
    """
    m = y.shape[0]
    p = softmax(X)
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) / m
    return loss
