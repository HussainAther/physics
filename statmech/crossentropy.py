"""
"""

def softmax(X):
    """
    Softmax function takes an N-dimensional vector of real numbers and transforms 
    it into a vector of real number in range (0,1) which add upto 1.
    """
    exps = np.exp(X)
    return exps / np.sum(exps)
