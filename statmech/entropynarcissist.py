import math
from collections import Counter
 
"""
A program that compute its own entropy.
"""

def entropy(s):
    """
    For some input value of text s, calculate entropy.
    """
    p, lns = Counter(s), float(len(s))
    return -sum( count/lns * math.log(count/lns, 2) for count in p.values())
 
with open(__file__) as f: # Open this file itself
    b=f.read() # Read this file
 
print(entropy(b)) # Calculate the entropy of this file
