import numpy as np
import matplotlib.pyplot as plt

from numpy import convolve as np_convolve
from scipy.signal import fftconvolve, lfilter, firwin
from scipy.signal import convolve as sig_convolve
from scipy.ndimage import convolve1d

"""
Finite impulse response (FIR) filter on input signal using numpy as scipy.
"""
