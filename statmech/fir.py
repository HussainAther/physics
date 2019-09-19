import numpy as np
import matplotlib.pyplot as plt

from numpy import convolve as np_convolve
from scipy.signal import fftconvolve, lfilter, firwin
from scipy.signal import convolve as sig_convolve
from scipy.ndimage import convolve1d

"""
Finite impulse response (FIR) filter on input signal using numpy as scipy.
"""

m = 1
n = 2 ** 18
x = np.random.random(size=(m, n))

conv_time = []
npconv_time = []
fftconv_time = []
conv1d_time = []
lfilt_time = []

diff_list = []
diff2_list = []
diff3_list = []

ntaps_list = 2 ** np.arange(2, 14)

for ntaps in ntaps_list:
    # Create a FIR filter.
    b = firwin(ntaps, [0.05, 0.95], width=0.05, pass_zero=False)
    # Signal convolve.
    tstart = time.time()
    conv_result = sig_convolve(x, b[np.newaxis, :], mode='valid')
    conv_time.append(time.time() - tstart)

