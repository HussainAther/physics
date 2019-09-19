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
    conv_result = sig_convolve(x, b[np.newaxis, :], mode="valid")
    conv_time.append(time.time() - tstart)
    # --- numpy.convolve ---
    tstart = time.time()
    npconv_result = np.array([np_convolve(xi, b, mode="valid") for xi in x])
    npconv_time.append(time.time() - tstart)
    # fft convolve (fast fourier transform) convolution.
    tstart = time.time()
    fftconv_result = fftconvolve(x, b[np.newaxis, :], mode="valid")
    fftconv_time.append(time.time() - tstart)
    # 1-dimensional (one-dimensional) convolution.
    tstart = time.time()
    # convolve1d doesn't have a "valid" mode, so we expliclity slice out
    # the valid part of the result.
    conv1d_result = convolve1d(x, b)[:, (len(b)-1)//2 : -(len(b)//2)]
    conv1d_time.append(time.time() - tstart)
    tstart = time.time()
    lfilt_result = lfilter(b, [1.0], x)[:, len(b) - 1:]
    lfilt_time.append(time.time() - tstart)
    diff = np.abs(fftconv_result - lfilt_result).max()
    diff_list.append(diff)
    diff2 = np.abs(conv1d_result - lfilt_result).max()
    diff2_list.append(diff2)
    diff3 = np.abs(npconv_result - lfilt_result).max()
    diff3_list.append(diff3)

