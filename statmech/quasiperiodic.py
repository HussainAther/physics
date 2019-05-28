import numpy as np
import matplotlib.pyplot as plt

from scipy.special import erf, erfinv

"""
Quasiperiodic (quasi periodic) signal.
"""

def gaussian_frequency(array_length = 10000, central_freq = 100, std = 10):
    """
    Output Gaussian frquencies for an array length, central frequency,
    and standard deviation.
    """
    n = np.arange(array_length)
    f = np.sqrt(2)*std*erfinv(2*n/array_length - erf(central_freq/np.sqrt(2)/std)) + central_freq
    return f

t = range(300) # time
phi = np.linspace(0,2*np.pi, len(f)) # phases
s = [] # signal
for i in t:
    s.append(np.exp(-i-150)**2/30**2) * np.cos(2*np.pi*.1*i)

plt.figure()
plt.plot(t, s)
