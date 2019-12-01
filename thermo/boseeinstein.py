import matplotlib
import matplotlib.pyplot as plt

from matplotlib import animation
from numpy import exp, pi, arange, meshgrid, sqrt, linspace
from numpy.fft import fft2, ifft2, fftshift

"""
Bose-Einstein condensate (bose einstein BEC) for evolving Gross-Pitaevskii
equation (gross pitaevskii gpe). 
"""
