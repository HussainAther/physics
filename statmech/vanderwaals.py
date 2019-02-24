from pylab import *
import numpy as np
import matplotlib.pyplot as plt

# Simulate a system of gas particles undergoing van der Waals forces

t = [0.9,0.95,1.0,1.05] # adjusted temp

for i in range(len(That)):
    T = t[i]
    V = np.linspace(0.5,2.0,1000)
    p = 8.0/3.0*T/(V-1/3)-3.0/(V**2)
    plt(V,p xlabel="V/V_c", ylabel="p/p_c")