import numpy as np
from qutip import *
from pylab import *

"""
Calculate correlation functions of the kind ⟨𝐴(𝑡1 + 𝑡2)𝐵(𝑡1)⟩, i.e., the correlation
function of a system that is not in its steadystate. In QuTiP, we can evoluate such correlation functions using the
function qutip.correlation.correlation_2op_2t. The default behavior of this function is to return
a matrix with the correlations as a function of the two time coordinates (𝑡1 and 𝑡2).
"""

times = np.linspace(0, 10.0, 200)
a = destroy(10)
x = a.dag() + a
H = a.dag() * a
alpha = 2.5
rho0 = coherent_dm(10, alpha)
corr = correlation_2op_2t(H, rho0, times, times, [np.sqrt(0.25) * a], x, x)
pcolor(corr)
xlabel(r"Time $t_2$")
ylabel(r"Time $t_1$")
title(r"Correlation $\left<x(t)x(0)\right>$")
show()
