import numpy as np
import scipy.stats as stats

class SqBessel_process(p):
    """
    The (lambda0 dimensional) squared Bessel process is defined by the SDE:
    dX_t = lambda_0*dt + nu*sqrt(X_t)dB_t
    Based on R.N. Makarov and D. Glew's research on simulating squared bessel process. See "Exact
    Simulation of Bessel Diffusions", 2011.
    """
