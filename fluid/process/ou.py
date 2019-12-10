import numpy as np
import scipy.stats as stats

class ouprocess(p):
    """
    Orstein-Uhlenbeck process is a stochastic process that
    uses a mean reverting model. It's also called the Vasicek model.
    It uses the stochastic differential equation:
    
    dOU_t = theta * (m - OU_t) * dt + sigma * dB_t 
    
    """
