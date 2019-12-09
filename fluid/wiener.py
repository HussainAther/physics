

class wienerprocess():
    """
    Wiener process implementation as a special case of brownian motion
    with 0 drift and variance of t.
    dW_t = mu*dt + sigma*dB_t
    W_t ~ N*(mu*t, sigma**2t)
    """
