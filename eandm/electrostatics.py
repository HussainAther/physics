import functools
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
from scipy.integrate import ode
from scipy.interpolate import splrep, splev

"""
Electrostatics (electrostatic) functionality
"""

xmin, xmax, ymin, ymax = None, None, None, None
zoom = None
xoffset = None

def arrayargs(func):
    """
    Ensure all args are arrays.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Ensures all args are arrays."""
        # pylint: disable=star-args
        return func(*[array(a) for a in args], **kwargs)
    return wrapper

