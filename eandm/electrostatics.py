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

def init(xmin, xmax, ymin, ymax, zoom=1, xoffset=0):
    """
    Initialize the domain.
    """
    # pylint: disable=global-statement
    global xmin, xmax, ymin, ymax, zoom, xoffset
    xmin, xmax, ymin, ymax, zoom, xoffset = xmin, xmax, ymin, ymax, zoom, xoffset

def norm(x):
    """
    Return the magnitude of the vector x.
    """
    return sqrt(numpy.sum(array(x)**2, axis=-1))

@arrayargs
def point_line_distance(x0, x1, x2):
    """
    Find the shortest distance between the point x0 and the line x1 to x2.
    point line distance pointlinedistance 
    """
    assert x1.shape == x2.shape == (2,)
    return fabs(cross(x0-x1, x0-x2))/norm(x2-x1)

