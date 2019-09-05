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

@arrayargs
def angle(x0, x1, x2):
    """
    Return angle between three points.
    """
    assert x1.shape == x2.shape == (2,)
    a, b = x1 - x0, x1 - x2
    return arccos(dot(a, b)/(norm(a)*norm(b)))

@arrayargs
def is_left(x0, x1, x2):
    """
    Return True if x0 is left of the line between x1 and x2. False otherwise.
    """
    assert x1.shape == x2.shape == (2,)
    matrix = array([x1-x0, x2-x0])
    if len(x0.shape) == 2:
        matrix = matrix.transpose((1, 2, 0))
    return det(matrix) > 0

def lininterp2(x1, y1, x):
    """Linear interpolation at points x between numpy arrays (x1, y1).
    Only y1 is allowed to be two-dimensional.  The x1 values should be sorted
    from low to high.  Returns a numpy.array of y values corresponding to
    points x.
    """
    return splev(x, splrep(x1, y1, s=0, k=1))

def finalize_plot():
    """
    Finalize the plot.
    """
    ax = pyplot.axes()
    ax.set_xticks([])
    ax.set_yticks([])
    pyplot.xlim(xmin/zoom+xoffset, xmax/zoom+xoffset)
    pyplot.ylim(ymin/zoom, ymax/zoom)
    pyplot.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

