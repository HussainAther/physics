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
        return func(*[np.array(a) for a in args], **kwargs)
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
    return np.sqrt(np.sum(array(x)**2, axis=-1))

@arrayargs
def point_line_distance(x0, x1, x2):
    """
    Find the shortest distance between the point x0 and the line x1 to x2.
    point line distance pointlinedistance 
    """
    assert x1.shape == x2.shape == (2,)
    return np.fabs(np.cross(x0-x1, x0-x2))/np.norm(x2-x1)

@arrayargs
def angle(x0, x1, x2):
    """
    Return angle between three points.
    """
    assert x1.shape == x2.shape == (2,)
    a, b = x1 - x0, x1 - x2
    return np.arccos(np.dot(a, b)/(np.norm(a)*np.norm(b)))

@arrayargs
def is_left(x0, x1, x2):
    """
    Return True if x0 is left of the line between x1 and x2. False otherwise.
    """
    assert x1.shape == x2.shape == (2,)
    matrix = np.array([x1-x0, x2-x0])
    if len(x0.shape) == 2:
        matrix = matrix.transpose((1, 2, 0))
    return np.det(matrix) > 0

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
    plt.xlim(xmin/zoom+xoffset, xmax/zoom+xoffset)
    plt.ylim(ymin/zoom, ymax/zoom)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

class PointCharge:
    """
    A point charge.
    """
    R = 0.01  # The effective radius of the charge

    def __init__(self, q, x):
        """
        Initialize the quantity of charge 'q' and position vector 'x'.
        """
        self.q, self.x = q, np.array(x)

    def E(self, x):  # pylint: disable=invalid-name
        """
        Electric field vector.
        """
        if self.q == 0:
            return 0
        else:
            dx = x-self.x
            return (self.q*dx.T/np.sum(dx**2, axis=-1)**1.5).T

    def V(self, x):  # pylint: disable=invalid-name
        """
        Potential.
        """
        return self.q/np.norm(x-self.x)

    def is_close(self, x):
        """
        Return True if x is close to the charge; false otherwise.
        """
        return np.norm(x-self.x) < self.R

    def plot(self):
        """
        Plot the charge.
        """
        color = "b" if self.q < 0 else "r" if self.q > 0 else "k"
        r = 0.1*(np.sqrt(np.fabs(self.q))/2 + 1)
        circle = plt.Circle(self.x, r, color=color, zorder=10)
        plt.gca().add_artist(circle)

class PointChargeFlatland(PointCharge):
    """
    A point charge in Flatland.
    """
    def E(self, x):  # pylint: disable=invalid-name
        """
        Electric field vector.
        """
        dx = x-self.x
        return (self.q*dx.T/np.sum(dx**2, axis=-1)).T

    def V(self, x):
        raise RuntimeError("Not implemented")

class LineCharge:
    """
    A line charge.
    """
    R = 0.01  # The effective radius of the charge
    def __init__(self, q, x1, x2):
        """
        Initialize the quantity of charge 'q' and end point vectors
        'x1' and 'x2'.
        """
        self.q, self.x1, self.x2 = q, np.array(x1), np.array(x2)

    def get_lam(self):
        """
        Return the total charge on the line.
        """
        return self.q / np.norm(self.x2 - self.x1)

    def E(self, x):  # pylint: disable=invalid-name
        """
        Electric field vector.
        """
        x = np.array(x)
        x1, x2, lam = self.x1, self.x2, self.lam

        # Get lengths and angles for the different triangles
        theta1, theta2 = angle(x, x1, x2), pi - angle(x, x2, x1)
        a = point_line_distance(x, x1, x2)
        r1, r2 = np.norm(x - x1), norm(x - x2)

        # Calculate the parallel and perpendicular components
        sign = where(is_left(x, x1, x2), 1, -1)

        # pylint: disable=invalid-name
        Epara = lam*(1/r2-1/r1)
        Eperp = -sign*lam*(np.cos(theta2)-np.cos(theta1))/where(a == 0, infty, a)

        # Transform into the coordinate space and return
        dx = x2 - x1

        if len(x.shape) == 2:
            Epara = Epara[::, newaxis]
            Eperp = Eperp[::, newaxis]

        return Eperp * (np.array([-dx[1], dx[0]])/np.norm(dx)) + Epara * (dx/np.norm(dx))

    def is_close(self, x):
        """
        Return True if x is close to the charge.
        """
        theta1 = angle(x, self.x1, self.x2)
        theta2 = angle(x, self.x2, self.x1)
        if theta1 < radians(90) and theta2 < radians(90):
            return point_line_distance(x, self.x1, self.x2) < self.R
        else:
            return np.min([np.norm(self.x1-x), np.norm(self.x2-x)], axis=0) < \
              self.R

    def plot(self):
        """
        Plot the charge.
        """
        color = "b" if self.q < 0 else "r" if self.q > 0 else "k"
        x, y = zip(self.x1, self.x2)
        width = 5*(np.sqrt(np.fabs(self.lam))/2 + 1)
        plt.plot(x, y, color, linewidth=width)

class FieldLine:
    """
    A Field Line.
    """
    def __init__(self, x):
        """
        Initialize the field line points 'x'.
        """
        self.x = x

    def plot(self, linewidth=None, startarrows=True, endarrows=True):
        """
        Plot the field line and arrows.
        """
        if linewidth == None:
            linewidth = rcParams["lines.linewidth"]
        
        x, y = zip(*self.x)
        plt.plot(x, y, "-k", linewidth=linewidth)
        n = int(len(x)/2) if len(x) < 225 else 75
        if startarrows:
            plt.arrow(x[n], y[n], (x[n+1]-x[n])/100., (y[n+1]-y[n])/100.,
                         fc="k", ec="k",
                         head_width=0.1*linewidth, head_length=0.1*linewidth)

        if len(x) < 225 or not endarrows:
            return
        plt.arrow(x[-n], y[-n],
                     (x[-n+1]-x[-n])/100., (y[-n+1]-y[-n])/100.,
                     fc="k", ec="k",
                     head_width=0.1*linewidth, head_length=0.1*linewidth)

class ElectricField:
    """
    The electric field owing to a collection of charges.
    """
    dt0 = 0.01  # The time step for integrations

    def __init__(self, charges):
        """
        Initialize the field given 'charges'.
        """
        self.charges = charges

    def vector(self, x):
        """
        Return the field vector.
        """
        return np.sum([charge.E(x) for charge in self.charges], axis=0)

    def magnitude(self, x):
        """
        Return the magnitude of the field vector.
        """
        return np.norm(self.vector(x))

    def angle(self, x):
        """
        Return the field vector's angle from the x-axis (in radians).
        """
        return np.arctan2(*(self.vector(x).T[::-1])) # arctan2 gets quadrant right

    def direction(self, x):
        """Returns a unit vector pointing in the direction of the field."""
        v = self.vector(x)
        return (v.T/norm(v)).T

    def projection(self, x, a):
        """
        Return the projection of the field vector on a line at given angle
        from x-axis.
        """
        return self.magnitude(x) * np.cos(a - self.angle(x))

    def line(self, x0):
        """
        Return the field line passing through x0.
        """
        if None in [xmin, xmax, ymin, ymax]:
            raise ValueError("Domain must be set using init().")
        # Set up integrator for the field line
        streamline = lambda t, y: list(self.direction(y))
        solver = ode(streamline).set_integrator("vode")
        # Initialize the coordinate lists
        x = [x0]
        # Integrate in both the forward and backward directions
        dt = 0.008
        # Solve in both the forward and reverse directions
        for sign in [1, -1]:
            # Set the starting coordinates and time
            solver.set_initial_value(x0, 0)
            # Integrate field line over successive time steps
            while solver.successful():
                # Find the next step
                solver.integrate(solver.t + sign*dt)
                # Save the coordinates
                if sign > 0:
                    x.append(solver.y)
                else:
                    x.insert(0, solver.y)
                # Check if line connects to a charge
                flag = False
                for c in self.charges:
                    if c.is_close(solver.y):
                        flag = True
                        break
                # Terminate line at charge or if it leaves the area of interest
                if flag or not (xmin < solver.y[0] < xmax) or \
                  not ymin < solver.y[1] < ymax:
                    break
        return FieldLine(x)

    def plot(self, nmin=-3.5, nmax=1.5):
        """
        Plot the field magnitude.
        """
        x, y = meshgrid(
            np.linspace(xmin/zoom+xoffset, xmax/zoom+xoffset, 200),
            np.linspace(ymin/zoom, ymax/zoom, 200))
        z = zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i, j] = np.log10(self.magnitude([x[i, j], y[i, j]]))
        levels = np.arange(nmin, nmax+0.2, 0.2)
        cmap = plt.cm.get_cmap('plasma')
        plt.contourf(x, y, numpy.clip(z, nmin, nmax),
                        10, cmap=cmap, levels=levels, extend="both")
