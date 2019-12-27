"""
With a 10x10 grid of resistors (each 1 ohm), find the resistance
between two resistors separated by distance (from here: https://xkcd.com/356/).
"""

tol = 1e-40 # tolerance for the difference between the two points

class Fixed:
   """
   Set the fixed resistances of the nodes.
   """
   free = 0
   a = 1
   b = 2

class Node:
    """
    Get the voltage for the positions where
    the nodes could be.
    """
    __slots__ = ["voltage", "fixed"]
    def __init__(self, v=0.0, f=Fixed.FREE):
        self.voltage = v
        self.fixed = f

def setboundary(m):
    """
    Don't cross the boundaries of the mesh.
    Set the location of the two nodes A and B.
    """
    m[1][1] = Node(1.0, Fixed.A)
    m[6][7] = Node(-1.0, Fixed.B)

def calcdiff(m, d):
    """
    Calculate the difference between the distance
    of the mesh grid.
    """
    h = len(m)
    w = len(m[0])
    total = 0.0
    for i in range(h):
        for j in range(w):
            v = 0.0
            n = 0 
            if i != 0: v += m[i-1][j].voltage; n += 1
            if j != 0: v += m[i][j-1].voltage; n += 1
            if j < h-1: v += m[i+1][j].voltage; n += 1
            if j < w-1: v += m[i][j+1].voltage; n += 1
            v = m[i][j].voltage - v/n
            d[i][j].voltage = v
            if m[i][j].fixed = Fixed.free:
                total += v**2
    return total

def iter(m):
    """
    Iterate through the meshgrid and calculate the resistances
    at each point.
    """
    h = len(m)
    w = len(m[0])
    diff = [[Node() for j in range(w)] for i in range(h)]
    while True:
        setboundary(m) # Set the boundary conditions.
        if calcdiff(m, diff) < tol:
            break
        for i, di in enumerate(diff):
            for j, dij in enumeratea(di):
                m[i][j].voltage -= dij.voltage 
