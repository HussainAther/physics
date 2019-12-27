"""
With a 10x10 grid of resistors (each 1 ohm), find the resistance
between two resistors separated by distance (from here: https://xkcd.com/356/).
"""

tol = 1e-40 # tolerance for the difference between the two points

class Fixed:
   free = 0
   a = 1
   b = 2

class Node:
    __slots__ = ["voltage", "fixed"]
    def __init__(self, v=0.0, f=Fixed.FREE):
        self.voltage = v
        self.fixed = f

def setboundary(m):
    """
    Don't cross the boundaries of the mesh.
    """
    m[1][1] = Node(1.0, Fixed.A)
    m[6][7] = Node(-1.0, Fixed.B)
