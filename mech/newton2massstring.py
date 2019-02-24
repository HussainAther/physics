from numpy.linalg import solve
from vpython.graph import *

"""
Use the Newton-Raphson search to solve the two-mass-on-a-string problem.
We set up matrices using coupled linear equations and check the physical reasonableness
of them by a variety of weights and lengths. We need to check that the tensions we calculate are positive
and that the deduced angles correspond to a physical geometry. The solution will show graphically
the step-by-step search for a solution.
"""

scene = display(x=0, y=0, width=500, height=500, title="String and masses configuration")

tempe = curve(x=range(0, 500), color=color.black)

n = 9
eps = 1e-6 # precision
deriv = zeros((n,n), float) # just get some zeros
f = zeros(n, float)
x = array([.5, .5, .5 .5, .5, .5, .5, 1, 1, 1])

