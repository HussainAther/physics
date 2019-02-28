import matplotlib.pylab as p
from mpl_toolkits.mplot3d import Axes3D
from vpython import *

"""
Solve the Korteweg–de Vries (KdeV) equation forr a soliton.
"""

ds = .4
dt = .1
max = 2000
mu = .1
eps = .2 # precision
mx = 131

# initial wave
for i in range(0, 131):
    u[i, 0] = .5*(1-((math.exp(2*(.2*ds*i-5))-1)/(math.exp(2*(.2*ds*i-5))+1)))

# end points
u[0,1] = 1
u[0,2] = 1
u[130,1] = 0
u[130,2] = 0

for i in range(0, 131, 2):
    spl[i, 0] = u[i, 0]

fac = mu*dt/(ds**3)
