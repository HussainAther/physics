import matplotlib.pylab as p

from mpl_toolkits.mplot3d import Axes3D
from vpython import *

"""
Solve the Korteweg–de Vries (KdeV) equation for a soliton. It's a mathematical model
of waves of shallow water surfaces. We can use a nonlinear partial differential
equation to solve it.
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

for i in range(1, mx-10):
    a1 = eps*dt*(u[i+1, 0] + u[i, 0] + u[i-1, 0])/(ds*6)
    if i > 1 and i <129:
        a2 = u[i+2,0] + 2*i[i-1,0] - 2*u[i+1,0]-u[i-2,0]
    else:
        a2 = u[i-1, 0] - u[i+1, 0]
    a3 = u[i+1, 0] - u[i-1, 0]
    u[i, 1] = u[i, 0] - a1*a3 - fac*a2/3

for j in range(1, max+1):
    for i in range(1, mx-2):
        a1 = eps*dt*(u[i+1, 1] + u[i, 1] + u[i-1, 1])/(3*ds)
        if i > 1 and i < mx-2:
            a2 = u[i+2, 1] + 2*u[i-1, 1] - 2*u[i+1,,1] - u[i-2,1]
        else:
            a2 = u[i-1, 1] - u[i+1, 1]
        a3 = u[i+1, 0] - u[i-1, 0]
        u[i, 2] = u[i, 0] - a1*a3 - 2*fac*a2/3
    if j % 100 == 0:
        for i in range(1, mx-2):
            spl[i, m] = u[i, 2]
        print(m)
        m += 1
    for k in range(0, mx):
        u[k, 0] = u[k, 1]
        u[k, 1] = u[k, 2]

x = list(range(0, mx, 2))
y = list(range(0, 21))
X, Y = p.meshgrid(x, y)

fig = p.figure()
ax = Axes3D(fig)
ax.plot_wireframe(X, Y, spl[X, Y], color="r")
ax.set_xlabel("Position")
ax.set_ylabel("Time")
ax.set_zlabel("Disturbance")
p.show()
