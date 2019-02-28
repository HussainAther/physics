from vpython import *
import numpy as np

"""
Solve Maxwell's equations with FDTD time-stepping for circularly polarized wave propogations
in the z-direction in free space.
"""

scene = display(x=0, y=0, width=600, height=400, range=200, title="Circular polarization: E field \
    in white. H field in yellow")
global phy, pyx
max = 201

c = .01 # Courant stability condition

# set arrays
Ex = np.zeros((max+2, 2), float)
Hy = np.zeros((max+2, 2), float)
Ey = np.zeros((max+2, 2), float)
Hx = np.zeros((max+2, 2), float)

arrowcol = color.white
Earrows = []
Harrows = []

for i in range(0, max, 10):
    Earrows.append(arrow(pos=(0, i-100), axis=(0,0,0), color=arrowcol))
    Harrows.append(arrow(pos=(0, i-100), axis=(0,0,0), color=color.yellow))

def plotfields(Ex, Ey, Hx, Hy):
    for n, arr in enumerate(Earrows):
        arr.axis = (35*Ey[10*n, 1],0,35*Ex[10*n,1])
    for n, arr in enumerate(Harrows):
        arr.axis = (35*Hy[10*n, 1],0,35*Hx[10*n,1])

def inifields(): # initialize the fields
    phx = .5*pi
    phy = 0
    k = arange(0, max)
    Ex[:-2,0] = cos(-2*pi*k/200 + phx)
    Ey[:-2,0] = cos(-2*pi*k/200 + phy)
    Hx[:-2,0] = cos(-2*pi*k/200 + phy + pi)
    Hy[:-2,0] = cos(-2*pi*k/200 + phx)

def newfields():
    while True:
        rate(1000)
        Ex[1:max-1, 1] = Ex[1:max-1, 0] + c*(Hy[:max-2, 0]-Hy[2:max,0])
        Ey[1:max-1, 1] = Ey[1:max-1, 0] + c*(Hx[2:max, 0]-Hx[:max-2,0])
        Hx[1:max-1, 1] = Hx[1:max-1, 0] + c*(Ey[2:max, 0]-Ey[:max-2,0])
        Hy[1:max-1, 1] = Hy[1:max-1, 0] + c*(Ex[:max-2, 0]-Ex[2:max,0])
        Ex[0,1] = Ex[0,0] + c*(Hy[200-1,0] - Hy[1,0])
        Ex[200, 1] = Ex[200,0] + c*(Hy[200-1,0] - Hy[1,0])
        Ey[0,1] = Ey[0,0] + c*(Hx[1,0] - Hx[200-1,0])
        Ey[200, 1] = Ey[200,0] + c*(Hx[1,0] - Hx[200-1,0])
        Hx[0,1] = Hx[0,0] + c*(Ey[1,0] - Ey[200-1,0])
        Hx[200, 1] = Hx[200,0] + c*(Ey[1,0] - Ey[200-1,0])
        Hy[0,1] = Hy[0,0] + c*(Ex[200-1,0] - Ex[1,0])
        Hy[200, 1] = Hy[200,0] + c*(Ex[200-1,0] - Ex[1,0])
