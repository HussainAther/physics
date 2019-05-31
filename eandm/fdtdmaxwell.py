import numpy as np
import vpython as vp

"""
Finite-Difference Time Domain Method of findings solutions to Maxwell's equations for linearly
polarized wave propogation in the z-direction in free space.
"""

xmax = 201
ymax = 100
zmax = 100

scene = vp.display(x=0, y=0, width=800, height=500, title="E: cyan, H: red. Periodic Boundary conditions",
            forward=(-.6,-.5,-1))
Efield = vp.curve(x=list(range(0, xmax), color=color.cyan, radius=1.5, display=scene))
Hfield = vp.curve(x=list(range(0, xmax), color=color.cyan, radius=1.5, display=scene))
vplane = vp.curve(pos=[(-xmax, ymax), (xmax, ymax), (xmax, -ymax), (-xmax, -ymax),
                    (-xmax, ymax)], color=color.cyan)
zaxis = vp.curve(pos=[(-xmax, 0), (xmax, 0)], color=color.magenta)
hplace = vp.curve(pos=[(-xmax, 0, zmax), (xmax, 0, zmax), (xmax, 0, -zmax), (-xmax, 0, -zmax),
                    (-xmax,0, zmax)], color=color.magenta)
ball1 = vp.sphere(pos =(xmax+30, 0, 0), color=color.black, radius=2)
ts = 2
beta = .01

# initialize arrays
Ex = np.zeros((xmax, ts), float)
Hy = np.zeros((xmax, ts), float)

# label
Exlabel1 = label(text="Ex", pos=(-xmax-10, 50), box=0)
Exlabel2 = label(text="Ex", pos=(xmax+10, 50), box=0)
Hylabel = label(text="Hy", pos=(-xmax-10, 0, 50), box=0)
zlabel = label(text="Z", pos=(xmax+10, 0), box=0)
t1 =0

def inifields():
    """
    Initialize the fields with the appropriate values
    """
    k = np.arange(xmax)
    Ex[:xmax, 0] = .1*np.sin(2*np.pi*k/100)
    Hy[:xmax, 0] = .1*np.sin(2*np.pi*k/100)

def plotfields(ti):
    """
    Plot the solutions to the field equations.
    """
    k = arange(xmax)
    Efield.x = 2*k-xmax
    Efield.y = 800*Ex[k,ti]
    Hfield.x = 2*k-xmax
    Hfield.z = 800*Hy[k,ti]

inifields()
plotfields(ti)

while True:
    rate(600)
    Ex[1:xmax-1, 1] = Ex[1:xmax-1,0] + beta*(Hy[0:xmax-1,0]-Hy[2:xmax,0])
    Hy[1:xmax-1, 1] = Hy[1:xmax-1,0] + beta*(Ex[0:xmax-1,0]-Ex[2:xmax,0])
    Ex[0,1] = Ex[0,0] + beta*(Hy[xmax-2,0] - Hy[1,0]) # boundary condition
    Ex[xmax-1, 1] = Ex[xmax-1, 0] + beta*(Hy[xmax-2,0] - Hy[1,0])
    Hy[0,1] = Hy[0,0] + beta*(Ex[xmax-2,0] - EX[1,0]) # boundary condition
    Hy[xmax-1, 1] = Hy[xmax-1, 0] + beta*(Ex[xmax-2,0] - Ex[1,0])
    plotfields(ti)
    Ex[:xmax,0] = Ex[:xmax,1]
    Hy[:xmax,0] = Hy[:xmax,1]
    
