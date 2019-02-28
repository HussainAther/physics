from vpython import *

"""
Finite-Difference Time Domain Method of findings solutions to Maxwell's equations for linearly
polarized wave propogation in the z-direction in free space.
"""

xmax = 201
ymax = 100
zmax = 100

scene = display(x=0, y=0, width=800, height=500, title="E: cyan, H: red. Periodic Boundary conditions",
            forward=(-.6,-.5,-1))
Efield = curve(x=list(range(0, xmax), color=color.cyan, radius=1.5, display=scene))
Hfield = curve(x=list(range(0, xmax), color=color.cyan, radius=1.5, display=scene))
vplane = curve(pos=[(-xmax, ymax), (xmax, ymax), (xmax, -ymax), (-xmax, -ymax),
                    (-xmax, ymax)], color=color.cyan)
zaxis = curve(pos=[(-xmax, 0), (xmax, 0)], color=color.magenta)
hplace = curve(pos=[(-xmax, 0, zmax), (xmax, 0, zmax), (xmax, 0, -zmax), (-xmax, 0, -zmax),
                    (-xmax,0, zmax)], color=color.magenta)
ball1 = sphere(pos =(xmax+30, 0, 0), color=color.black, radius=2)
ts = 2
beta = .01

# initialize arrays
Ex = zeros((xmax, ts), float)
Hy = zeros((xmax, ts), float)

# label
Exlabel1 = label(text="Ex", pos=(-xmax-10, 50), box=0)
Exlabel2 = label(text="Ex", pos=(xmax+10, 50), box=0)
Hylabel = label(text="Hy", pos=(-xmax-10, 0, 50), box=0)
zlabel = label(text="Z", pos=(xmax+10, 0), box=0)
t1 =0

def inifields():
    k = arange(xmax)
    Ex[:xmax, 0] = .1*sin(2*pi*k/100)
    Hy[:xmax, 0] = .1*sin(2*pi*k/100)

def plotfields(ti):
    k = arange(xmax)
    Efield.x = 2*k-xmax
    Efield.y = 800*Ex[k,ti]
    Hfield.x = 2*k-xmax
    Hfield.z = 800*Hy[k,ti]


