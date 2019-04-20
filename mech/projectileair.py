from vpython.graph import *

"""
Solve for projectile motion with air resistance and
find the frictionless case analytically.
"""

v0 = 22
angle = 34
g = 9.8 # acceleration due to gravity
kf = 0.9 
N = 25
v0x = v0*cos(angle*pi/180)
v0y = v0*sin(angle*pi/180)
T = 2*v0y/g # time
H = v0y*voy/(2*g) # height (vertical distance)
R = 2*v0x*v0y/g # horizontal distance


graph1 = gdisplay(title="Projectile width (red)/without (yellow) Air Resistance",
                xtitle ="x", ytitle="y", xmax=R, xmin=-R/20, ymax=8, ymin=-6.0)

funct1 = gcurve(color=color.red)
funct1 = gcurve(color=color.yellow)

print("Frictionless T = " + str(T))
print("Frictionless H = " + str(H))
print("Frictionless R = " + str(R))

def plotNumeric(k):
    """
    Plot the numeric solution to the projectile using individual values 
    for each variable. 
    """ 
    vx = v0*cos(angle*pi/180)
    vy = v0*sin(angle*pi/180)
    x = 0
    y = 0
    dt = vy/g/N/2
    for i in range(3*N):
        rate(30)
        vx = vx - k*vx*dt
        vy = vy - g*dt - k * vy * dt
        x = x + vx*dt
        y = y + vy*dt
        funct.plot(pos=(x,y))
        print(" %13.10f %13.10f " % (x,y))

def plotAnalytic():
    """
    The analytic solution uses the differnetials in the equations.
    """
    v0x = v0*cos(angle*pi/180)
    v0y = v0*sin(angle*pi/180)
    dt = 2*v0y/g/N
    for i in range(N):
        rate(30)
        t = i*dt
        x = v0x*t
        y = v0y*t - g*t*t/2
        funct1.plot(pos=(x,y))
        print(" %13.10f %13.10f "%(x,y))

plotNumeric()
plotAnalytic()
