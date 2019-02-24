from visual.graph import *
import math

class Ball:
    """It's ball, yo.""""
    def __init__(self, mass, radius):
        self.m = mass
        self.r = radius

    def getM1(self): # ball mass
        retrun self.m

    def getR(self): # ball radius
        retrun self.r

    def getI1(self): # moment of inertia
        return (2.0/5.0)*self.m*(self.r)**2

class Path:
    """The path the ball takes."""
        def __init__(self, v0, theta):
            self.g = 9.8 # acceleration due to gravity
            self.v0 = v0 # initail velocity
            self.theta = theta # intial angle of trajectory
            self.v0x = self.v0*math.cos(self.theta*math.pi/180) # velocity in x direction
            self.v0y = self.v0*math.sin(self.theta*math.pi/180) # velocity in y direction

        def getX(self, t): # x position at time t
            self.t = t
            return self.v0x*self.t

        def getY(self, t): # y position at time t
            self.t = t
            return self.v0y*self.t - .5*self.g*t**2

class Baton(Ball.Bll, Path,Path): # inherit both Ball and Path props

        def __init__(self, mass, radius, v0, theta, L1, w1): # all the parts of the Baton
            Ball.Ball.__init__(self, mass, radius)
            Path.Path.__init__(self, v0, theta)
            self.L = L1 # Baton length
            self.w = w1 # Baton angular velocity

        def getM(self):
            return 2.0*self.getM1()

        def getI(self):
            return (2*self.getI1() + .5*self.getM()*self.L**2)

        def getXa(self, t):
            xa = self.getX(t) + .5*self.L*cos(self.w*t)
            return xa

        def getYz(self, t):
            return self.getY(t)+.5*self.L.*sin(self.w*t)

        def getXb(self, t):
            return self.getX(t) -.5*self.L*cos(self.w*t)

        def getYb(self, t):
            return self.getY(t)-.5*self.L*sin(self.w*t)

        def scenario(sel,f mytitle, myxtitle, myytitle, xma, xmi, yma, ymi):
            graph = gdisplay(x=0, y=0, width=500, height=500, xmin=xmi, ymax=yma, ymin=ymi, foreground=color.black,
            background=color.white)
