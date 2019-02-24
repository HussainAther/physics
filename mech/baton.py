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

