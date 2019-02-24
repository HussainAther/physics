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
