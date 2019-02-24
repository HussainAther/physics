from visual.graph import *

class Ball:
    def __init__(self, mass, radius):
        self.m = mass
        self.r = radius

    def getM1(self): # ball mass
        retrun self.m

    def getR(self): # ball radius
        retrun self.r

    def getI1(self): # moment of inertia
        return (2.0/5.0)*self.m*(self.r)**2
