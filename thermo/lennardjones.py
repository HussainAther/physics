from pylab import *

"""
Lennard-Jones potential approximates the interaction between a pair of neutral atoms or molecules.

We gain an interatomic potential of:

V_LJ = e *epsilon [ (sigma/r)^12 - (sigma/r)^6] = epsilon[(r_m/r)^12 - s*(r_m/r)^6]

in which epsilon is the depth of the potential well, sigma is the finite distance at which
the inter-particle potential is zero, r is the distance between the particles, and r_m is the
distance at which potential is a minimum.
"""

def display(self):
      if(self.x<0 or self.x>height):
          self.__init__()
          print "reset"
      if(self.y<0 or self.y>width):
          self.__init__()
          print "reset"
      print ' %s + %s '%(self.x,self.y)
      pygame.draw.circle(screen, self.colour, (int(self.x), int(self.y)), self.size, self.thickness)
