import os
import pygame
import sys

from pygame.locals import
from random import randint

"""
Brownian tree with pygame

Make it beautiful.
"""

maxspeed = 15
size = 3
color = (45, 90, 45)
windowsize = 400
timetick = 1
maxpart = 50

# Get the sprites.
freeParticles = pygame.sprite.Group()
tree = pygame.sprite.Group()

# Set up the display. 
window = pygame.display.set_mode((windowsize, windowsize))
pygame.display.set_caption("Brownian Tree")
 
screen = pygame.display.get_surface()

class Particle(pygame.sprite.Sprite):
    """
    Define each particle.
    """
    def __init__(self, vector, location, surface):
        """
        Initialize each vector and where its located along with
        how it appears on the surface. 
        """
        pygame.sprite.Sprite.__init__(self)
        self.vector = vector
        self.surface = surface
        self.accelerate(vector)
        self.add(freeParticles)
        self.rect = pygame.Rect(location[0], location[1], SIZE, SIZE)
        self.surface.fill(color, self.rect)
