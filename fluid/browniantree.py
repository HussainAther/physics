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
        self.rect = pygame.Rect(location[0], location[1], size, size)
        self.surface.fill(color, self.rect)

    def onEdge(self):
        """
        The edge where the vectors meet.
        """
        if self.rect.left <= 0:
            """
            If the left side of the rectangle of the particle
            is to the left of the x=0 line, we take the absolute
            value of its velocity in the x-direction in determining 
            the new vector. 
            """
            self.vector = (abs(self.vector[0]), self.vector[1])
        elif self.rect.top <= 0:
            self.vector = (self.vector[0], abs(self.vector[1]))
        elif self.rect.right >= windowsize:
            self.vector = (-abs(self.vector[0]), self.vector[1])
        elif self.rect.bottom >= windowsize:
            """
            When a particle hits the bottom of the window, we push
            it back upward.
            """
            self.vector = (self.vector[0], -abs(self.vector[1]))

    def update(self):
        """
        Update each particle's velocity and location.
        """
        if freeParticles in self.groups():
            self.surface.fill((0,0,0), self.rect)
            """
            Fill the particles with white color when they're 
            free.
            """
            self.remove(freeParticles)
            if pygame.sprite.spritecollideany(self, freeParticles):
                """
                When a collision occurs, ecah particle experiences
                an acceleration.
                """
                self.accelerate((randint(-maxspeed, maxspeed), 
                                 randint(-maxspeed, maxspeed)))
                self.add(freeParticles)
            elif pygame.sprite.spritecollideany(self, tree):
                """
                Stop the particles from moving when they collide
                with the tree.
                """
                self.stop()
            else:
                """
                If they haven't collided with the tree, they're
                still free particles.
                """
                self.add(freeParticles)
            self.onEdge()
 
            if (self.vector == (0,0)) and tree not in self.groups():
                self.accelerate((randint(-maxspeed, maxspeed), 
                                 randint(-maxspeed, maxspeed)))
            self.rect.move_ip(self.vector[0], self.vector[1])
        self.surface.fill(color, self.rect)

    def stop(self):
        """
        If a particle stops, it's no longer a free particle, and it is
        added to the tree.
        """
        self.vector = (0, 0)
        self.remove(freeParticles)
        self.add(tree)
 
    def accelerate(self, vector):
        """
        Accelerate the particle by changing the vector for velocity.
        """
        self.vector = vector
