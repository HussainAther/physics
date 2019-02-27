import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
import math

"""
Lennard-Jones potential approximates the interaction between a pair of neutral atoms or molecules.

We gain an interatomic potential of:

V_LJ = e *epsilon [ (sigma/r)^12 - (sigma/r)^6] = epsilon[(r_m/r)^12 - s*(r_m/r)^6]

in which epsilon is the depth of the potential well, sigma is the finite distance at which
the inter-particle potential is zero, r is the distance between the particles, and r_m is the
distance at which potential is a minimum. We use the Verlet algorithm (as was used in
"""

number_of_particles = 70
my_particles = []
background_colour = (255,255,255)
width, height = 500, 500
sigma = 1
e = 1
dt = 0.1
v = 0
a = 0
r = 1

class Particle():
    def __init__(self, (x, y), size):
        self.x = x
        self.y = y
        self.size = size
        self.colour = (0, 0, 255)
        self.thickness = 1
        self.speed = 0
        self.angle = 0

    def display(self):
        pygame.draw.circle(screen, self.colour, (int(self.x), int(self.y)), self.size, self.thickness)

    def move(self):
        self.x += np.sin(self.angle)
        self.y -= np.cos(self.angle)

    def bounce(self):
        if self.x > width - self.size:
            self.x = 2*(width - self.size) - self.x
            self.angle = - self.angle

        elif self.x < self.size:
            self.x = 2*self.size - self.x
            self.angle = - self.angle

        if self.y > height - self.size:
            self.y = 2*(height - self.size) - self.y
            self.angle = np.pi - self.angle

        elif self.y < self.size:
            self.y = 2*self.size - self.y
            self.angle = np.pi - self.angle    

def r(p1,p2):
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    angle = 0.5 * math.pi - math.atan2(dy, dx)
    dist = np.hypot(dx, dy)
    return dist

def collide(p1, p2): # simulate collisions
    dx = p1.x - p2.x
    dy = p1.y - p2.y

    dist = np.hypot(dx, dy)
    if dist < (p1.size + p2.size):
        tangent = math.atan2(dy, dx)
        angle = 0.5 * np.pi + tangent

        angle1 = 2*tangent - p1.angle
        angle2 = 2*tangent - p2.angle
        speed1 = p2.speed
        speed2 = p1.speed
        (p1.angle, p1.speed) = (angle1, speed1)
        (p2.angle, p2.speed) = (angle2, speed2)

        overlap = 0.5*(p1.size + p2.size - dist+1)
        p1.x += np.sin(angle) * overlap
        p1.y -= np.cos(angle) * overlap
        p2.x -= np.sin(angle) * overlap
        p2.y += np.cos(angle) * overlap

def verlet_step(): # Velocity-Verlet algorithm
    """
    This version of the Verlet algorithm, with its increased stability,
    uses a forward-difference approximation for the derivative to advance
    both the position and velocity simultaneously.
    """
    v += a*0.5*dtt
    x += v*dt
    t += dt
    do_collisions(t,x,v,dt)
    a = eval_a(x)
    v += a*0.5*dt
    do_statistics(t,x,v)

def LJ(r): # calculate the Lennard-Jones potential
    return -24*e*((2/r*(sigma/r)**12)-1/r*(sigma/r)**6)
