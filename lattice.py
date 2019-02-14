import numpy as np
import math

"""
First, we need to initialize the system by generating a lattice of atoms
with random velocities. We write a small script to generate a lattice and
write it to disk. The lattice consists of units cells of size b (measure in
units of σ, as are everything in the simulation). If a unit cell starts at r0,
then the 4 atoms in the units cell are at positions r0, r0 + (b/2, b/2, 0),
r0 + (b2, 0, b2), and r0 + (0, b/2, b/2). If a system consists of L × L × L
such cells it is simple to create the system: We simply loop through all L^3
positions of r0 and for each such position we add atoms at the four
positions relative to r0. This is done using the following script...
"""


L = 5; % Lattice size
b = 2.0; % Size of unit cell (units of sigma)
v0 = 1.0; % Initial kinetic energy scale
N = 4*L^3; % Nr of atoms
r = zeros(N,3);
