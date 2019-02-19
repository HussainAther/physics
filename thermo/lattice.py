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

v = zeros(N,3);
bvec = [0 0 0; b/2 b/2 0; b/2 0 b/2; 0 b/2 b/2];
ip = 0;

# Generate positions
for ix = 0:L-1
    for iy = 0:L-1
        for iz = 0:L-1
            r0 = b*[ix iy iz]; % Unit cell base position
            for k = 1:4
                ip = ip + 1; % Add particle
                r(ip,:) = r0 + bvec(k,:);

# Generate velocities
for i = 1:ip
    v(i,:) = v0*randn(1,3);

def writelammps(filename,Lx,Ly,Lz,r,v)
# WRITELAMMPS Write data to lammps file
    fp = open(filename,’w’)
    s = size(r)
    ip = s(1)
    fp.write(fp,’ITEM: TIMESTEP\n’)
    fp.write(fp,’0\n’)
    fp.write(fp,’ITEM: NUMBER OF ATOMS\n’)
    fp.write(fp,’%d\n’,ip); % Nr of atoms
    fp.write(fp,’ITEM: BOX BOUNDS pp pp pp\n’)
    fp.write(fp,’%f %f\n’,0.0,Lx); # box size, x
    fp.write(fp,’%f %f\n’,0.0,Ly); # box size, y
    fp.write(fp,’%f %f\n’,0.0,Lz); # box size, z
    fp.write(fp,’ITEM: ATOMS id type x y z vx vy vz\n’)
    for i = 1:ip
        fprintf(fp,’%d %d %f %f %f %f %f %f \n’, i,1,r(i,:),v(i,:))
    fp.close()

# Output to file
writelammps(’mymdinit.lammpstrj’,L*b,L*b,L*b,r,v);
