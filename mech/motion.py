import lattice
from numpy import zeros

[Lx, Ly, Lz, r, v]=readlammps(’mymdinit.lammpstrj’)
L = [Lx, Ly, Lz]
s = len(r)
N = [1]

t = 3.0; dt = 0.001
n = ceil(t/dt)
a = zeros(N,3) # Store calculated accelerations
for i in range(1, n-1): # Loop over timesteps
    a(:,:) = 0;
    for i1 in range(1, N):
        for i2 in range(i1+1, N):
            dr = r(i,i1,:) - r(i,i2,:)
            for k = 1:3 % Periodic boundary conditions
                    if (dr(k)>L(k)/2):
                        dr(k) = dr(k) - L(k)
                    if (dr(k)<-L(k)/2):
                        dr(k) = dr(k) + L(k)
            rr = dot(dr,dr);
            aa = -24*(2*(1/rr)^6-(1/rr)^3)*dr/rr
            a(i1,:) = a(i1,:) + aa(1) # from i2 on i1
            a(i2,:) = a(i2,:) - aa(2) # from i1 on i2
    v(i+1,:,:) = v(i,:,:) + a*dt
    r(i+1,:,:) = r(i,:,:) + v(i+1,:,:)*dt
    # Periodic boundary conditions
    for i1 in range(1, N):
        for k in range(1,3):
            if (r(i+1,i1,k)>L(k)):
                r(i+1,i1,k) = r(i+1,i1,k) - L(k)
            if (r(i+1,i1,k)<0):
                r(i+1,i1,k) = r(i+1,i1,k) + L(k)
writelammps(’mymddump.lammpstrj’,Lx,Ly,Lz,r,v);
