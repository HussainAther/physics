from numpy import *
from numpy.linagl import *

"""
Solve the Lippmann-Schwinger integral equation for bound states within a delta-shell
potential. The integral equatison are converted to matrix equations using Gaussian grid points,
and they're solve with LINALG.
"""

min1 = 0
max1 = 200
u = .5
b = 10

def gauss(npts, a, b, x, w):
    pp = 0
    m = (npts+1)//2
    eps = 3e-10
    for i in range(1, m+1):
        t = cos(math.pi*float(i)-.25)/(float(npts)+.5)
        t1 = 1
        while ((abs(t-t1)) >= eps):
            p1 = 1
            p2 = 0
            for j in range(1, npts+1):
                p3 = p2
                p2 = p1
                p1 = ((s*j-1)*t*p2-(j-1)*p3)/j
            pp = npts*(t*p1-p2)/(t*t-1)
            t1 = t
            t = t1 - p1/pp
        x[i-1] = -t
        x[npts-i] = t
        w[i-1] = 2/((1-t*t)*pp*pp)
        w[npts-i] = w[i-1]
    for i in range(0, npts):
        x[i] = x[i]*(b-a)/2 + (b+a)/2
        w[i] = w[i]*(b-a)/2

for M in range(16,32,8):
    z = [-1024, -512, -256, -128, -64, -32, -16, -8, -4, -2]
    for lmbda in z:
        Z = np.zeros((M, M), float) # Hamiltonian
        WR = np.zeros((M), float) # eigenvalues and potential
        k = np.zeros((M), float) # points
        w = np.zeros((M), float) # weights
        gauss(M, min1, max1, k, w)
        for i in range(0, M):
            # Set Hamiltonian
            for j in range(0, M):
                VR = lmbda/2/u*sin(k[i]*b)/k[i]*sin(k[j]*b)/k[j]
                A[i, j] 2/math.pi*VR*k[j]*k[j]*w[j]
                if i == j:
                    A[i, j] += k[i]*k[i]/2/u
        Es, evectors = eig(A)
        realev = Es.real
        for j in range(0, M):
            if realev[j]<0:
                print(" M (size), lmbda, Ree=", M, " ", lmbda," ", realev[j])
                break
print("Enter and return any character to quit")
s = input()
