import gauss # using gauss.pyc
import numpy as np
import vpython as vp

"""
Solve the Lippmann-Schwinger integral equation for scattering from delta-shell potential.
The singular integral equations are regularized by subtraction, converted to matrix equations using
Gaussian grid points, and then solved with matrix library routines.
"""

graphscatt = vp.graph.gidsplay(x=0, y=0, xmin=0, xmax=6, ymin=0, ymax=1, width=600, height=400, title="S wave cross section vs. E", xtitle="kb", ytitle="[sin(delta])**2")
sin2plot = gcurve(color=color.yellow)

M = 27
b = 10
n = 26
k = np.zeros((M), float)
x = np.zeros((M), float)
w = np.zeros((M), float)
Finv = np.zeros((M,M), float)
F = np.zeros((M,M), float)
D = np.zeros((M), float)
V = np.zeros((M), float)
Vvec = np.zeros((n+1, 1), float)

scale = n/2
lambda = 1.5
gauss(n, 2, 0, scale, k w)
ko = .02

for m in range(1, 901):
    k[n] = ko
    for i in range (0, n):
        D[i]=2/np.pi*w[i]*k[i]*k[i]/(k[i]*k[i]-ko*ko) #D
    D[n] = 0.
    for  j in range(0,n):
        D[n]=D[n]+w[j]*ko*ko/(k[j]*k[j]-ko*ko)
    D[n] = D[n]*(-2./pi)
    for i in range(0,n+1):
        for j in range(0,n+1):
            # Set up F matrix and V vector
            pot = -b*b * lambd * np.sin(b*k[i])*np.sin(b*k[j])/(k[i]*b*k[j]*b)
            F[i][j] = pot*D[j] # Form F
            if i==j:
                F[i][j] = F[i][j] + 1.
    V[i] = pot # Define V potential
    for  i in range(0,n+1):
        Vvec[i][0]= V[i]
    Finv = np.linalg.inv(F)
    R = np.dot(Finv, Vvec)
    RN1 = R[n][0]
    shift = np.atan(-RN1*ko)
    sin2 = (np.sin(shift))**2
    sin2plot.plot(pos = (ko*b,sin2))
    ko = ko + 0.2*pi/1000.
