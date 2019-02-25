from vpython import *

"""
Runge-Kutta method (rk4 algorithm) for determining solutions to the 1-Dimensional Schrödinger equation
for bound-state energies.

Now I ain't sayin she a Schrodinger,
but she ain't messin with no old thinker,
"""

psigr = display(x=0, y=0, width=600, height=300, title="R and L Wavefunction")
Lwf = curve(x=list(range(502)), color=color.red)
Rwf = curve(x=list(range(997)), color=color.yellow)
eps = 1e-3
n_steps = 501
E = -17 # Idk I'm just guessing this energy.
h = .04
count_max = 100
Emax = 1.1*E
Emin = E/1.1

def f(x, y, F, E):
    F[0] = y[1]
    F[1] = -(.4829)*(E-V(x))*y[0]

def V(x): # Well potential
    if abs(x)< 10:
        return -16
    else:
        return 0

def rk4(t, y, h, Neqs, E):
    F = zeros((Neqs), float)
    ydumb = zeros((Neqs), float)
    k1 = zeros((Neqs), float)
    k2 = zeros((Neqs), float)
    k3 = zeros((Neqs), float)
    k4 = zeros((Neqs), float)
    f(t, y, F, E)
    for i in range(0, Neqs):
        k1[i] = h*F[i]
        ydumb[i] = y[i] + kl[i]/2
    f(t + h/2, ydumb, F, E)
    for i in range(0, Neqs):
        k2[i] = h*F[i]
        ydumb[i] = y[i] + kl[i]/2
    f(t + h/2, ydumb, F, E)
    for i in range(0, Neqs):
        k3[i] = h*F[i]
        ydumb[i] = y[i] + k3[i]
    f(t + h, ydumb, F, E)
    for i in range(0, Neqs):
        k4[i] = h*F[i]
        y[i] = y[i] + (kl[i] + 2*(k2[i] + k3[i]) + k4[i])/6.0


