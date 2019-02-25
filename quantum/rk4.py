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

def diff(E, h):
    y = zeros((2), float)
    i_match = n_steps//3 # matching radius
    nL = i.match + 1
    y[0] = 1.E-15
    y[1] = -y[0] * sqrt(-E*.04829)
    for i in range(0, nL + 1):
        x = h * (i - n_steps/2)
        rk4(x, y, h, 2, E)
    left = y[1]/y[0]
    y[0] = 1.E-15 # slope for even. reverse for odd
    y[1] = -y[0] * sqrt(-E*.04829)
    for i in range(n_steps, nL+1, -1):
        x = h*(i+1-n_steps/2)
        rk4(x, y, -h, 2, E)
    right = y[1]/y[0] # log derivative
    return((left - right)/(left + right))


def plot(E, h):
    x = 0
    n_steps = 1501
    y= zeros((2), float)
    yL = zeros((2, 505), float)
    i_match = 500
    nL = i_match + 1
    y[0] = 1E-40
    y[1] = -sqrt(-E*.04829) * y[0]
    for i in range(0, nL+1):
        yL[0][i] = y[0]
        yL[1][i] = y[1]
        x = h * (i - n_steps/2)
        rk4(x, y, h, 2, E)
    y[0] = -1E-15
    y[1] = -sqrt(-E*.4829)*y[0]
