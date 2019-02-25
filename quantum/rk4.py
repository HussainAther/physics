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
count_max s
