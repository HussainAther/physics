from vpython import *

"""
Runge-Kutta method (rk4 algorithm) for determining solutions to the 1-Dimensional Schrödinger equation
for bound-state energies.

Now I ain't sayin she a Schrodinger,
but she ain't messin with no old thinker,
"""

psigr = display(x=0, y=0, width=600, height=300, title="R and L Wavefunction")
Lwf = curve(x=list(range(502)), color=color.red)
