from vpython import *

"""
Runge-Kutta method (rk4 algorithm) for determining solutions to the 1-Dimensional Schrödinger equation
for bound-state energies.
"""

psigr = display(x=0, y=0, width=600, height=300, title="R and L Wavefunction")
Lwf = curve(x=list(range(502)), color=color.red)

