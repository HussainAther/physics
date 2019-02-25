from vpython import *

"""
Use the Numerov method to solve the 1-D time-independent Schrodinger equation
for bound-state energies.
"""

psigr = display(x=0, y=0, width=600, height=300, title="R and L Wave Functions")
psi = curve(x=list(range(0, 1000)), display=psigr, color=color.yellow)
psi2gr = display(x=0, y=300, width=600, height=300, title="Wave func^2")
psio = curve(x=list(range(0, 1000)), color=color.magenta, display=psi2gr)
energr = display(x=0, y=500, width=600, height=200, title="Potential and E")

