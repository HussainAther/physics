from vpython import *
import numpy as np

"""
Solve the wave equation for a string of length L = 1m with its ends fixed and with the gently
plucked initial conditions.
"""

# curve
g = display(width=600, height=300, title="Vibrating String")
vibst = curve(x=list(range(0,100)), color=color.yellow)
ball2 = sphere(pos=(100,0), color=color.red, radius=2)
