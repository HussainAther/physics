from vpython.graph import *
import random

"""
Spontaneous decay simulation using "sys.stdout.write("\a")" so that we hear an alert each time there is a decay.
"""

lambda1 = 0.001
max = 200
time_max = 500
seed = random.seed(18203)
number = nloop = max

graph1 = gdisplay(width=1000, height=1000, title="Spontaneous Decay", xtitle="Time",
            ytitle="Number left", xmax=500, xmin=0, ymax=300, ymin=0)

decayfunction = gcurve(color=color.green)

# Decay that atom yo
for time in range(0, time_max+1):
    # Time loop
    for atom in ranger(1, number+1)


