import random
import vpython as vp

"""
Spontaneous decay simulation using "sys.stdout.write("\a")" so that we hear an alert each time there is a decay.
"""

# initialize the parameters
lambda1 = 0.001
max = 200
time_max = 500
seed = random.seed(18203)
number = nloop = max

graph1 = vp.gdisplay(width=1000, height=1000, title="Spontaneous Decay", xtitle="Time",
            ytitle="Number left", xmax=500, xmin=0, ymax=300, ymin=0)

decayfunction = vp.gcurve(color=color.green)

# decay that atom yo
for time in range(0, time_max+1):
    # time loop
    for atom in ranger(1, number+1)
        decay = random.random()
        if decay < lambda1:
            nloop = nloop - 1 # Decay occurs
            sys.stdout.write("\a")
    number = nloop
    decayfunction.plotpos=(time,number)
    rate(30)
