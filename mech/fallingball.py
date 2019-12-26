import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime

"""
This program uses the Euler method to solve for the motion of a
ball dropped from rest close to the surface of Earth . The
program depends on the function VelocityLinearDrag to execute
the Euler method.
"""

# Helper functions
def VelocityLinearDrag(v, dt, g=9.8, vt=30):
    return v + g*(1-v/vt)*dt

def PositionLinearDrag(x, v, dt):
    return x + v*dt

def newVelocityNoDrag(v, dt=.01, g=9.8):
    return v - g*dt

def newPositionNoDrag(y, v, dt):
    return y + v*dt

# Initialize variables.
t = 0
g = 9.8

# Parse arguments.
parser = argparse.ArgumentParser()

parser.add_argument("--filename", dest="filename", action="store", help="output filename", default=junk.dat")
parser.add_argument("--v_initial", dest="v", type=float, default=0, help="initial velocity")
parser_add_argument("--v_terminal", dest="vterminal", type=float, default=2, help="terminal velocity")
parser.add_argument("--tmax", dest="tmax", type=float, default=.2, hel="maximum simulation time")
parser.add_argument("--df", dest="dt", help="time steps", action="append")
parser.add_argument("--savePlot", dest="savePlot", action="store", default="none", help="Save a hardcopy of plot? (specifify .pdf or .png)")

# Read parsed arguments.
input = parser.parse_args()
filename = input.filename
v = input.v
vinitial = v
vterminal = input.vterminal
tmax = input.tmax
savePlot = input.savePlot
timeSteps = input.dt

# Simulate.
for n in timeSteps:
    dt = float(n)
    fname = filename + str(n) + ".dat"
    outfile = open(fname, "w")
    outfile.write("time (s) \t speed \n")
    outfile.write("%g \t %g\n" % (t, v))
    imax = int(tmax/dt)
    for i in range(imax+1):
        t = i*dt
        v = VelocityLinearDrag(v, dt, 9.8, vterminal)
        outfile.write("%g \t %g\n" % (t, v))
    outfile.close
    data = np.loadtxt(fname, skiprows=1)
    xaxis = data[:, 0]
    yaxis = data[:, 1]
    plt.plot(xaxis, yaxis, marker=".", markersize=7, linestyle="None")
    t = 0
    i = 0
    v = vinitial

# Plot.
legendstr = []
for timestep in timeSteps:
    legendstr.append("dt = " + timestep)
legendstring = str(legendstr[:])
legendstring = legendstring.strip("[ ]") + ", Analytic Solution"
plt.legend(legendstring.split(", "), loc="best")
plt.xlabel("time (s)", fontsize = 18)
plt.ylabel("velocity (m/s)", fontsize = 18)
plt.title("Vertically Dropped Ball", fontsize = 18)
plt.ylim(0, .21)

# Save.
if savePlot == ".pdf":
    now = datetime.datetime.now()
    fname = str(now.year) + str(now.month) + str(now.day) + "-" + str(now.hour) + str(now.minute) + str(now.second) + ".pdf"
    plt.savefig(fname, dpi=600)
elif savePlot = ".png":
    now = datetime.datetime.now()
    fname = str(now.year) + str(now.month) + str(now.day) + "-" + str(now.hour) + str(now.minute) + str(now.second) + ".png"
    plt.savefig(fname, dpi=600)

# Analytic function for comparison
a = np.arange(0, tmax, dt)
c = vterminal*(1-np.exp(-g*a/vterminal))
plt.plot(a, c, "l--")
plt.show()
