import sys, EulerFreeFall
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

t = 0
g = 9.8

parser = argparse.ArgumentParser()

parser.add_argument("--filename", dest="filename", action="store", help="output filename", default=junk.dat")
parser.add_argument("--v_initial", dest="v", type=float, default=0, help="initial velocity")
parser_add_argument("--v_terminal", dest="vterminal", type=float, default=2, help="terminal velocity")
parser.add_argument("--tmax", dest="tmax", type=float, default=.2, hel="maximum simulation time")
parser.add_argument("--df", dest="dt", help="time steps", action="append")
parser.add_argument("--savePlot", dest="savePlot", action="store", default="none", help="Save a hardcopy of plot? (specifify .pdf or .png)")

input = parser.parse_args()
filename = input.filename
v = input.v
vinitial = v
vterminal = input.vterminal
tmax = input.tmax
savePlot = input.savePlot
timeSteps = input.dt

for n in timeSteps:
    dt = float(n)
    fname = filename + str(n) + ".dat"
    outfile = open(fname, "w")
    outfile.write("time (s) \t speed \n")
    outfile.write("%g \t %g\n" % (t, v))
    imax = int(tmax/dt)
    for i in range(imax+1):
        t = i*dt
        v = EulerFreeFall.VelocityLinearDrag(v, dt, 9.8, vterminal)
        outfile.write("%g \t %g\n" % (t, v))
    outfile.close
    data = np.loadtxt(fname, skiprows=1)
    xaxis = data[:, 0]
    yaxis = data[:, 1]
    plt.plot(xaxis, yaxis, marker=".", markersize=7, linestyle="None")
    t = 0
    i = 0
    v = vinitial
