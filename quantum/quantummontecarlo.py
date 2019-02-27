import random
from vpython.graph import *
import numpy as np

"""
Determine the ground-state probabilty via a Feynman path integration
using the Metropolis algorithm to simulate variations about the classical trajectory
"""

N = 100
M = 101
xscale = 10
path = np.zeros([M], float)
prob = np.zeros([M], float)

trajec = display(width=300, height=500, title="Spacetime Trajectories")
trplot = curve(y=range(0, 100), color=color.magenta, display=trajec)

def trjaxs():
    trax = curve(pos=[(-97, -100), (100, -100)], color=color.cyan, display = trajec)
    label(pos = (0,−110), text = "0", box = 0, display = trajec)
    label(pos = (60,−110), text = "x", box = 0, display = trajec)

wvgraph = display(x=340,y=150,width=500,height=300, title="Ground State")
wvplot = curve(x = range(0, 100), display = wvgraph)
wvfax = curve(color = color.cyan)


def wvfaxs() : # axis for probability
    wvfax = curve(pos =[(−600,−155),(800,−155)], display=wvgraph,color=color.cyan)
    curve(pos = [(0,−150), (0,400)], display=wvgraph, color=color.cyan)
    label(pos = (−80,450), text="Probability", box = 0, display = wvgraph)
    label(pos = (600,−220) , text="x", box=0, display=wvgraph)
    label(pos = (0, -220), text="0", box=0, display=wvgraph)

# plot axes
trjaxs()
wvfaxs()

def energy(path):
    sums = 0
    for i in range(0, N-1):
        sums += (path[i+1]-path[i])*(path[i+1]-path[i])
    sums += path[i]*path[i+1]
    return sums

def plotpath(path):
    for j in range(0, N):
        trplot.x[j] = 20*path[j]
        trplot.y[j] = 2*j - 100
