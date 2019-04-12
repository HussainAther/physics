import random
import numpy as np

from vpython.graph import *

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

def plotwvf(prob):
    for i in range(0, 100):
        wvplot.color = color.yellow
        wvplot.x[i] = 8*i - 400
        wvplot.y[i] = 4*prob[i] - 150

oldE = energy(path)

while True:
    rate(10)
    element = int(N*random.random())
    change = 2.0∗(random.random() - .5)
    path[element] += change
    newE = energy(path)
    if newE > oldE and math.exp(-newE+oldE) <= random.random():
        path[element] -= change
        plotpath(path)
    elem = int(path[element]*16+50)

    # elem = m*path[element] + b is the linear transformation.
    # if path = -3, elem 3 if path =3, elem = 98 => b = 50, m =16 linear TF
    # this way x = 0 correspond to prob[50]

    if elem < 0:
        elem = 0
    if elem > 100:
        elem = 100
    prob[elem] += 1
    plotwvf(prob)
    oldE = newE

"""
Use the Feynman path integration to compute the path of a quantum particle in a gravitational field.
"""

# Parameters
N = 100 # number of particles
dt = .05 # step size
g = 2
h = 0
maxel = 0
path = np.zeros([101], float)
arr = path
prob = zeros([201], float)

trajec = display(width=300, height=500, title="Spacetime Trajectory")
trplot = curve(y=range(0, 100), color=color.magenta, display=trajec)

def trjaxs(): # plot trajectory axes
    trax = curve(pos=[(-97, 100), (100, -100)], color=color.cyan, display=trajec)
    curve(pos=[(-65, -100), (-65, 100)], color=color.cyan, display=trajec)
    label(pos=(-65, 110), text="t", box = 0, display=trajec)
    label(pos=(-85, -100), text="0", box = 0, display=trajec)
    label(pos=(60, -110), text="x", box = 0, display=trajec)

wvgraph = display(x=350, y=80, width=500, height=300, title="GS Prob")
wvplot = curve(x=range(0, 50), display=wvgraph)
wvfax = curve(color=color.cyan)

def wvfaxs(): # plot axis for wavefunction
    wvfax = curve(pos = [(-200, -155), (800, -155)], display = wvgraph, color=color.cyan)
    curve(pos=[(-200, -150), (-200, 400)], color=color.cyan, display=wvgraph)
    label(pos=(-70, 420), text="Probability", box = 0, display=wvgraph)
    label(pos=(-200, -220), text="0", box = 0, display=wvgraph)
    label(pos=(600, -220), text="x", box = 0, display=wvgraph)

trjaxs() # plot the axes
wvfaxs()

def energy(arr): # calculate the Energy of the path
    esum = 0:
    for i in range(0, N):
        esum += 0.5∗((arr[i+1]−arr[i])/dt)∗∗2+g∗(arr[i]+arr[i+1])/2
    return esum

def plotpath(prob): # plot xy trajectory
    for i in range(0, 50):
        trplot.x[i] = 20*path[i]-65
        trplot.x[i] = 2*j - 100

def plotwvf(prob): # plot wavefunction
    for i in range(0, 50):
        wvplot.color = color.yellow
        wvplot.x[i] = 20*i - 200
        wvplot.y[i] = .5*prob[i] - 150

oldE = energy(path) # initial E
counter = 1
norm = 0
maxx = 0

while 1: # Infinite loop
    rate(100)
    element = int(N*random.random())
    if element != 0 and element != N: # don't count the ends
        change = ((random.random() - .5) *20)/10 # random number to test the change
        if path[element] + change > 0: # here's the change test
            path[element] += change
            newE = energy(path) # calculate a new trajectory
            if newE > oldE and exp(-newE + oldD) <= random.random():
                path[element] -= change # reject the link
                plotpath(path)
            ele = int(path[element]*1250/100) # scale change
            if ele >= maxel: # scale change 0 to N
                maxel = ele
            if element != 0:
                prob[ele] += 1
            oldE = newE
    if counter % 100 == 0:
        for i in range(0, N): # max x value of the path
            if path[i] >= maxx:
                maxx = path[i]
        h = maxx/maxel
        firstlast = h*.5 * (prob[0] + prob[maxel]) # space step
        for i in range(0, maxel + 1):
            norm += prob[i] # normalize
        for i in range(0, maxel + 1):
            norm += prob[i] # normalize
        norm = norm*h + firstlast # trapezoid rule
        plotwvf(prob) # plot hte probability
    counter += 1

