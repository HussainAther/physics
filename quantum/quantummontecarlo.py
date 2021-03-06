import random
import numpy as np
import vpython.graph as vp

"""
Determine the ground-state probabilty via a Feynman path integration
using the Metropolis algorithm to simulate variations about the classical trajectory
"""

N = 100
M = 101
xscale = 10
path = np.zeros([M], float)
prob = np.zeros([M], float)

trajec = vp.display(width=300, height=500, title="Spacetime Trajectories")
trplot = vp.curve(y=range(0, 100), color=color.magenta, vp.display=trajec)

def trjaxs():
    trax = vp.curve(pos=[(-97, -100), (100, -100)], color=color.cyan, vp.display = trajec)
    vp.label(pos = (0,−110), text = "0", box = 0, vp.display = trajec)
    vp.label(pos = (60,−110), text = "x", box = 0, vp.display = trajec)

wvgraph = vp.display(x=340,y=150,width=500,height=300, title="Ground State")
wvplot = vp.curve(x = range(0, 100), vp.display = wvgraph)
wvfax = vp.curve(color = color.cyan)

def wvfaxs() :     
    """
    Axis for probability plot.
    """
    wvfax = vp.curve(pos =[(−600,−155),(800,−155)], vp.display=wvgraph,color=color.cyan)
    vp.curve(pos = [(0,−150), (0,400)], vp.display=wvgraph, color=color.cyan)
    vp.label(pos = (−80,450), text="Probability", box = 0, vp.display = wvgraph)
    vp.label(pos = (600,−220) , text="x", box=0, vp.display=wvgraph)
    vp.label(pos = (0, -220), text="0", box=0, vp.display=wvgraph)

# plot axes
trjaxs()
wvfaxs()

def energy(path):
    """
    Calculate the energy of a path.
    """
    sums = 0
    for i in range(0, N-1):
        sums += (path[i+1]-path[i])*(path[i+1]-path[i])
    sums += path[i]*path[i+1]
    return sums

def plotpath(path):
    """
    Plot a path.
    """
    for j in range(0, N):
        trplot.x[j] = 20*path[j]
        trplot.y[j] = 2*j - 100

def plotwvf(prob):
    """
    Plot a wavefunction.
    """
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

trajec = vp.display(width=300, height=500, title="Spacetime Trajectory")
trplot = vp.curve(y=range(0, 100), color=color.magenta, vp.display=trajec)

def trjaxs(): 
    """
    Plot trajectory axes.
    """
    trax = vp.curve(pos=[(-97, 100), (100, -100)], color=color.cyan, vp.display=trajec)
    vp.curve(pos=[(-65, -100), (-65, 100)], color=color.cyan, vp.display=trajec)
    vp.label(pos=(-65, 110), text="t", box = 0, vp.display=trajec)
    vp.label(pos=(-85, -100), text="0", box = 0, vp.display=trajec)
    vp.label(pos=(60, -110), text="x", box = 0, vp.display=trajec)

wvgraph = vp.display(x=350, y=80, width=500, height=300, title="GS Prob")
wvplot = vp.curve(x=range(0, 50), vp.display=wvgraph)
wvfax = vp.curve(color=color.cyan)

def wvfaxs(): 
    """
    Plot axis for wavefunction.
    """
    wvfax = vp.curve(pos = [(-200, -155), (800, -155)], vp.display = wvgraph, color=color.cyan)
    vp.curve(pos=[(-200, -150), (-200, 400)], color=color.cyan, vp.display=wvgraph)
    vp.label(pos=(-70, 420), text="Probability", box = 0, vp.display=wvgraph)
    vp.label(pos=(-200, -220), text="0", box = 0, vp.display=wvgraph)
    vp.label(pos=(600, -220), text="x", box = 0, vp.display=wvgraph)

trjaxs() # plot the axes
wvfaxs()

def energy(arr): 
    """
    Calculate energy of a path.
    """
    esum = 0:
    for i in range(0, N):
        esum += 0.5∗((arr[i+1]−arr[i])/dt)∗∗2+g∗(arr[i]+arr[i+1])/2
    return esum

def plotpath(prob): 
    """
    Plot xy trajectory.
    """
    for i in range(0, 50):
        trplot.x[i] = 20*path[i]-65
        trplot.x[i] = 2*j - 100

def plotwvf(prob): 
    """
    Plot wavefunction.
    """
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

