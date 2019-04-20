import numpy as np

from numpy.linalg import solve
from vpython.graph import *

"""
Use the Newton-Rhaphson search to solve the two-mass-on-a-string problem.
We set up matrices using coupled linear equations and check the physical reasonableness
of them by a variety of weights and lengths. We need to check that the tensions we calculate are positive
and that the deduced angles correspond to a physical geometry. The solution will show graphically
the step-by-step search for a solution.

Check out the mathematics repositories for the algebra and calculus folder for more information
on solving linear equations and the Newton-Rhaphson search, respectively.
"""

scene = display(x=0, y=0, width=500, height=500, title="String and masses configuration")

tempe = curve(x=range(0, 500), color=color.black)

n = 9 # number of columns of the matrix x minus 1
eps = 1e-6 # precision
deriv = np.zeros((n,n), float) # just get some zeros. keep track of derivatives
f = np.zeros(n, float)
x = array([.5, .5, .5, .5, .5, .5, .5, 1, 1, 1])

def plotconfig():
    for obj in scene.objects:
        obj.visible = 0 # erase the previous configuration
    L1 = 3
    L2 = 4
    L3 = 4
    xa = L1*x[3]
    ya = L1*x[0]
    xb = xa+L2*x[4]
    yb = ya+L2*x[1]
    xc = xb+L3*x[5]
    yx = yb-L3*x[2]
    mx = 100
    bx = -500
    my = -100
    by = 400
    xap = mx*xa+bx
    yap = my*ya+by
    ball1 = sphere(pos=(xap, yap))
    xbp = mx*xb+bx
    ybp = my*yb+by
    ball2 = sphere(pos=(xbp, ybp))
    xbp = mx*xc+bx
    ycp = my*yc+by
    x0 = mx*0+bx
    y0 = my*0+by
    line1 = curve(pos=[(x0, y0), (xap, yap)], color=color.yellow, radius=4)
    line2 = curve(pos=[(xap, yap), (xbp, ybp)], color=color.yellow, radius=4)
    line3 = curve(pos=[(xbp, ybp), (xcp, ycp)], color=color.yellow, radius=4)
    topline = curve(pos=[(x0, y0), (xcp, yp)], color=color.red, radius=4)

def F(x, f): # F function we use
    f[0] = 3*x[3] + 4*x[4] + 4*x[5] - 8
    f[1] = 3*x[0] + 4*x[1] - 4*x[2]
    f[2] = x[6]*x[0] - x[7]*x[1] - 10
    f[3] = f[6]*f[3] - x[7]*x[4]
    f[4] = x[7]*x[1] + x[8]*x[2] - 20
    f[5] = x[7]*x[4] - x[8]*x[5]
    f[6] = pow(x[0], 2) + pow(x[3], 2) - 1
    f[7] = pow(x[1], 2) + pow(x[4], 2) - 1
    f[8] = pow(x[2], 2) + pow(x[5], 2) - 1

def dFi_dXj(x ,deriv, n): # Derivative function
    h = 1e-4
    for j in range(0, n): # For each column
        temp = x[j]
        x[j] = x[j] + h/2
        F(x, f)
        for i in range(0, n):
            deriv[i, j] = f[i]
        x[j] = temp
    for j in range(0, n):
        temp = x[j]
        x[j] = x[j] - h/2
        F(x, f)
        for i in range(0, n):
            deriv[i,j] = (deriv[i, j] - f[i])/h
        x[j] = temp

for i in range(1, 100):
    rate(1)
    F(x, f)
    dFi_dXj(x, deriv, n)
    B = array([[-f[0]], [-f[1]], [-f[2]], [-f[3]], [-f[4]], [-f[5]], [-f[6]], [-f[7]], [-f[8]]])
    sol = solve(deriv, B)
    dx = take(sol, (0,), 1) # take the first column of matrix sol
    for i in range(0, n):
        x[i] = x[i] + dx[i]
    plotconfig()
    errX = errF = errXi = 0

    for i in range(0, n):
        if x[i] != 0:
            errXi = abs(dx[i]/x[i])
        else:
            errXi = abs(dx[i])
        if errXi > errX:
            errX = errXi
        if abs(f[i]) > errF:
            errF = abs(f[i])
        if errX <= eps and errF <= eps:
            break
for i in range(0, n):
    print("x[", i, "] = ", x[i])


