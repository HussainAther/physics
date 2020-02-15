import matplotlib.pyplot as plt
import numpy as np

"""
Bifurcations are transitions between dynamical states used in nonlinear dynamics.
This is a saddle-node bifurcation defined by
dx/dt = r-x^2 
It has equilibrium points at x_eq = +/- sqrt(r)
and critical condition found by taking the derivative of dx/dt = F(x)
so we get
dF/dx = -2x
so the bifurcation occurs at x=x_eq, which is
dF/dx = 0
"""

def xeq1(r):
    """
    Stable equilibrium
    """
    return np.sqrt(r)

def xeq2(r):
    """
    Unstable equilibrium
    """
    return np.sqrt(r)

# Plot.
fig = plt.figure(figsize=(9,6))
ax1 = fig.add_subplot(1, 1, 1)
domain = linspace(0, 10)
ax1.plot(domain, xeq1(domain), "b-", label = "stable equilibrium", linewidth = 3)
ax1.plot(domain, xeq2(domain), "r--", label = "unstable equilibrium", linewidth = 3)
ax1.legend(loc="upper left")
# Neutral equilibrium point
ax1.plot([0], [0], "go")
ax1.axis([-10, 10, -5, 5])
ax1.set_xlabel("r")
ax1.set_ylabel("x_eq")
ax1.set_title("Saddle-node bifurcation")

# Add black arrows indicating the attracting dynamics of the stable and the 
# repelling dynamics of the unstable equilibrium point. 
ax1.annotate("", xy=(-7, -4), xytext=(-7, 4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
ax1.annotate("", xy=(-5, -4), xytext=(-5, 4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
ax1.annotate("", xy=(-3, -4), xytext=(-3, 4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
ax1.annotate("", xy=(-1, -4), xytext=(-1, 4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
ax1.annotate("", xy=(1, -4), xytext=(1, -1.5), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
ax1.annotate("", xy=(1, 0.7), xytext=(1, -0.7), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
ax1.annotate("", xy=(1, 1.5), xytext=(1, 4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
ax1.annotate("", xy=(3, -4), xytext=(3, -2), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
ax1.annotate("", xy=(3, 1.5), xytext=(3, -1.5), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
ax1.annotate("", xy=(3, 2), xytext=(3, 4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
ax1.annotate("", xy=(5, -4), xytext=(5, -2.5), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
ax1.annotate("", xy=(5, 2), xytext=(5, -2), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
ax1.annotate("", xy=(5, 2.5), xytext=(5, 4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
ax1.annotate("", xy=(7, -4), xytext=(7, -3), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
ax1.annotate("", xy=(7, 2.3), xytext=(7, -2.3), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
ax1.annotate("", xy=(7, 3), xytext=(7, 4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)

"""
A transcritical bifurcation happens when the equilibrium point "passes through" another one that exchanges their
stabilities. A supercritical pitchfork bifurcation can make a stable equilibrium point split into two stable
and one stable equilibriums.

In this case, we use equation dx/dt = rx - x^3 which has three equilibrium points x_eq = 0, +/- sqrt(r) with
the latter two points existiing only for r >= 0.
"""

def xeq1(r):
    """
    First equilibrium point
    """
    return 0

def xeq2(r):
    """
    Second
    """
    return np.sqrt(r)

def xeq3(r):
    """
    Third
    """
    return -np.sqrt(r)

# Plot.
domain1 = linspace(-10, 0)
domain2 = linspace(0, 10)
plt.plot(domain1, xeq1(domain1), "b-", linewidth = 3)
plt.plot(domain2, xeq1(domain2), "r--", linewidth = 3)
plt.plot(domain2, xeq2(domain2), "b-", linewidth = 3)
plt.plot(domain2, xeq3(domain2), "b-", linewidth = 3)
# Neutral equilibrium point
plt.plot([0], [0], "go")
plt.axis([-10, 10, -5, 5])
plt.xlabel("r")
plt.ylabel("x_eq")
plt.title("Supercritical pitchfork bifurcation")

# Add arrows.
plt.annotate("", xy=(0, -1), xytext=(0, -4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(0, 1), xytext=(0, 4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(-5, -0.5), xytext=(-5, -4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(-5, 0.5), xytext=(-5, 4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(3, 1.5), xytext=(3, 0.5), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(3, -1.5), xytext=(3, -0.5), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(3, 2.2), xytext=(3, 4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(3, -2.2), xytext=(3, -4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(7, 2), xytext=(7, 0.5), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(7, -2), xytext=(7, -0.5), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(7, 3), xytext=(7, 4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(7, -3), xytext=(7, -4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)

"""
A subcritical pitchfork bifurcation causes the unstable equilibrium point to split into two unstable
and one stable equilibriums.
The equation dx/dt = rx+x^3 has three equilibrium points x_eq = 0 and x_eq = +/- sqrt(r). 
"""

def xeq1(r):
    return 0

def xeq2(r):
    return np.sqrt(-r)

def xeq3(r):
    return -np.sqrt(-r)

# Plot.
domain1 = linspace(-10, 0)
domain2 = linspace(0, 10)
plt.plot(domain1, xeq1(domain1), "b-", linewidth = 3)
plt.plot(domain1, xeq2(domain1), "r--", linewidth = 3)
plt.plot(domain1, xeq3(domain1), "r--", linewidth = 3)
plt.plot(domain2, xeq1(domain2), "r--", linewidth = 3)
# Neutral equilibrium point
plt.plot([0], [0], "go")
plt.axis([-10, 10, -5, 5])
plt.xlabel("r")
plt.ylabel("x_eq")
plt.title("Subcritical pitchfork bifurcation")

# Black arrows
plt.annotate("", xy=(1, -4), xytext=(1, -1), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(1, 4), xytext=(1, 1), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(5, -4), xytext=(5, -0.5), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(5, 4), xytext=(5, 0.5), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(-3, 0.5), xytext=(-3, 1.5), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(-3, -0.5), xytext=(-3, -1.5), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(-3, 4), xytext=(-3, 2.2), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(-3, -4), xytext=(-3, -2.2), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(-7, 0.5), xytext=(-7, 2), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(-7, -0.5), xytext=(-7, -2), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(-7, 4), xytext=(-7, 3), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(-7, -4), xytext=(-7, -3), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)

"""
Combined bifurcations use the system dx/dt = r+x-x^3 such that, when you solve dx/dt = 0 in terms of r, you get
r = -x+x^3 to draw the bifurcation diagram.

You can use the Jacobian matrix to get the stability information. This system shows hysteresis.
"""

def xeq1(r):
    return -r + r**3

# Plot.
domain1 = linspace(-1.3, -sqrt(1/3.))
domain2 = linspace(-sqrt(1/3.), sqrt(1/3.))
domain3 = linspace(sqrt(1/3.), 1.3)
plt.plot(xeq1(domain1), domain1, "b-", linewidth = 3)
plt.plot(xeq1(domain2), domain2, "r--", linewidth = 3)
plt.plot(xeq1(domain3), domain3, "b-", linewidth = 3)
plt.axis([-1, 1, -1.5, 1.5])
plt.xlabel("r")
plt.ylabel("x_eq")
plt.title("Combination of two saddle-node bifurcations")

plt.annotate("", xy=(0.75, 1.2), xytext=(0.75, -1.4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(0.5, 1.1), xytext=(0.5, -1.4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(0.5, 1.25), xytext=(0.5, 1.4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(0.25, -0.9), xytext=(0.25, -1.4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(0.25, -0.8), xytext=(0.25, -0.3), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(0.25, 1), xytext=(0.25, -0.1), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(0.25, 1.15), xytext=(0.25, 1.4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(0, -1.05), xytext=(0, -1.4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(0, -0.9), xytext=(0, -0.1), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(0, 0.9), xytext=(0, 0.1), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(0, 1.05), xytext=(0, 1.4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(-0.75, -1.2), xytext=(-0.75, 1.4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(-0.5, -1.1), xytext=(-0.5, 1.4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(-0.5, -1.25), xytext=(-0.5, -1.4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(-0.25, 0.9), xytext=(-0.25, 1.4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(-0.25, 0.8), xytext=(-0.25, 0.3), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(-0.25, -1), xytext=(-0.25, 0.1), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)
plt.annotate("", xy=(-0.25, -1.15), xytext=(-0.25, -1.4), arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1),)

"""
The Hopf bifurcation lets the limit cycle appear around the equilibiurm point. It makes a cyclic, closed trajectory
in the phase space. The van der Pol oscillator shows this with the second-order differential equation
d^2x/dt^2 + r(x^2-1)dx/dt + x = 0
in which we introduce y = dx/dt to make the system first-order
dx/dt = y
dy/dt = -r(x^2-1)y-x with (0, 0) as the only equilibrium point of hte system.
The Jacobian matrix is 
[0 1 -1 r] and you can calculate the eigenvalues as 
|0-λ 1 -1 r-λ| = 0 
such that λ = r+/-sqrt(r^2-4)/2
and the critical condition is Re(λ) = 0

This code will iterate through five values of r and show the appearance of the limit cycle at r = 0.
"""

dt = 0.01

# prepare plots
fig = plt.figure(figsize=(18,6))

def plot_phase_space():
    x = y = 0.1
    xresult = [x]
    yresult = [y]
    for t in range(10000):
        nextx = x + y * dt
        nexty = y + (-r * (x**2 - 1) * y - x) * dt
        x, y = nextx, nexty
        xresult.append(x)
        yresult.append(y)
    plt.plot(xresult, yresult)
    plt.axis("image")
    plt.axis([-3, 3, -3, 3])
    plt.title("r = " + str(r))

rs = [-1, -0.1, 0, .1, 1]
for i in range(len(rs)):
    fig.add_subplot(1, len(rs), i + 1)
    r = rs[i]
    plot_phase_space()
