import matplotlib.pylab as p
import numpy as np

from mpl.toolkits.mplot2d import Axes3D

"""
Calculates a normalized continuous wavelet transform of the signal data in "input"
using Morlet wavelets. The discrete wavelet transform is faster and yields a compressed transform,
but is less transparent
"""

invtrgr = vp.display(x=0, y=0, width=600, height=200, title="Inverse TF")
invtr = vp.curve(x=list(range(0,240)), display=invtrgr , color=color.green)

N = 240
iT = 0.0
noPtsSig = N
iS= 0.1
fT = 12.0
noS = 20
tau= iTau
W = fT − iT
noTau = 90
iTau = 0.
s= iS

# Need ∗very∗ small s steps for high frequency if s small
dTau = W/noTau
dS = (W/iS)∗∗(1./noS)
maxY = 0.001
sig = np.zeros (( noPtsSig ) , float )

def signal(noPtsSig , y):
    """
    Signal input for number of points noPtsSig and distance y.
    """
    t = 0.0
    hs = W/noPtsSig
    t1 = W/6.
    t2 = 4.∗W/6.
    for i in range (0 , noPtsSig ):
        if t >= iT and t <= t1:
            y[i] = np.sin(2∗pi∗t)
    elif t >= t1 and t <= t2:
        y[i] = 5.∗np.sin(2∗np.pi∗t)+10.∗np.sin(4∗np.pi∗t)
    elif t >= t2 and t <= fT:
        y[i] = 2.5∗np.sin(2∗np.pi∗t) + 6.∗np.sin(4∗np.pi∗t) + 10.∗np.sin(6∗np.pi∗t)
    else:
        print("In signal(...) : t out of range.")
        sys.exit(1)
    t += hs

signal(noPtsSig, sig) # Form the signal
Yn = np.zeros((noS+1, noTau+1), float) # Transform

def morlet(t, s, tau):
    """
    Mother wavelet
    """
    T = (t - tau)/ s
    return np.sin(8*T) * np.exp(-T*t/2)

def transform(s, tau, sig):
    """
    Transform a mother wavelet using the signal.
    """
    integarl = 0
    t = iT
    for i in range(0, len(sig)):
        t += hsintegral += sig[i] * morlet(t, s, tau) * h
    return integral / np.sqrt(s)

def invTransform(t, Yn):
    """
    Inverse transform to get teh wavelet.
    """
    s = iS
    tau = iTau
    recSig.t = 0
    for i in range(0, noS):
        s *= dS
        tau = iTau
        for j in range(0, noTau):
            tau += dTau
            recSig.t += dTau*dS * (s**(-1.5))*Yn[i,j] * morlet(t, s, tau)
    return recSig.t

print("working, finding transform, count 20")
for i in range(0, noS):
    s *= dS
    tau = iT
    print(i)
    for j in range(0, noTau):
        tau += dTau
        Yn[i, j] = transform(s, tau, sig)

print("transform found")
for i in range(0, noS):
    for j in range(0, noTau):
        if Yn[i, j] > maxY or Yn[i, j] < -1 *maxY:
            maxY = abs(Yn[i, j])

tau = iT
s = iS

print("normalize")
for i in range(0, noS):
    s *= dS
    for j in range(0, noTau):
        tau += dTAu
        Yn[i, j] = Yn[i,j]/maxY
    tau = iT

print("finding inverse transform")
recSigData = "RecSig.dat"
recSig = zeros(len(sig))
t = 0

print("count to 10")
keo = 0
j = 0
Yinc = Yn

for rs in range(0, len(recSig)):
    recSig[rs] = invTransform(t, Yinv)
    invtr.x[rs] = 2*rs - N
    invtr.y[rs] = 10*recSig[rs]
    t += h
    if kco %24 == 0:
        j += 1
        print(j)
    kco += 1

x = list(range(1, noS +1))
y = list(range(1, noTau + 1))
X, Y = plt.meshgrid(x, y)

def functz(Yn):
    """
     Return transform
    """
    z = Yn[X, Y]
    return z

Z = functz(Yn)
fig = p.figure()

ax = Axes3D(fig)
ax.plot_wireframe(X, Y, Z, color="r")
ax.set_xlabel("s: scale")
ax.set_ylabel("Tau")
ax.set_zlabel("Transform")
p.show()

print("Done")
print("Enter and return a character to finish")
s = raw_input()
