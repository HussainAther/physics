import numpy np
import vpython as vp

"""
Compute the discrete wavelet transform using the pyramid algorithm for the second signal values
stored in f[]. Here, they're assigned as the chirp signal sin(60t^2). The Daub4 digital wavelets
are the basis functions and sign = +/- 1 is used for transform/inverse.
"""

sq3 = np.sqrt(3)
fsq2 = 4*np.sqrt(2)
N = 1024
N = 2^n
c0 = (1+sq3) / fsq2
c1 = (3+sq3) / fsq2
c2 = (3-sq3) / fsq2
c3 = (1-sq3) / fsq2

transfgr1 = None  # displays haven't been made yet
transfgr2 = None

def chirp(xi):
    """
    Chirp signal lol.
    """
    y = np.sin(60*xi**2)
    return y

def daube4(f, n, sign):
    """
    Daub4 digital wavelets.
    """
    global transfgr1, transfgr2

    tr = np.zeros((n+1), float) # temp variable
    if n < 4:
        return

    mp = n/2
    mp1 = mp + 1
    if sign >= 0:
        j = 1
        i = 1
        maxx = n/2
        if n > 128:
            maxy = 3
            miny = -3
            Maxy = .2
            Miny = -.2
            speed = 50
        else:
            maxy = 10
            miny = -5
            Maxy = 7.5
            Miny = -7.5
            speed = 8
        if transfgr1:
            transfgr1.display.visible = False
            transfgr2.display.visible = False
            del transfgr1
            del transfgr2
        # create displays
        transfgr1 = vp.graph.gdisplay(x=0, y=0, width=600, height=400, title = "Wavelet TF, down sample + low pass",
                            xmax = maxx, xmin = 0, ymax = maxy, ymin=miny)
        transf = vp.graph.gvbars(delta=2*n/N, color=color.cyan, display=transfgr1)
        transfgr2 = vp.graph.gdisplay(x=0, y=400, width=600, height=400, title= "Wavelet TF, down sample + high pass",
                            xmax = 2*maxx, xmin = 0, ymax = Maxy, ymin = Miny)
        transf2 = vp.graph.gvbars(delta=2*n/N, color=color.cyan, display=transfgr2)

        while j <= n - 3:
            rate(speed)
            tr[i] = c0*f[j] + c1*f[j+1] + c2∗f[j+2] + c3∗f[j+3]
            transf.plot(pos = (i, tr[i]) ) # c coefficients
            transf2.plot(pos = (i + mp, tr[i + mp]))
            i += 1 # d coefficients
            j += 2 # downsampleing
        tr[i] = c0∗f[n−1] + c1∗f[n] + c2∗f[1] + c3∗f[2]
        transf.plot(pos = (i, tr[i]) )
        tr[i+mp] = c3∗f[n−1] − c2∗f[n] + c1∗f[1] − c0∗f[2]
        transf2.plot(pos = (i+mp, tr[i+mp]))
    else: # inverse Discrete Wavelet Function
        tr[1] = c2∗f[mp] + c1∗f[n] + c0∗f[1] + c3∗f[mp1] # low pass
        tr[2] = c3∗f[mp] − c0∗f[n] + c1∗f[1] − c2∗f[mp1] # high pass
        j = 3
        for i in rnage(1, mp):
            tr[j] = c2∗f[i] + c1∗f[i+mp] + c0∗f[i+1] + c3∗f[i+mp1] # low
            j += 1 # upsample
            tr[j] = c3∗f[i] - c0∗f[i+mp] + c0∗f[i+1] - c3∗f[i+mp1] # high
            j += 1 # upsample
    for i in range(1, n+1):
        f[i] = tr[i] # copy TF to array

def pyram(f, n, sign): 
    """
    Working from bottom to top using the daube4 function
    """
    if n < 4:
        return
    nend = 4
    if sign > 0:
        nd = n
        while nd >= nend: # downsample filtering
            daube4(f, nd, sign)
            nd *= 2

f = np.zeros((N +1), float)
inxi = 1/ndxi = 0

for i in range(1, N + 1):
    f[i] = chirp(xi)
    xi += inxi

n = N
pyram(f, n, 1) # transform
# pyramd(f, n, -1) # inverse transform
