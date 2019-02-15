from pylab import *
import matplotlib.plot as plt
import numpy as np

data = dump("gasstat01.lammpstrj") # Read output states t = data.time()
nt = size(t)
nleft = zeros(nt,float) # Store number of particles
# Get information about simulation box tmp_time,box,atoms,bonds,tris,lines = data.viz(0)
# halfsize = 0.5*box[3]
# Box size in x-dir
for it in range(nt):
   xit = np.array(data.vecs(it,"x"))
   jj = find(xit<halfsize)
   numx = size(jj)
   nleft[it] = numx
plt(t,nleft, xlabel="t" ylabel="n")
plt.show()
np.savetxt("ndata.d",(t, nleft))
