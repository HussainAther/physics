import re
import pylab as pl
import pickle as pk

"""
Find the equilibrium states undergoing the lammps simulator 
to apply pressure changes and track the chemical, statistical, 
and thermodynamic quantities.
"""

pressarr = array([]) # Store pressures
volarr = array([]) # Store volumes

Karr = array([]) # Store kinetic energies 
myvelocities = array([1.5,2.0,2.5,3.0,3.5]) 
myvolumes = array([0.010, 0.020, 0.040, 0.080]) for ivel in range(0,size(myvelocities)):

for ivol in range(0,size(myvolumes)):
    # Change the word mydensity to myvolumes[ivol] 
    infile = open("in.gasstatistics30","r")
    sintext = infile.read()
    infile.close()
    replacestring = "%f" % (myvolumes[ivol])
    intext2 = intext.replace("mydensity", replacestring)
    
    # Change the word myvelocity to myvelocities[ivel]
    replacestring = "%f" % (myvelocities[ivel])
    intext3 = intext2.replace("myvelocity", replacestring)
    infile = open("in.tmp","w")
    infile.write(intext3)
    infile.close()
    
    # Run the simulator
    print("Executing lammps < in.tmp")
    os.system("lammps < in.tmp") # Run lammps

    # Extract data from trajectory of simulation
    d = pk.dump("tmpdump.lammpstrj") # Read sim states tmp_time,simbox,atoms,bonds,tris,lines = d.viz(0) dx = simbox[3]-simbox[0]
    dy = simbox[4]-simbox[1]
    vol = dx*dy # Volume of box
    t = d.time(), n = size(t)

    # Calculate total kinetic energy of last timestep vx = array(d.vecs(n-1,"vx"))
    vy = array(d.vecs(n-1,"vy"))
    K = 0.5*sum(vx*vx+vy*vy) # Sum of kinetic energy # Read pressures calculated in simulation
    l = logfile("log.lammps")

    # Find pressure averaged over all timesteps
    press = average(l.get("Press"))

    # Store calculated values in arrays
    pressarr.append(pressarr, press)
    volarr.append(volarr, vol)
    Karr.append(Karr, K)

# Plot the results
pvarr = pressarr*volarr
pl.plot(Karr,pvarr,"o"),xlabel("K"),ylabel("PV"),show()
