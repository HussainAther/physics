from pylab import *
import re
pressarr = array([]) # Store pressures
volarr = array([]) # Store volumes
Karr = array([]) # Store kinetic energies myvelocities = array([1.5,2.0,2.5,3.0,3.5]) myvolumes = array([0.010, 0.020, 0.040, 0.080]) for ivel in range(0,size(myvelocities)):
for ivol in range(0,size(myvolumes)):
    # Change the word mydensity to myvolumes[ivol] infile = open("in.gasstatistics30",’r’)
    sintext = infile.read()
    infile.close()
    replacestring = "%f" % (myvolumes[ivol])
    intext2=intext.replace("mydensity",replacestring)
    # Change the word myvelocity to myvelocities[ivel]
    replacestring = "%f" % (myvelocities[ivel])
    intext3=intext2.replace("myvelocity",replacestring)
    infile = open("in.tmp","w")
    infile.write(intext3)
    infile.close()
    # Run the simulator
    print("Executing lammps < in.tmp")
