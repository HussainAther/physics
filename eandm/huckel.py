import csv
import numpy as np

"""
The (Huckel) Hückel method or Hückel molecular orbital method (HMO), proposed by Erich Hückel 
in 1930, is a very simple linear combination of atomic orbitals molecular orbitals (LCAO MO) 
method for the determination of energies of molecular orbitals of pi electrons in conjugated 
hydrocarbon systems, such as ethene, benzene and butadiene. It is the theoretical basis for 
the Hückel's rule. The extended Hückel method developed by Roald Hoffmann is computational 
and three-dimensional and was used to test the Woodward–Hoffmann rules. It was later extended 
to conjugated molecules such as pyridine, pyrrole and furan that contain atoms other than carbon, 
known in this context as heteroatoms.

For an input MOLPRO file and Cartesian coordinates, form a Huckel Hamiltonian matrix
and return the eigenvalues that we plot against the normalized eigenvalues ordinal numbers.
"""

a = 0 # number of carbon atoms to be deleted or removed from the list
inter = range(10) # interval distances
with open("molproout.csv") as file:
    copy = False
    for line in file:
        if line.strip() == "NR  ATOM    CHARGE       X              Y              Z":
            copy = True
        elif line.strip() == "Bond lengths in Bohr (Angstrom)":
            copy = False
        elif copy:
            open("coord", "w").write(line)

open("coord", "w").close()
