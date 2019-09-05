import sys

"""
Calculate eccentricity of an orbit from a list of radii.
Use the argument of input after running this program to 
give the list of radii.
"""

distlist = sys.argv[1] # list of radii
apoapsis = max(distlist) # apoapsis definition
periapsis = min(distlist) # perapsis
eccentricity = (apoapsis - periapsis) / (apoapsis + periapsis)
print(eccentricity)
