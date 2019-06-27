import cmath
import numpy as np

"""
More circuit stuff.
"""

# Initialize.
E1 = 50 # in Volts
thetae1  = 0 # in degrees
r = 5 # in ohm
R1 = 20 # in ohm
L1 = 0.2 # in Henry
R = 8 # in ohm
L = 0.1 # in Henry
L2 = 0.4 # in Henry
R2 = 25 # in ohm
RL = 20 # in ohm
M = 0.1 # in Henry
f = 75/np.pi # in Hz

# Kirchhoff's law (kirchhoff) to primary circuit
w = 2*np.pi*f

# Solve both primary and secondary circuit.
I2 = E1/((r + R1 + 1j*w*L1 + R + 1j*w*L)*(R2 + RL + 1j*w*L2 + R + 1j*w*L)/(1j*w*M + R + 1j*w*L) + (-1*(1j*w*M + R + 1j*w*L)))
I1 = I2*(R2 + RL + 1j*w*L2 + R + 1j*w*L)/(1j*w*M + R + 1j*w*L)

# Reverse direction
I2r = E1/((r + R1 + 1j*w*L1 + R + 1j*w*L)*(R2 + RL + 1j*w*L2 + R + 1j*w*L)/(-1*1j*w*M + R + 1j*w*L) + (-1*(-1*1j*w*M + R + 1j*w*L)))
I1r = I2r*(R2 + R + 1j*w*L2 + R + 1j*w*L)/(-1*1j*w*M + R + 1j*w*L)

# Results
print("\n\n  Result  \n\n")
print("primary  current  I1  is  ",round(abs(I1),2),"/_",round(cmath.phase(complex(I1.real,I1.imag))*180/math.pi,2),"deg  A")
print("load  current  I2  is  ",round(abs(I2),2),"/_",round(cmath.phase(complex(I2.real,I2.imag))*180/math.pi,2),"deg   A")
print("reversed  primary  current  I1r  is  ",round(abs(I1r),2),"/_",round(cmath.phase(complex(I1r.real,I1r.imag))*180/math.pi,2),"deg   A")
print("reversed  load  current  I2r  is  ",round(abs(I2r),2),"/_",round(cmath.phase(complex(I2r.real,I2r.imag))*180/math.pi,2),"deg   A")
