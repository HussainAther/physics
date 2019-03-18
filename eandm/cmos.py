import numpy as np
import matplotlib.pyplot as plt

"""
We can use complementary metal–oxide–semiconductor (cmos) circuits to implement
functions using logic gates. We can also use metal-oxide-semiconductor field-effect
transistor (mosfet) as a type of field-effect transistor to generate different voltages.

This circuit is on page 6 of the following document: 
https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-004-computation-structures-spring-2009/labs/MIT6_004s09_lab01.pdf
"""

Vmeter = 0 # 1st voltage source
Vds = 0 # 2nd voltage source
Vgs = 0 # 3rd voltage source

result = []

m1 = 1.2e-6 # mosfet length itself in microns

for i in range(0, 6, .1):
    Vds = i
    if i % 1 == 0:
        Vgs = i
    result.append(Vgs - Vds - m1*600) # Vmeter voltage

plt(result)
