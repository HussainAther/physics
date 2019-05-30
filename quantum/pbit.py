import numpy as np

"""
Weighted p-bit uesd in p-circuits and p-computing (p bit pbit pcircuit p circuit pcomputing p computing).
Probabilistic computers (p-computers pcomputers p computers) can operate
at room temperature with existing technology on p-bits that change between
0 and 1. We use an input with stochastic output.
"""

def Ii(C0, C, V0, Vout):
    """
    Current response to weighted p-bit integrating relevant parts of the synapse onto the 
    neurons for C0 input capacitance of the transistor, C is an array of the capacitances
    of each node, V0 is the voltage of the transistor, and Vout is the output voltage
    at each node.
    """
    summ = 0
    for i in C:
        num = i * Vout[i] 
        den = V0 * (C0 + sum(C))
        summ += num/den
    return summ 
