import math
import random
import argparse

"""
Shor's algorithm for quantum integer factorization
"""

class Mapping:
    def __init__(self, state, amplitude):
        self.state = state
        self.amplitude = amplitude

class QuantumState:
    def __init__(self, amplitude, register):
        self.amplitude = amplitude
        self.register = register
        self.entangled = {}

    def entangle(self, fromState, amplitude):
        register = fromState.register
        entanglement = Mapping(fromState, amplitude)
        try:
            self.entangled[register].append(entanglement)
        except KeyError:
            self.entangled[register] = [entanglement]

    def entangles(self, register = None):
        entangles = 0
        if register is None:
            for states in self.entangled.values():
                entangles += len(states)
        else:
            entangles = len(self.entangled[register])

        return entangles

class QubitRegister:
    def __init__(self, numBits):
        self.numBits = numBits
        self.numStates = 1 << numBits
        self.entangled = []
        self.states = [QuantumState(complex(0.0), self) for x in range(self.numStates)]
        self.states[0].amplitude = complex(1.0)
