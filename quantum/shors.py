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

    def propagate(self, fromRegister = None):
        if fromRegister is not None:
            for state in self.states:
                amplitude = complex(0.0)

                try:
                    entangles = state.entangled[fromRegister]
                    for entangle in entangles:
                        amplitude += entangle.state.amplitude * entangle.amplitude

                    state.amplitude = amplitude
                except KeyError:
                    state.amplitude = amplitude

        for register in self.entangled:
            if register is fromRegister:
                continue

            register.propagate(self)

    # Map will convert any mapping to a unitary tensor given each element v
    # returned by the mapping has the property v * v.conjugate() = 1
    #

    def map(self, toRegister, mapping, propagate = True):
        self.entangled.append(toRegister)
        toRegister.entangled.append(self)

        # Create the covariant/contravariant representations
        mapTensorX = {}
        mapTensorY = {}
        for x in range(self.numStates):
            mapTensorX[x] = {}
            codomain = mapping(x)
            for element in codomain:
                y = element.state
                mapTensorX[x][y] = element

                try:
                    mapTensorY[y][x] = element
                except KeyError:
                    mapTensorY[y] = { x: element }

        # Normalize the mapping:
        def normalize(tensor, p = False):
            lSqrt = math.sqrt
            for vectors in tensor.values():
                sumProb = 0.0
                for element in vectors.values():
                    amplitude = element.amplitude
                    sumProb += (amplitude * amplitude.conjugate()).real

                normalized = lSqrt(sumProb)
                for element in vectors.values():
                    element.amplitude = element.amplitude / normalized

        normalize(mapTensorX)
        normalize(mapTensorY, True)

        # Entangle the registers
        for x, yStates in mapTensorX.items():
            for y, element in yStates.items():
                amplitude = element.amplitude
                toState = toRegister.states[y]
                fromState = self.states[x]
                toState.entangle(fromState, amplitude)
                fromState.entangle(toState, amplitude.conjugate())

        if propagate:
            toRegister.propagate(self)

    def measure(self):
        measure = random.random()
        sumProb = 0.0

        # Pick a state
        finalX = None
        finalState = None
        for x, state in enumerate(self.states):
            amplitude = state.amplitude
            sumProb += (amplitude * amplitude.conjugate()).real

            if sumProb > measure:
                finalState = state
                finalX = x
                break

        # If state was found, update the system
        if finalState is not None:
            for state in self.states:
                state.amplitude = complex(0.0)

            finalState.amplitude = complex(1.0)
            self.propagate()

        return finalX

    def entangles(self, register = None):
        entangles = 0
        for state in self.states:
            entangles += state.entangles(None)

        return entangles

    def amplitudes(self):
        amplitudes = []
        for state in self.states:
            amplitudes.append(state.amplitude)

        return amplitudes
        
def printEntangles(register):
    printInfo("Entagles: " + str(register.entangles()))

def printAmplitudes(register):
    amplitudes = register.amplitudes()
    for x, amplitude in enumerate(amplitudes):
        printInfo('State #' + str(x) + '\'s amplitude: ' + str(amplitude))
