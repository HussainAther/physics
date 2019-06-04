import numpy as np
import argparse

from random import random

"""
Shor's algorithm for quantum integer factorization.
"""

class Mapping:
    """
    Initialize a state and amplitude to map inputs to outputs. 
    """
    def __init__(self, state, amplitude):
        """
        The state and amplitude let us identify nodes or units on the various
        mappings between one another.
        """
        self.state = state
        self.amplitude = amplitude

class QuantumState:
    """
    Identify a quantum state with the corresponding entanglements.
    """
    def __init__(self, amplitude, register):
        """
        Quantum states should account for this entanglement we observe.
        """
        self.amplitude = amplitude
        self.register = register
        self.entangled = {}

    def entangle(self, fromState, amplitude):
        """
        Entangle with the state the particle is in with a corresponding
        amplitude. This function keeps track of the entanglements.
        """
        register = fromState.register
        entanglement = Mapping(fromState, amplitude)
        try:
            self.entangled[register].append(entanglement)
        except KeyError:
            self.entangled[register] = [entanglement]

    def entangles(self, register = None):
        """
        Perform the entanglement process while checking with the appropriate
        registers. 
        """
        entangles = 0
        if register is None:
            for states in self.entangled.values():
                entangles += len(states)
        else:
            entangles = len(self.entangled[register])
        return entangles

class QubitRegister:
    """
    Register input and output.  
    """
    def __init__(self, numBits):
        """
        Initialize hte variables we need.
        """
        self.numBits = numBits
        self.numStates = 1 << numBit # bitwise operator for easy access
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
    
    def map(self, toRegister, mapping, propagate = True):
        """
        Map the tensor using the appropriate mapping. This converts any mapping to a unitary
        tensor given each element v returned by the mapping has the property v * v.conjugate() = 1.
        """
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

        def normalize(tensor, p = False):
            """
            Normalize the tensor mapping with the sum of the probabilities.
            """
            for vectors in tensor.values():
                sumProb = 0.0
                for element in vectors.values():
                    amplitude = element.amplitude
                    sumProb += (amplitude * amplitude.conjugate()).real

                normalized = np.sqrt(sumProb)
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
        measure = random()
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

def hadamard(x, Q):
    codomain = []
    for y in range(Q):
        amplitude = complex(pow(-1.0, bitCount(x & y) & 1))
        codomain.append(Mapping(y, amplitude))

    return  codomain

def qModExp(a, exp, mod):
    """
    Quantum Modular Exponentiation (quantum modular exponentiation)
    """
    state = modExp(a, exp, mod)
    amplitude = complex(1.0)
    return [Mapping(state, amplitude)]

def qft(x, Q):
    """
    Quantum Fourier transform (QFT quantum Transform)
    """
    fQ = float(Q)
    k = -2.0 * np.pi
    codomain = []

    for y in range(Q):
        theta = (k * float((x * y) % Q)) / fQ
        amplitude = complex(np.cos(theta), np.sin(theta))
        codomain.append(Mapping(y, amplitude))

    return codomain

def findPeriod(a, N):
    nNumBits = N.bit_length()
    inputNumBits = (2 * nNumBits) - 1
    inputNumBits += 1 if ((1 << inputNumBits) < (N * N)) else 0
    Q = 1 << inputNumBits
    printInfo("Finding the period...")
    printInfo("Q = " + str(Q) + "\ta = " + str(a))
    inputRegister = QubitRegister(inputNumBits)
    hmdInputRegister = QubitRegister(inputNumBits)
    qftInputRegister = QubitRegister(inputNumBits)
    outputRegister = QubitRegister(inputNumBits)
    printInfo("Registers generated")
    printInfo("Performing Hadamard on input register")
    inputRegister.map(hmdInputRegister, lambda x: hadamard(x, Q), False)
    # inputRegister.hadamard(False)
    printInfo("Hadamard complete")
    printInfo("Mapping input register to output register, where f(x) is a^x mod N")
    hmdInputRegister.map(outputRegister, lambda x: qModExp(a, x, N), False)
    printInfo("Modular exponentiation complete")
    printInfo("Performing quantum Fourier transform on output register")
    hmdInputRegister.map(qftInputRegister, lambda x: qft(x, Q), False)
    inputRegister.propagate()
    printInfo("Quantum Fourier transform complete")
    printInfo("Performing a measurement on the output register")
    y = outputRegister.measure()
    printInfo("Output register measured\ty = " + str(y))
    # Interesting to watch - simply uncomment
    # printAmplitudes(inputRegister)
    # printAmplitudes(qftInputRegister)
    # printAmplitudes(outputRegister)
    # printEntangles(inputRegister)
    printInfo("Performing a measurement on the periodicity register")
    x = qftInputRegister.measure()
    printInfo("QFT register measured\tx = " + str(x))
    if x is None:
        return None
    printInfo("Finding the period via continued fractions")
    r = cf(x, Q, N)
    printInfo("Candidate period\tr = " + str(r))
    return r

"""
Classical components
"""

BIT_LIMIT = 12

def bitCount(x):
    """
    Convert to bits. 
    """
    sumBits = 0
    while x > 0:
        sumBits += x & 1
        x >>= 1

    return sumBits

def gcd(a, b):
    """
    Return greatest common divisor.
    """
    while b != 0:
        tA = a % b
        a = b
        b = tA

    return a

def extendedGCD(a, b):
    """
    Extended Euclidean algorithm to get the polynomial greatest common divisor
    and the coefficients of Bézout's identity of two univariate polynomials.
    """
    fractions = []
    while b != 0:
        fractions.append(a // b)
        tA = a % b
        a = b
        b = tA

    return fractions

def cf(y, Q, N):
    """
    Continued fractions we get through iterating processes using sums and 
    reciprocals of another number.
    """
    fractions = extendedGCD(y, Q)
    depth = 2

    def partial(fractions, depth):
        c = 0
        r = 1

        for i in reversed(range(depth)):
            tR = fractions[i] * r + c
            c = r
            r = tR

        return c

    r = 0
    for d in range(depth, len(fractions) + 1):
        tR = partial(fractions, d)
        if tR == r or tR >= N:
            return r

        r = tR

    return r

def modExp(a, exp, mod):
    """
    Modular exponentiation used to exponentiate over a modulus.
    """
    fx = 1
    while exp > 0:
        if (exp & 1) == 1:
            fx = fx * a % mod
        a = (a * a) % mod
        exp = exp >> 1

    return fx

def pick(N):
    """
    Pick a number at random.
    """
    a = np.floor((random() * (N - 1)) + 0.5)
    return a

def checkCandidates(a, r, N, neighborhood):
    if r is None:
        return None

    # Check multiples
    for k in range(1, neighborhood + 2):
        tR = k * r
        if modExp(a, a, N) == modExp(a, a + tR, N):
            return tR

    # Check lower neighborhood
    for tR in range(r - neighborhood, r):
        if modExp(a, a, N) == modExp(a, a + tR, N):
            return tR

    # Check upper neigborhood
    for tR in range(r + 1, r + neighborhood + 1):
        if modExp(a, a, N) == modExp(a, a + tR, N):
            return tR

    return None

def shors(N, attempts = 1, neighborhood = 0.0, numPeriods = 1):
    """
    Carry out the steps of the algorithm.
    """
    if(N.bit_length() > BIT_LIMIT or N < 3):
        return False
    periods = []
    neighborhood = np.floor(N * neighborhood) + 1
    printInfo("N = " + str(N))
    printInfo("Neighborhood = " + str(neighborhood))
    printInfo("Number of periods = " + str(numPeriods))
    for attempt in range(attempts):
        printInfo("\nAttempt #" + str(attempt))
        a = pick(N)
        while a < 2:
            a = pick(N)
        d = gcd(a, N)
        if d > 1:
            printInfo("Found factors classically, re-attempt")
            continue
        r = findPeriod(a, N)
        printInfo("Checking candidate period, nearby values, and multiples")
        r = checkCandidates(a, r, N, neighborhood)
        if r is None:
            printInfo("Period was not found, re-attempt")
            continue
        if (r % 2) > 0:
            printInfo("Period was odd, re-attempt")
            continue
        d = modExp(a, (r // 2), N)
        if r == 0 or d == (N - 1):
            printInfo("Period was trivial, re-attempt")
            continue
        printInfo("Period found\tr = " + str(r))
        periods.append(r)
        if(len(periods) < numPeriods):
            continue
        printInfo("\nFinding least common multiple of all periods")
        r = 1
        for period in periods:
            d = gcd(period, r)
            r = (r * period) // d
        b = modExp(a, (r // 2), N)
        f1 = gcd(N, b + 1)
        f2 = gcd(N, b - 1)
        return [f1, f2]
    return None

"""
Command-line functions.
"""

def parseArgs():
    parser = argparse.ArgumentParser(description="Simulate Shor\'s algorithm for N.")
    parser.add_argument("-a", "--attempts", type=int, default=20, help="Number of quantum attempts to perform")
    parser.add_argument("-n", "--neighborhood", type=float, default=0.01, help="Neighborhood size for checking candidates (as percentage of N)")
    parser.add_argument("-p", "--periods", type=int, default=2, help="Number of periods to get before determining least common multiple")
    parser.add_argument("-v", "--verbose", type=bool, default=True, help="Verbose")
    parser.add_argument("N", type=int, help="The integer to factor")
    return parser.parse_args()

def main():
    args = parseArgs()

    global printInfo
    if args.verbose:
        printInfo = printVerbose
    else:
        printInfo = printNone

    factors = shors(args.N, args.attempts, args.neighborhood, args.periods)
    if factors is not None:
        print("Factors:\t" + str(factors[0]) + ", " + str(factors[1]))

if __name__ == "__main__":
    main()
