import numpy as np

from pyquil import Program, get_qc
from pyquil.api import WavefunctionSimulator
from pyquil.gates import CNOT, H, MEASURE, circuit
from foresttools import init_qvm_and_quilc, plot_histogram

"""
Gate-model in quantum computing.
"""

np.set_printoptions(precision=3, suppress=True)
qvm_server, quilc_server, fc = init_qvm_and_quilc("/")
qc = get_qc("2q-qvm", connection=fc)

wf_sim = WavefunctionSimulator(connection=fc)
circuit = Program()
circuit += H(0)
circuit += CNOT(0, 1)
plot_circuit(circuit)
ro = circuit.declare("ro", "BIT", 2)
circuit += MEASURE(0, ro[0])
circuit += MEASURE(1, ro[1])
circuit.wrap_in_numshots_loop(100)
executable = qc.compile(circuit)
result = qc.run(executable)
plot_histogram(result)
