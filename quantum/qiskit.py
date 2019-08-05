from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit import BasicAer as Aer
from qiskit.tools.visualization import circuit_drawer, plot_histogram
from foresttools import plot_histogram

"""
Noisy quantum computers.
"""

q = QuantumRegister(2)
c = ClassicalRegister(2)
circuit = QuantumCircuit(q, c)
circuit.h(q[0])
circuit.cx(q[0], q[1])
circuit_drawer(circuit)
circuit.measure(q, c)
circuit_drawer(circuit)
backend = Aer.get_backend("qasm_simulator")
job = execute(circuit, backend, shots=100)
plot_histogram(job.result().get_counts(circuit))
