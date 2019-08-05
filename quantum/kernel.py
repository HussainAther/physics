import matplotlib.pyplot as plt
import numpy as np

from pyquil import Program, get_qc
from pyquil gates import CCNOT, CNOT, H, MEASURE, RZ, X
from foresttools import init_qvm_and_quilc

"""
Kernel method for spatial mapping.
"""

qvm_server, quilc_server, fc = init_qvm_and_quilc("")
qc == get_qc("4q-qvm", connection=fc)

# Data
training_set = [[0, 1], [0.78861006, 0.61489363]]
labels = [0, 1]
test_set = [[-0.549, 0.836], [0.053 , 0.999]]

def get_angle(amplitude_0):
    """
    Solve quantum equation to get angle.
    """
    return 2*np.arccos(amplitude_0)

test_angles = [get_angle(test_set[0][0])/2, get_angle(test_set[1][0])/2]
training_angle = get_angle(training_set[1][0])/4

def prepare_state(angles):
    ancilla_qubit = 0
    index_qubit = 1
    data_qubit = 2
    class_qubit = 3
    circuit = Program()
    # Put the ancilla and the index qubits into uniform superposition
    circuit += H(ancilla_qubit)
    circuit += H(index_qubit)
    # Prepare the test vector
    circuit += CNOT(ancilla_qubit, data_qubit)
    circuit += RZ(-angles[0], data_qubit)
    circuit += CNOT(ancilla_qubit, data_qubit)
    circuit += RZ(angles[0], data_qubit)
    # Flip the ancilla qubit > this moves the input 
    # vector to the |0> state of the ancilla
    circuit += X(ancilla_qubit)
    # Prepare the first training vector
    # [0,1] -> class 0
    # We can prepare this with a Toffoli
    circuit += CCNOT(ancilla_qubit, index_qubit, data_qubit)
    # Flip the index qubit > moves the first training vector to the 
    # |0> state of the index qubit
    circuit += X(index_qubit)
    # Prepare the second training vector
    # [0.78861, 0.61489] -> class 1
    circuit += CCNOT(ancilla_qubit, index_qubit, data_qubit)
    circuit += CNOT(index_qubit, data_qubit)
    circuit += RZ(angles[1], data_qubit)
    circuit += CNOT(index_qubit, data_qubit)
    circuit += RZ(-angles[1], data_qubit)
    circuit += CCNOT(ancilla_qubit, index_qubit, data_qubit)
    circuit += CNOT(index_qubit, data_qubit)
    circuit += RZ(-angles[1], data_qubit)
    circuit += CNOT(index_qubit, data_qubit)
    circuit += RZ(angles[1], data_qubit)
    # Flip the class label for training vector #2
    circuit += CNOT(index_qubit, class_qubit)
    return circuit

# from qiskit.tools.visualization import circuit_drawer
angles = [test_angles[0], training_angle]
state_preparation_0 = prepare_state(angles)
plot_circuit(state_preparation_0)

# Natural kernel on a shallow circuit 
def interfere_data_and_test_instances(circuit, angles):
    ro = circuit.declare(name="ro", memory_type="BIT", memory_size=4)
    circuit += H(0)
    for q in range(4):
        circuit += MEASURE(q, ro[q])
    return circuit

x = np.linspace(-2, 2, 100)
plt.xlim(-2, 2)
plt.ylim(0, 1.1)
plt.plot(x, 1-x**2/4)

def postselect(result_counts):
    total_samples = sum(result_counts.values())
    # define lambda function that retrieves only results where the ancilla is in the |0> state
    post_select = lambda counts: [(state, occurences) for state, occurences in counts.items() if state[-1] == '0']
    # perform the postselection
    postselection = dict(post_select(result_counts))
    postselected_samples = sum(postselection.values())
    print(f"Ancilla post-selection probability was found to be {postselected_samples/total_samples}")
    retrieve_class = lambda binary_class: [occurences for state, occurences in postselection.items() if state[0] == str(binary_class)]
    prob_class0 = sum(retrieve_class(0))/postselected_samples
    prob_class1 = sum(retrieve_class(1))/postselected_samples
    print("Probability for class 0 is", prob_class0)
    print("Probability for class 1 is", prob_class1)

circuit_0 = interfere_data_and_test_instances(state_preparation_0, angles)
circuit_0.wrap_in_numshots_loop(1024)
executable = qc.compile(circuit_0)
measures = qc.run(executable)
count = np.unique(measures, return_counts=True, axis=0)
count = dict(zip(list(map(lambda l: "".join(list(map(str, l))), count[0].tolist())), count[1]))
print(count)
postselect(count)

angles = [test_angles[1], training_angle]
state_preparation_1 = prepare_state(angles)

circuit_1 = interfere_data_and_test_instances(state_preparation_1, angles)
executable = qc.compile(circuit_1)
measures = qc.run(executable)

count = np.unique(measures, return_counts=True, axis=0)
count = dict(zip(list(map(lambda l: ''.join(list(map(str, l))), count[0].tolist())), count[1]))
print(count)

postselect(count)

qvm_server.termninate()
quilc_server.terminate()
