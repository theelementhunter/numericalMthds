# Core imports
from qiskit import QuantumCircuit # main class allowing us to define QCs
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator #simulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Runtime Imports
from qiskit_ibm_runtime import SamplerV2 as Sampler, QiskitRuntimeService

#matplotlib so I can see!
import matplotlib.pyplot as plt

# Instance of QuantumCircuit class w/ 2 qubits
bell = QuantumCircuit(2)

# Apply hadamar gate to the 1st qubit
# Bits qubit 1 into a superposition of 0 and 1
bell.h(0)

# Apply CNOT gate (
# qubit1 = control ; qubit2 = target
bell.cx(0, 1) # qubits are now entangled!

# Measure both qubits (stores results in a *classical* register
bell.measure_all()

# Draw the circuit
bell.draw("mpl")

# Helper Function
# Helps run QC and retrieves measurement results
def run_circuit_and_get_counts(circuit, backend, shots=1000):
    """
    Runs QC on specified backend and rturns measurment counts.
    :param circuit:  QC to run
    :param backend: Qiskit backend (real device or simulator)
    :param shots: The number of shots ot run the circuit
    :return:
        dict: A dictionalary of measurement counts
    """

    # Optimization
    # Rewrites circuit into a form the hardware can understand + efficienty
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(circuit)

    # Sampler ~ tools used to execute circuits and collect measurement data
    sampler = Sampler(mode=backend)

    # Runs circuit multiple times and retrieves the results to extract measurment counts
    job = sampler.run([isa_circuit], shots=shots)
    result = job.result()

    return result[0].data.meas.get_counts()


# Run job in cloud
QiskitRuntimeService.save_account(channel='ibm_quantum_platform',
                                  token='xDtXqkVOE_A2AkM1WS6Rv5EsCd70Q4nARLdCMwHUskOr',
                                  overwrite=True,
                                  set_as_default=True)

service = QiskitRuntimeService()

# least busy
backend = service.least_busy(operational=True, simulator=False,min_num_qubits=127)
print(backend.name)

# Pass into helper function and plot histogram
counts = run_circuit_and_get_counts(bell, backend, shots=1000)
plot_histogram(counts)

# show me the money!
plt.show()
