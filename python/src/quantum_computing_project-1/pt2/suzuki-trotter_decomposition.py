import numpy as np
from functools import reduce
from scipy.linalg import expm

# Identity Matrix
I_matrix = np.array([[1, 0],
                    [0, 1]],
                    dtype=complex)
# Pauli Matrices
# X ~ bit-flip matrix
X_matrix = np.array([[0, 1],
                    [1, 0]],
                    dtype=complex)
# Z ~ phase-flip matrix
Z_matrix = np.array([[1, 0],
                    [0, -1]],
                    dtype=complex)
# Y ~ bit-phase-flip matrix
Y_matrix = np.array([[0 + 0j, 0 - 1j],
                     [0 + 1j, 0 + 0j]],
                    dtype=complex)

# Transfer Integral Term
J = 1.0

# Number of sites
N = 3

# Number of hops
num_hops = N - 1

#  Total terms
tot_terms = 2 * num_hops

# Number of Identity terms
num_ident = N - 2

#Automatic stop for N <= 1 (bc that makes no physical sense)
if N <= 1:
    raise ValueError("Error: Number of sites cannot be less than 2.")

H_total = []
for i in range(num_hops):
    #X matrices
    kron_list_X = [I_matrix] * N
    kron_list_X[i] = X_matrix
    kron_list_X[i+1] = X_matrix
    H_partly_X = reduce(np.kron, kron_list_X)

    # Y matrices
    kron_list_Y = [I_matrix] * N
    kron_list_Y[i] = Y_matrix
    kron_list_Y[i + 1] = Y_matrix
    H_partly_Y = reduce(np.kron, kron_list_Y)

    # Applying kronecker delta
    H_full = (J / 2) * (H_partly_X + H_partly_Y)
    H_total.append(H_full)

# Initializing excitation on site 1

# Occupied site
occupied = np.array([0,1])

# Empty site
empty = np.array([1,0])

#Initializaiton List
init_list = [1]

# Safeguard against non-physical shenanigans
if len(init_list) < 1 or len(init_list) > N or max(init_list) > N:
    print("Invalid list. Initializing on site 1.")
    init_list = [1]

kron_list_init = [empty] * N

# Populating the correct states with excitations
for i in init_list:
    kron_list_init[i-1] = occupied

# Another Kronecker product!
init_array = reduce(np.kron, kron_list_init)

# This is where the paths diverge from pt1
# Quantum computers cannot add all of the matrices together; we are limited to
# 1- and 2-qubit operations by quantum computing hardware
# full_hamiltonian = (J / 2) * np.sum(H_total, axis=0) ~ N/A
# Instead we rely on trotterization

#
t = 1.0
dt = 0.001
time_steps = int(t/dt)
H_total_exp = [expm(-1j * h * dt) for h in H_total]

for i in range(time_steps):
    init_array = reduce(lambda vec, mat: mat @ vec, H_total_exp, init_array)

probabilities = np.abs(init_array)**2
print(f"\nProbabilities of all {2**N} states at t={t}:")
print(np.round(probabilities, 3))



