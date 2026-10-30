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
for i in range(1, tot_terms-1):
    #X matrices
    kron_list_X = [0] * N
    kron_list_X = [I_matrix if x == 0 else x for x in kron_list_X]
    kron_list_X[i-1] = X_matrix
    kron_list_X[i] = X_matrix

    # Applying kronecker delta
    H_partly_X = reduce(np.kron, kron_list_X)
    H_total.append(H_partly_X)

    # Y matrices
    kron_list_Y = [0] * N
    kron_list_Y = [I_matrix if x == 0 else x for x in kron_list_Y]
    kron_list_Y[i - 1] = Y_matrix
    kron_list_Y[i] = Y_matrix

    # Applying kronecker delta
    H_partly_Y = reduce(np.kron, kron_list_X)
    H_total.append(H_partly_Y)


full_hamiltonian = np.sum(H_total, axis=0)

print(full_hamiltonian)



