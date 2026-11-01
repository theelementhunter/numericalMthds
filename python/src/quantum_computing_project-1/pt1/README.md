# Stage 1: Exact Diagonalization (NumPy-First Quantum Simulation)

This repository contains `fermion_to_pauli.py`, the first step of a project inteded to teach me a deeper understanding of quantum simulations. It performs classical exact diagonalization to simulate propagation along a 1D chain.

Rather than relying on abstract quantum SDKs, this script explicitly constructs the Hamiltonian and state vectors from the ground up using NumPy and Kronecker products. 

## Requirements

Ensure you have the following Python packages installed:
```bash
pip install numpy scipy
```

## Physics Overview

The script models a single exciton hopping between adjacent sites on a 1D chain of $N$ sites.
* **Fermion-to-Spin Mapping:** Exciton operators are mapped to Pauli spin operators. For a 1D chain with nearest-neighbor hopping, the Jordan-Wigner $Z$-strings cancel out, leaving the local XY model: $H_{\text{hop}} = \frac{J}{2} \sum (X_i X_{i+1} + Y_i Y_{i+1})$.
* **The Kronecker Product:** Physical interactions are local (2-body). To maintain the mathematical dimensions of the full state space ($2^N$), non-interacting sites are padded with the $2 \times 2$ Identity matrix ($I$) via the Kronecker product.

## Script Breakdown

1. **Hamiltonian Construction:** Iterates over every adjacent bond, scales the local $X \otimes X$ and $Y \otimes Y$ interactions into global $2^N \times 2^N$ matrices using `functools.reduce(np.kron)`, and sums them.
2. **State Initialization:** Defines physical sites in the computational basis (Empty: `[1, 0]`, Occupied: `[0, 1]`). Builds the global initial state vector $|\psi(0)\rangle$ using Kronecker products based on the `init_list`.
3. **Time Evolution:** Uses the continuous-time Schrödinger equation $|\psi(t)\rangle = e^{-iHt}|\psi(0)\rangle$. The exact time evolution operator is computed via `scipy.linalg.expm`.

## Usage

Run the script directly from your terminal:
```bash
python fermion_to_pauli.py
```

### Configuration Variables
You can easily modify the physics by changing these variables inside the script:
* `N`: Total number of sites in the 1D chain. *(Note: Computational memory scales exponentially as $2^N$)*.
* `t`: The duration of continuous time evolution.
* `J`: The transfer integral (hopping amplitude).
* `init_list`: A list of 1-indexed physical sites to place initial excitons on (e.g., `[1]` for an excitation on site 1, or `[1, 3]` for sites 1 and 3).

## Expected Output
The script will print the absolute probabilities of the system being in any of the $2^N$ possible states at time $t$.