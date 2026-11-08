# Stage 2: Suzuki-Trotter Decomposition (Hardware-Constrained Simulation)

This repository contains `suzuki-trotter_decomposition.py`, the second stage of a quantum simulation framework. This script bridges the gap between classical exact diagonalization (Stage 1) and the physical constraints of quantum hardware.

## The Problem: Non-Commutativity and Hardware Limits
In Stage 1, we summed all physical interactions into a single global Hamiltonian matrix and exponentiated it. Quantum computers cannot do this. They are physically limited to firing localized microwave pulses at 1 or 2 qubits at a time. 

Mathematically, this forces us to exponentiate the localized bonds *separately*. However, because adjacent bonds share a site (e.g., Bond 1-2 and Bond 2-3 both act on site 2), their matrices do not commute ($[H_A, H_B] \neq 0$). Exponentiating them separately introduces a mathematical error.

## The Solution: Trotterization
We use the Suzuki-Trotter approximation to simulate the time evolution. By slicing the total evolution time $t$ into microscopically small steps $\Delta t$, we can sequentially multiply the localized bond matrices. As $\Delta t \to 0$, the commutation error vanishes, and the sequential pulses perfectly mimic simultaneous global evolution.

## Script Breakdown
1. **Isolated Bond Construction:** The script builds the $X \otimes X$ and $Y \otimes Y$ interactions just like Stage 1, but keeps each bond as a separate matrix in a list rather than summing them globally.
2. **Pre-computing Pulses:** Each individual bond matrix is exponentiated by $\Delta t$ to create a list of localized unitary operators (simulating 2-qubit microwave pulses).
3. **The Trotter Loop:** A `for` loop iterates through time steps. Inside the loop, `functools.reduce` sequentially applies each bond's unitary matrix to the state vector, mimicking the rapid-fire sequence of a quantum circuit.

## Usage
Run the script directly from your terminal:
```bash
python suzuki-trotter_decomposition_3.py
```

### Configuration Variables
* `t`: Total continuous time to simulate.
* `dt`: The time step. **(Try changing this from 0.01 to 0.001 to observe how the Trotter error vanishes and aligns perfectly with Stage 1!)**
* `J`: The transfer integral (hopping amplitude).
* `N`: Number of sites.