import numpy as np
import random
import pandas as pd

#NOTES
# define an interval (xmin, xmax)
# evenly divide it into N partitions
#enter psi[0] and psi[1]
#use these to integrate through the rest of the line
#count number of nodes (-E -> +E)
##
"""
Energy Guess ~ Epsilon
Get f_n for each point on the line
Calculate the Wavefunctions
Check if number of nodes, s,  in the current calculated wfn is equal to the number of nodes, n
    If s > (<) n ~ energy is too high take lower (higher) half of energy interval
Repeat until correct number of nodes is reached
Print output file (x, psi psi^2, )
"""

#inputs:
#x_max for integration
#N ~ number of grid points (creates array of x values and Psi array for solutions)
#output filename
#n ~ number of nodes (nodes correspond to which energy we are trying ot find)

#n-even
# y_0 ~ arbitrary / finite
# y_1 ~ determined by Numerov's formula

#n- odd
# y_0 ~ zero
# y_1 ~ arbitrary / finite

## --- USER INPUTS --- ##
N = 1000 #number of grid points
n = 3 #number of nodes #nth excited state #CODE STOPS IF n is negative
x_max = 3
epsilon = 26 #random energy guess, that is modified using the 'shooting method'
## --- END --- ##

## --- INTEGRATOR FUNCTION #eqn 1.33 --- ##
def numerov_integrator(psi: np.ndarray, f_n: np.ndarray, N: int):

    for i in range(2, N): #first two psi's are initial conditions and are already "solved"
        num1 = ( 12 - ( 10 * f_n[i-1] ) ) * psi[i-1]
        num2 = f_n[i-2] * psi[i-2]
        numerator = num1 - num2
        denominator = f_n[i]
        psi[i] = numerator / denominator

    return psi

## --- CHECK NUMBER OF NODES FUNCTION --- ##
##takes in wavefunction
##returns number of nodes (this is compared to n)
def number_of_nodes(psi: np.ndarray):
    node_count = 0
    for i in range(psi.shape[0] - 1):
        if (psi[i] > 0 and psi[i+1]) < 0 or (psi[i] < 0 and psi[i+1] > 0):
            node_count += 1
        else:
            continue

    return node_count

#using the power of numpy arrays, we get all the g_n and f_n
#eqn 1.32
def f_n(epsilon: float, x: float, dx: float):
    g_n = (2 * epsilon) - (x** 2)
    f_n = 1 + (g_n * ((dx ** 2) / 12))
    return f_n

## --- PSI --- ##
#create empty array to store wfn solutions
psi = np.zeros(N, dtype=float)

## --- SPACE DISCRETIZATION & SPACING --- ##
#discretize a line segment with evenly spaced intervals
#dx ~ step size ('h' in the numerov formula
x, dx = np.linspace(0, x_max, N, retstep=True) # retstep gives us the stepsize

## --- INITIAL ENERGY RANGE --- ##
#used to try and find the eigenvalue E_n or epsilon_n
epsilon_range = np.array((0.0,1000.0))

## --- INITIAL CONDITIONS --- ##
#different depending on whether n is even or odd

f_n_initial = f_n(epsilon, x, dx)

if n % 2 == 0: #n ~ even
    psi[0] = 1 #random.randint(1, 10) #arbitrary finite number
    psi[1] = ( ( 12 - ( 10 * f_n_initial[0] ) ) * psi[0] ) / 2 * f_n_initial[1] #eqn 1.34
else: #n ~ odd
    psi[0] = 0
    psi[1] = 1 #arbitrary finite number

#SHOOTING ALGORITHM
#Solve wavefunciton for every point in space, filling the psi np.array
#check number of nodes
#   -if num_of_n > n -> Run again
#   -f num_of_n < n -> Run again
#   -if num_of_n == n -> print out x, psi, and |psi|^2 in csv

solution_reached = False
iteration = 0
while solution_reached == False:
    print(f"epsilon {epsilon}")
    ## --- F_N --- ##
    # using the power of numpy arrays, we get all the g_n and f_n
    # eqn 1.32
    f_n_updated = f_n(epsilon, x, dx)  # np.array

    ## --- PSI_WAVEFUNCTION --- ##
    psi_updated = numerov_integrator(psi, f_n_updated, N)

    ## --- COUNT NODES IN PSI --- ##
    n_count = number_of_nodes(psi_updated)

    if n_count == n:
        # --- SOLUTION FOUND --- #
        print("Epsilon just right. Solution has been found.")
        solution_reached = True
    else:
        # --- KEEP SEARCHING --- #
        if n_count > n:
            epsilon_range[1] = epsilon
            print("Epsilon is too high. Lowering it...")
        elif n_count < n:
            epsilon_range[0] = epsilon
            print("Epsilon is too low. Raising it...")

        # --- UPDATE EPSILON --- #
        epsilon = (epsilon_range[0] + epsilon_range[1]) / 2
        iteration += 1

##
print(f"Solution took {iteration} iterations.")

## --- PSI SQUARED --- ##
psi_squared = psi_updated * psi_updated

solution_df = pd.DataFrame({
    "x": x,
    "psi": psi,
    "psi_squared": psi_squared
})

## --- save solutions --- ##
solution_df.to_csv("harmonic0.csv", index=False)

print(f"\nData saved!")

