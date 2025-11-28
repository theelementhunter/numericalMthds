#Two Integrations
#   1. forward recursion, starting from x = 0
#   2. backward one, starting from x_max

#Differences from harmonic0
#   - Integration
#       - It no longer is a simple "outward shooting". We go from 0 -> x_cl and x_cl <- xmax
#         Giving us a left and right wavefunction. This ensures that we aren't integrating into
#         the region where our solution diverges, allowing bits of the "bad solution" to blow it
#         up.
#   - Solution check
#       - Now we check our solution, not by counting the nodes, but by checking if there is continuity
#         between the left and right solutions of the wavefunction. If the energy guess is bad we will see
#         a discontinuity where they meet. Psi_cl(right) - Psi_cl(left) will tell us this information.
#
# x_cl ~ where v(x_cl) = E ; depends on the trial energy
#      ~ accuracy limited by the intervals between grid points
# Because we are on a discrete line grid, no interval will (probably) be satisfy v(x_cl) = E exactly
# This is where the accuracy is only as high as the intervals between the gridpoint
# the index (icl) for the first grid point x_c = icl*delta_x such that V(x_c) > E
# classical inversion point will be located somewhere between x_c - delta_x and x_c
#
# Outward integration is performed until grid point icl giving a psi_L defined in [0, x_c]
#
#For the 2 integrations must we have  separate psi_L and psi_R, or is this distinction made for clarity's sake?
#
#Okay, point of order, let's think this out
# - Find Classical Inversion Point ~ V(x_cl) = E
#   - Choose Trial Energy E
#   - V_n = zeta^2 / 2
#       - Ex:
#           epsilon = 5
#           zeta needs to be > sqrt(10)
#       - For us specifically zeta will be between x_c - delta_x and x_c
#         where x_c = (icl)*delta_x
#         where icl ~ index where V(x_c) > E
#   - V_1 > E -> if yes, then icl = 1, if no go to the next one
#   - Once icl is found, we'll call the integrator (we'll have to
#     change it so that we can specify the indexes for integration
#     since we have to do forward and backward integration)


import numpy as np
import pandas as pd

## --- INITIAL CONDITION SETUP --- ###
def set_initial_conditions(psi: np.ndarray, number_of_nodes: int,
                           f_n_initial: np.ndarray, delta_x: float, xgrid: np.ndarray):
    #psi_L initial conditions
    if number_of_nodes % 2 == 0:  # n ~ even
        psi[0] = 1  # can be any arbitrary finite number, we choose 1 for simplicity
        psi[1] = ((12 - (10 * f_n_initial[0])) * psi[0]) / (2 * f_n_initial[1])  # eqn 1.34
    else:  # n ~ odd
        psi[0] = 0
        psi[1] = 1  # can be any arbitrary finite number, we choose 1 for simplicity

    #psi_R initial conditions
    decay_ratio = np.exp( ( (xgrid[-1] ** 2) - (xgrid[-2] ** 2) ) / 2.0 )
    psi[-1] = delta_x
    psi[-2] = psi[-1] * decay_ratio

    return psi

### --- FIND icl POINT --- ###
def find_icl(epsilon: float, xgrid: np.ndarray):
    for i, x in enumerate(xgrid):

        #Calculate potential
        potential = (x ** 2) / 2.0

        #Check if we've hit the forbidden region
        if potential > epsilon:
            return i # return the index at which we hit the forbidden region
    #If we do not ever hit the forbidden region we return the last index in the array
    return len(xgrid) - 1


## --- INTEGRATOR FUNCTION MODIFIED #eqn 1.33 --- ##
def numerov_integrator(psi: np.ndarray, f_n: np.ndarray, N: int,
                       direction: str, icl: int):


    # Checking integration direction
    f_or_b = direction[0].lower() #character is lowercase

    # Set up parameters for forward and backward integration
    if f_or_b == "f":
        # Forward: Look "behind" (-1), iterate upward
        k = -1
        # Start at 2 (ICs set 0 & 1)
        loop_range = range(2, icl + 1)
    elif f_or_b == "b":
        # Backward: Look "ahead" (+1), iterate downward
        k = +1
        loop_range = range(N-3, icl -1, -1)
    else:
        print("Unknown direction. Pick between 'forward' or 'backward.'")

    #integrator loop
    for i in loop_range:
        prev_1 = i + k
        prev_2 = i + (2*k)

        var1 = (12 - (10 * f_n[prev_1])) * psi[prev_1]
        var2 = f_n[prev_2] * psi[prev_2]

        numerator = var1 - var2
        denominator = f_n[i]

        psi[i] = numerator / denominator

    return psi

#using the power of numpy arrays, we get all the g_n and f_n #eqn 1.32
#separated g_n and f_n into separate function so that I could use g_n to find icl
def g_n(epsilon: float, x: np.ndarray):
    g_n = (2 * epsilon) - (x ** 2)
    return g_n

def f_n(g_n: np.ndarray, dx: float):
    f_n = 1 + (g_n * ((dx ** 2) / 12))
    return f_n

### --- ALT WAY TO FIND icl --- ###
def find_icl_from_g_n(g_n: np.ndarray):
    for i, val in enumerate(g_n):
        if val < 0:
            return i
    return len(g_n) - 1

## --- CHECK NUMBER OF NODES FUNCTION --- ##
##takes in wavefunction
##returns number of nodes (this is compared to n)
def find_number_of_nodes(psi: np.ndarray, icl: int):
    node_count = 0
    for i in range(icl):
        if (psi[i] > 0 and psi[i+1]) < 0 or (psi[i] < 0 and psi[i+1] > 0):
            node_count += 1
        else:
            continue

    return node_count

### --- SCALING FUNCITON --- ###
def discontinuity_check(psi_L: np.ndarray, psi_R: np.ndarray, icl: int, delta_x: float, f_n: np.ndarray):

    #infinite discontinuity case (division by zero)
    if abs(psi_R[0]) < 1e-15:
        raise ValueError("Cannot divide by zero. Returning 'infinity' (1e9).")
        return 1e9

    #rescale psi_R
    scaling_factor = psi_L[-1] / psi_R[0]
    psi_R_scaled = scaling_factor * psi_R

    #check discontinuity #eqn 1.38
    num1 = psi_L[-2] + psi_R_scaled[1]
    num2 = ( 14 - (12 * f_n[icl]) ) * psi_L[-1]
    num = num1 - num2
    denom = delta_x

    difference = num / denom

    return difference, psi_R_scaled
#Solution time!
# There's two energy criteria. In the non-forbidden region we check for the correct number of nodes,
# and adjust the energy until they matched what we want. This is the forward integration
# Then we switch to the backward integration. This time the check is the discontinuity condition. We
# set an energy convergence threshold, and we keep adjusting the energy until the energies at the icl
# point are below this convergence threshold.
# We should make a function that takes in the two wavefunctions, the icl, and
# - Rescales the functions
# - Checks the discontinuity
# - returns the continuity value (psi'_R - psi'_L)
# Then, if the difference is less than or equal to the threshold value we've got the correct energy
# If the difference is negative the energy is too low it needs to be raised
# If  "      "       " positive  "    "    "   "  high "   "   "  "  lowered

### --- USER INPUTS --- ###
N = 1000 #number of grid points
target_number_of_nodes = 4 #number of nodes #nth excited state #CODE STOPS IF n is negative
x_max = 7
epsilon = 26 #random energy guess, that is modified using the 'shooting method'
convergence = 1e-10 #reasonable for the shooting method
### --- END --- ###

### --- SPACE DISCRETIZATION & SPACING --- ###
#discretize a line segment with evenly spaced intervals
#dx ~ step size ('h' in the numerov formula)
x, dx = np.linspace(0, x_max, N, retstep=True) # retstep gives us the stepsize

### --- INITIAL ENERGY RANGE --- ###
#used to try and find the eigenvalue E_n or epsilon_n
epsilon_range = np.array((0.0,1000.0)) # arbitrarily big range


#while energy loop #HOME STRETCH!
solution_reached = False
iteration = 0
while solution_reached == False:
    ### ---EPSILON GUESS --- ###
    epsilon_updated = (epsilon_range[0] + epsilon_range[1]) / 2

    ### --- UPDATER g_n and f_n --- ###
    g_n_updated = g_n(epsilon = epsilon_updated, x = x)
    f_n_updated = f_n(g_n =g_n_updated, dx = dx)

    ### --- UPDATE MATCHING POINT --- ###
    icl_updated = find_icl_from_g_n(g_n = g_n_updated)

    ### --- RE-INIIALIZATION (INITIAL CONDITIONS)--- ###
    #psi
    psi = np.zeros(N, dtype=float) # create empty array to store wfn solutions
    psi_initial = set_initial_conditions(psi=psi, number_of_nodes=target_number_of_nodes,
                                         f_n_initial=f_n_updated, delta_x=dx, xgrid=x)

    #print(f"psi-initial: {psi_initial}")
    ### --- FORWARD INTEGRATION --- ###
    psi_updated = numerov_integrator(psi = psi_initial, f_n = f_n_updated, N = N,
                                     icl = icl_updated, direction = "forward")

    ### --- FIRST ENERGY CHECK --- ###
    # We check to see if the energy we have gives us the correct number of nodes in the
    # classically allowed region
    number_of_nodes = find_number_of_nodes(psi = psi_updated, icl = icl_updated)

    if number_of_nodes != target_number_of_nodes:
        if number_of_nodes > target_number_of_nodes:
            epsilon_range[1] = epsilon_updated
        elif number_of_nodes < target_number_of_nodes:
            epsilon_range[0] = epsilon_updated
        iteration += 1
        continue

    print(f"Course Energy = {epsilon_updated} @ interaction = {iteration}.")
    #print(f"Psi: {psi_updated}")

    # if we've made it this far that means we've have the target number of nodes
    # it also means we can begin the backward integration and the fine energy adjustments

    ### --- EXTRACT psi_L --- ###
    # we have to do this before the backward integration, otherwise we will overwrite the icl
    # position
    psi_L = psi_updated[: icl_updated + 1].copy() #beginning -> icl

    ### --- BACKWARD INTEGRATION --- ###
    psi_updated = numerov_integrator(psi = psi_updated, f_n = f_n_updated, N = N,
                                     icl = icl_updated, direction = "backward")

    ### --- EXTRACT psi_R --- ###
    psi_R = psi_updated[icl_updated :] #icl -> end

    ### --- CHECK DISCONTINUITY --- ###
    diff = discontinuity_check(psi_L = psi_L, psi_R=psi_R, icl = icl_updated,
                               delta_x = dx, f_n = f_n_updated)

    print(f"Difference = {diff[0]}")

    if (epsilon_range[1] - epsilon_range[0]) > convergence:
        if diff[0] < 0:
            epsilon_range[0] = epsilon_updated
        else:
            epsilon_range[1] = epsilon_updated
        iteration += 1
        continue
    else:
        print(f"Fine Energy = {epsilon_updated} @ interaction = {iteration}.")
        #print(f"Psi: {psi_updated}")
        solution_reached = True

print(f"Solution took {iteration} iterations.")

print("\n--- Finalizing Solution ---")
#We generate clean psi_L and psi_R
g_final = g_n(epsilon = epsilon_updated, x = x)
f_final = f_n(g_n = g_final, dx = dx)
icl_final = find_icl_from_g_n(g_n = g_final)

### --- FORWARD --- ###
psi_final = np.zeros(N, dtype=float)
psi_initial = set_initial_conditions(psi=psi_final, number_of_nodes=target_number_of_nodes,
                                         f_n_initial=f_final, delta_x=dx, xgrid=x)

psi = numerov_integrator(psi = psi_initial, f_n = f_final, N = N,
                                     icl = icl_final, direction = "forward")
psi_L = psi[:icl_final +1].copy()

### --- BACKWARD --- ###
psi = numerov_integrator(psi = psi_initial, f_n = f_final, N = N,
                                     icl = icl_final, direction = "backward")
psi_R = psi[icl_final:].copy()


### --- RESCALE RIGHT TO TOUCH LEFT --- ###
scaling_factor = psi_L[-1] / psi_R[0]
psi_R_scaled = psi_R * scaling_factor

### --- STITCHING RIGHT AND LEFT --- ###
# psi_L ends at 'icl'. psi_R_scaled starts at 'icl'.
# We assume psi_L[-1] == psi_R_scaled[0] (we just forced it).
# So we take all of L, and everything from R *except* the first point (to avoid duplicates).
psi_final = np.concatenate((psi_L, psi_R_scaled[1:]))

### --- NORMALIZATION --- ###
# Calculate the Integral of |psi|^2 dx
# (Simple Riemann sum is sufficient for this grid density)
probability_density = psi_final ** 2
area = np.sum(probability_density) * dx

# Calculate normalization constant A = 1 / sqrt(Area)
norm_constant = 1.0 / np.sqrt(area)

# Apply it
psi_normalized = psi_final * norm_constant
prob_density_normalized = psi_normalized ** 2

print(f"Normalization Complete. Total Area = {np.sum(prob_density_normalized) * dx:.5f}")


# 5. PLOTTING
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(x, psi_normalized, label=f'n={target_number_of_nodes}, E={epsilon:.4f}')
plt.xlabel('Position (x)')
plt.ylabel('Psi')
plt.title(f'Harmonic Oscillator (n={target_number_of_nodes})')
plt.grid(True)
plt.legend()
plt.show()






