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
#   -
#
#
#
#
#
#
#




import numpy as np
import pandas as pd








