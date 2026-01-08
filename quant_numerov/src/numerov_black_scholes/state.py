import numpy as np
from numerov_black_scholes.system import BlackScholesSystem

class SimulationRunner:
    def __init__(self, system, solver, initial_condition):
        self.sys = system
        self.solver = solver

        self.u_curr = initial_condition.copy()
        self.t = 0.0 # Time Clock


        self.history = [] # To store (time, array) tuples

    def run(self):
        # Number of steps
        steps = int(self.sys.T_final / self.sys.dt)

        # Starting "snapshot"
        self.history.append((self.t, self.u_curr.copy()))

        # ZeroDivisionError fix
        save_interval = max(1, steps // 10)

        print(f"Starting simulation: {steps} steps...")

        # Grab boundary values from the Initial Condition
        # We assume they are static (Constant) for the transformed variable U
        bc_left = self.u_curr[0]
        bc_right = self.u_curr[-1]

        # THE LOOP: "Start walking."
        for n in range(steps):
            # Pass the boundaries explicitly!
            self.u_curr = self.solver.step(
                self.u_curr,
                boundary_left=bc_left,
                boundary_right=bc_right
            )

            self.t += self.sys.dt

            if n % save_interval == 0:
                self.history.append((self.t, self.u_curr.copy()))

        self.history.append((self.t, self.u_curr.copy()))
        print("Simulation complete.")