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

        # THE LOOP: "Start walking."
        for n in range(steps):
            self.u_curr = self.solver.step(self.u_curr)
            self.t += self.sys.dt

            if n % (steps // 10) == 0:
                self.history.append((self.t, self.u_curr.copy()))