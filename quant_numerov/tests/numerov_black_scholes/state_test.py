import pytest
import numpy as np
from numerov_black_scholes.state import SimulationRunner

class MockSystem:
    def __init__(self, T_final=1.0, dt=0.1):
        self.T_final = T_final
        self.dt = dt

class MockSolver:
    def step(self, u_curr, boundary_left=None, boundary_right=None):
        # We must return the new array so 'u_curr' isn't None
        next_u = u_curr + 1.0
        return next_u

@pytest.fixture # Fixtures are setup code that run before tests
def basic_runner():
    """Sets up a standard runner with 10 steps."""
    sys = MockSystem()
    solver = MockSolver()
    u0 = np.zeros(5)
    return SimulationRunner(system=sys, solver=solver, initial_condition=u0)

# --- Tests --- #
def test_initializaiton(basic_runner):
    """Does it start at t=0 with the correct data?"""
    assert basic_runner.t == 0
    np.testing.assert_array_equal(basic_runner.u_curr, np.zeros(5))
    assert basic_runner.history == []

def test_single_step_update(basic_runner):
    """Does the loop update u and t correctly?"""
    # Mnually run one step logic
    basic_runner.u_curr = basic_runner.solver.step(basic_runner.u_curr)
    basic_runner.t += basic_runner.sys.dt

    # Check u_curr (0 + 1 = 1)
    expected_u = np.ones(5)
    np.testing.assert_array_equal(basic_runner.u_curr, expected_u)

    # Check time (approx 0.1)
    assert basic_runner.t == pytest.approx(0.1, rel=1e-6)

def test_full_execution_count(basic_runner):
    """Does it run exactly the right number of steps?"""
    # Expect 10 steps (1.0 / 0.1) -> Final value should be 10.0
    basic_runner.run()

    expected_final_u = np.ones(5) * 10.0
    np.testing.assert_array_almost_equal(basic_runner.u_curr, expected_final_u)
    assert basic_runner.t == pytest.approx(1.0, rel=1e-6)

def test_history_recording(basic_runner):
    """Does it record history correctly?"""
    # With 10 steps, it saves every stepp (10 //10 = 1)
    # Plus initial stte (t=0) an dfinal state = ~12 snapshots
    basic_runner.run()

    assert len(basic_runner.history) >= 11

    # Check structure of first snapshot
    time, data = basic_runner.history[0]
    assert time == 0.0
    np.testing.assert_array_equal(data, np.zeros(5))

def test_large_simulation_storage():
    """Does it save SPACE when running many steps?"""
    sys = MockSystem(T_final=10.0, dt=0.1)
    solver = MockSolver()
    u0 = np.zeros(5)
    runner = SimulationRunner(sys, solver,u0)

    runner.run()

    # It should NOT save 100+ items.
    # Logic: Initial +every 10th + Final ~ 12 items
    assert len(runner.history) <= 15

def test_short_simulation_edge_case():
    """Does it handle simulations with < 10 steps without crashing?"""
    # This specifically tests the 'ZeroDivisionError' fix
    sys = MockSystem(T_final=0.4, dt=0.1)
    solver = MockSolver()
    u0 = np.zeros(5)
    runner = SimulationRunner(sys, solver,u0)

    # This should NOT raise ZeroDivisionError
    runner.run()

    # There should be at least two snapshots (start and end) saved in history
    assert len(runner.history) >= 2