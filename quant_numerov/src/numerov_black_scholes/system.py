import numpy as np
from dataclasses import dataclass, field

@dataclass(frozen=True)
class black_scholes_system():
    L: int                                   # Length of "Bar"
    dt: float                                # Timestep
    N_points: int                            # Number of Points in the Grid
    T_final: float                           # Final Time
    dx: float = field(init=False)            # Grid Spacing
    x_grid: np.ndarray = field(init=False)   # Grid

    # Dynamic dataclass creation
    # We've frozen this class because the system shouldn't be changing
    # But there are values that should be calculated from the user given variables
    # This allows us to "bypass" the frozen=True condition and calculate values
    # before our classes "freeze lock" fully engages
    def __post_init__(self):
        # Calculated dx
        calculated_dx = self.L / (self.N_points - 1)
        object.__setattr__(self, 'dx', calculated_dx)

        # Calculated x_grid
        calculated_x_grid = np.linspace(-self.L/2, self.L/2, self.N_points)
        object.__setattr__(self, 'x_grid', calculated_x_grid)
