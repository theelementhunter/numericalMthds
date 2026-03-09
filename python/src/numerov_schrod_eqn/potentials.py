from abc import ABC, abstractmethod
import numpy as np

class Potential1D(ABC):
    """Abstract Base Class; Defines what a 1D Potential must look like."""
    @abstractmethod
    def get_potential(self, x: np.ndarray) -> np.ndarray:
        pass #method does nothing. it is a placeholder to enforce the structure of the children classes

class HarmonicOscillator(Potential1D):
    def __init__(self, k: float = 1.0):
        self.k = k

    def get_potential(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * self.k * (x ** 2)