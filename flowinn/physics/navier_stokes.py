import tensorflow as tf
from typing import Tuple
from abc import ABC, abstractmethod


class NavierStokes(ABC):
    """Base class for Navier-Stokes equations."""
    
    def __init__(self, nu: float = 0.01) -> None:
        self._nu: float = None
        self.nu = nu

    @property
    def nu(self) -> float:
        """Kinematic viscosity coefficient."""
        return self._nu

    @nu.setter
    def nu(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError("Kinematic viscosity (nu) must be a number")
        if value <= 0:
            raise ValueError("Kinematic viscosity (nu) must be positive")
        self._nu = float(value)

    def _compute_first_derivatives(self, variables: list, coords: list, tape) -> list:
        """Compute first-order derivatives for each variable with respect to each coordinate."""
        return [tape.gradient(var, coord) for var in variables for coord in coords]

    def _compute_second_derivatives(self, first_derivatives: list, coordinates: list, tape) -> list:
        """Compute second-order derivatives."""
        return [tape.gradient(d, coord) for d in first_derivatives for coord in coordinates]

    @abstractmethod
    def get_residuals(self, *args, **kwargs) -> Tuple[tf.Tensor, ...]:
        """Calculate Navier-Stokes residuals."""
        pass
