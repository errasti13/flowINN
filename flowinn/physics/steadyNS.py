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


class NavierStokes2D(NavierStokes):
    """2D Navier-Stokes equations solver."""

    def get_residuals(self, velocities: tf.Tensor, pressure: tf.Tensor, coords: list, tape) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Calculate 2D Navier-Stokes residuals.
        
        Args:
            velocities: Tensor of velocity components [u, v]
            pressure: Pressure tensor
            coords: List of coordinate tensors [x, y]
            tape: Gradient tape for automatic differentiation
            
        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: (continuity, momentum_x, momentum_y) residuals
        """
        x, y = coords
        u = velocities[:, 0]
        v = velocities[:, 1]
        
        tape.watch([x, y])
        
        # First derivatives
        [u_x, u_y, v_x, v_y, p_x, p_y] = self._compute_first_derivatives(
            [u, v, pressure], [x, y], tape
        )
        
        # Second derivatives
        [u_xx, u_xy, u_yx, u_yy, v_xx, v_xy, v_yx, v_yy] = self._compute_second_derivatives(
            [u_x, u_y, v_x, v_y], [x, y], tape
        )

        # Continuity equation
        continuity = u_x + v_y

        # Momentum equations
        momentum_x = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        momentum_y = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)

        # Reshape residuals to ensure consistent shape
        continuity = tf.reshape(continuity, [-1])
        momentum_x = tf.reshape(momentum_x, [-1])
        momentum_y = tf.reshape(momentum_y, [-1])

        return continuity, momentum_x, momentum_y


class NavierStokes3D(NavierStokes):
    """3D Navier-Stokes equations solver."""

    def get_residuals(self, velocities: tf.Tensor, pressure: tf.Tensor, coords: list, tape) -> Tuple[tf.Tensor, ...]:
        """
        Calculate 3D Navier-Stokes residuals.
        
        Args:
            velocities: Tensor of velocity components [u, v, w]
            pressure: Pressure tensor
            coords: List of coordinate tensors [x, y, z]
            tape: Gradient tape for automatic differentiation
            
        Returns:
            Tuple[tf.Tensor, ...]: (continuity, momentum_x, momentum_y, momentum_z) residuals
        """
        x, y, z = coords
        u = velocities[:, 0]
        v = velocities[:, 1]
        w = velocities[:, 2]
        
        tape.watch([x, y, z])
        
        # First derivatives
        [u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z, p_x, p_y, p_z] = self._compute_first_derivatives(
            [u, v, w, pressure], [x, y, z], tape
        )
        
        # Second derivatives
        [u_xx, u_yy, u_zz, v_xx, v_yy, v_zz, w_xx, w_yy, w_zz] = [
            tape.gradient(d, c) for d, c in [
                (u_x, x), (u_y, y), (u_z, z),
                (v_x, x), (v_y, y), (v_z, z),
                (w_x, x), (w_y, y), (w_z, z)
            ]
        ]

        # Continuity equation
        continuity = u_x + v_y + w_z

        # Momentum equations
        momentum_x = u * u_x + v * u_y + w * u_z + p_x - self.nu * (u_xx + u_yy + u_zz)
        momentum_y = u * v_x + v * v_y + w * v_z + p_y - self.nu * (v_xx + v_yy + v_zz)
        momentum_z = u * w_x + v * w_y + w * w_z + p_z - self.nu * (w_xx + w_yy + w_zz)

        # Reshape residuals to ensure consistent shape
        continuity = tf.reshape(continuity, [-1])
        momentum_x = tf.reshape(momentum_x, [-1])
        momentum_y = tf.reshape(momentum_y, [-1])
        momentum_z = tf.reshape(momentum_z, [-1])

        return continuity, momentum_x, momentum_y, momentum_z
