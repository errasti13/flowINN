import tensorflow as tf
from typing import Tuple
from .navier_stokes import NavierStokes

class SteadyNavierStokes2D(NavierStokes):
    """2D Steady Navier-Stokes equations solver."""

    def __init__(self, Re: float = 1000.0):
        """
        Initialize the solver.
        
        Args:
            Re (float): Reynolds number
        """
        self.Re = Re

    @tf.function
    def get_residuals(self, velocities: tf.Tensor, pressure: tf.Tensor, coords: list, tape) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Calculate 2D Steady Navier-Stokes residuals.
        
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
        p = tf.reshape(pressure, [-1])
        
        tape.watch([x, y])
        
        [u_x, u_y, v_x, v_y, p_x, p_y] = self._compute_first_derivatives(
            [u, v, p], [x, y], tape
        )
        
        [u_xx, u_xy, u_yx, u_yy, v_xx, v_xy, v_yx, v_yy] = self._compute_second_derivatives(
            [u_x, u_y, v_x, v_y], [x, y], tape
        )

        # Continuity equation: ∂u/∂x + ∂v/∂y = 0
        continuity = u_x + v_y
        
        # Momentum equation x: u∂u/∂x + v∂u/∂y = -∂p/∂x + 1/Re(∂²u/∂x² + ∂²u/∂y²)
        momentum_x = (
            u * u_x + v * u_y +
            p_x -
            1 / self.Re * (u_xx + u_yy)
        )
        
        # Momentum equation y: u∂v/∂x + v∂v/∂y = -∂p/∂y + 1/Re(∂²v/∂x² + ∂²v/∂y²)
        momentum_y = (
            u * v_x + v * v_y +
            p_y  -
            1 / self.Re * (v_xx + v_yy)
        )

        continuity = tf.reshape(continuity, [-1])
        momentum_x = tf.reshape(momentum_x, [-1])
        momentum_y = tf.reshape(momentum_y, [-1])

        return continuity, momentum_x, momentum_y

    @tf.function
    def _compute_first_derivatives(self, variables, coords, tape):
        return super()._compute_first_derivatives(variables, coords, tape)

    @tf.function  
    def _compute_second_derivatives(self, first_derivs, coords, tape):
        return super()._compute_second_derivatives(first_derivs, coords, tape)
