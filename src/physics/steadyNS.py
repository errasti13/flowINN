import tensorflow as tf


class NavierStokes3D:
    def __init__(self, nu = 0.01):
        self.nu = nu  # kinematic viscosity

    def get_residuals(self, u, v, w, p, x, y, z, tape):
        """Calculate all NS residuals"""

        tape.watch([x, y])
        u_x = tape.gradient(u, x)
        u_y = tape.gradient(u, y)
        u_z = tape.gradient(u, z)
        v_x = tape.gradient(v, x)
        v_y = tape.gradient(v, y)
        v_z = tape.gradient(v, z)
        w_x = tape.gradient(w, x)
        w_y = tape.gradient(w, y)
        w_z = tape.gradient(w, z)
        p_x = tape.gradient(p, x)
        p_y = tape.gradient(p, y)
        p_z = tape.gradient(p, z)

        # Compute second derivatives (second order)
        u_xx = tape.gradient(u_x, x)
        u_yy = tape.gradient(u_y, y)
        u_zz = tape.gradient(u_z, z)
        v_xx = tape.gradient(v_x, x)
        v_yy = tape.gradient(v_y, y)
        v_zz = tape.gradient(v_z, z)
        w_xx = tape.gradient(w_x, x)
        w_yy = tape.gradient(w_y, y)
        w_zz = tape.gradient(w_z, z)

        # Continuity equation
        continuity = u_x + v_y + w_z

        # Momentum equations
        momentum_u = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy + u_zz)
        momentum_v = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy + v_zz)
        momentum_w = u * w_x + v * w_y + p_z - self.nu * (w_xx + w_yy + w_zz)

        return continuity, momentum_u, momentum_v, momentum_w
    

class NavierStokes2D:
    def __init__(self, nu: float = 0.01) -> None:
        self._nu: float = None
        self.nu = nu

    @property
    def nu(self) -> float:
        return self._nu

    @nu.setter
    def nu(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError("Kinematic viscosity (nu) must be a number")
        if value <= 0:
            raise ValueError("Kinematic viscosity (nu) must be positive")
        self._nu = float(value)

    def _compute_first_derivatives(self, u, v, p, x, y, tape) -> tuple:
        """Compute first-order derivatives."""
        u_x = tape.gradient(u, x)
        u_y = tape.gradient(u, y)
        v_x = tape.gradient(v, x)
        v_y = tape.gradient(v, y)
        p_x = tape.gradient(p, x)
        p_y = tape.gradient(p, y)
        return u_x, u_y, v_x, v_y, p_x, p_y

    def _compute_second_derivatives(self, u_x, u_y, v_x, v_y, x, y, tape) -> tuple:
        """Compute second-order derivatives."""
        u_xx = tape.gradient(u_x, x)
        u_yy = tape.gradient(u_y, y)
        v_xx = tape.gradient(v_x, x)
        v_yy = tape.gradient(v_y, y)
        return u_xx, u_yy, v_xx, v_yy

    def _compute_continuity(self, u_x: tf.Tensor, v_y: tf.Tensor) -> tf.Tensor:
        """Compute continuity equation residual."""
        return u_x + v_y

    def _compute_momentum(self, u, v, u_x, u_y, v_x, v_y, u_xx, u_yy, v_xx, v_yy, p_x, p_y) -> tuple:
        """Compute momentum equation residuals."""
        momentum_x = (u * u_x + v * u_y) + p_x - self.nu * (u_xx + u_yy)
        momentum_y = (u * v_x + v * v_y) + p_y - self.nu * (v_xx + v_yy)
        return momentum_x, momentum_y

    def get_residuals(self, u, v, p, x, y, tape) -> tuple:
        """Calculate Navier-Stokes residuals.
        
        Args:
            u, v: Velocity components
            p: Pressure
            x, y: Spatial coordinates
            tape: Gradient tape for automatic differentiation
            
        Returns:
            tuple: (continuity, momentum_x, momentum_y) residuals
        """
        tape.watch([x, y])
        
        # Compute derivatives
        u_x, u_y, v_x, v_y, p_x, p_y = self._compute_first_derivatives(u, v, p, x, y, tape)
        u_xx, u_yy, v_xx, v_yy = self._compute_second_derivatives(u_x, u_y, v_x, v_y, x, y, tape)
        
        # Compute residuals
        continuity = self._compute_continuity(u_x, v_y)
        momentum_x, momentum_y = self._compute_momentum(
            u, v, u_x, u_y, v_x, v_y, u_xx, u_yy, v_xx, v_yy, p_x, p_y
        )
        
        return continuity, momentum_x, momentum_y
