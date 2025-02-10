import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Tuple


class BaseNavierStokes(ABC):
    """
    Abstract base class for Navier-Stokes equations.

    Attributes:
        nu (float): Kinematic viscosity.
    """

    def __init__(self, nu: float) -> None:
        """
        Initialize with kinematic viscosity.

        Args:
            nu (float): Kinematic viscosity.
        """
        self._nu: float = None
        self.nu = nu

    @property
    def nu(self) -> float:
        """Get kinematic viscosity."""
        return self._nu

    @nu.setter
    def nu(self, value: float) -> None:
        """Set kinematic viscosity with validation."""
        if not isinstance(value, (int, float)):
            raise TypeError("Kinematic viscosity (nu) must be a number")
        if value <= 0:
            raise ValueError("Kinematic viscosity (nu) must be positive")
        self._nu = float(value)

    @abstractmethod
    def get_residuals(self, *args, **kwargs) -> Tuple[tf.Tensor, ...]:
        """
        Abstract method to compute the residuals of the Navier-Stokes equations.

        This method must be implemented by subclasses to define the specific
        Navier-Stokes equations being solved.

        Returns:
            Tuple[tf.Tensor, ...]: Residuals of the Navier-Stokes equations.
        """
        pass


class NavierStokes3D(BaseNavierStokes):
    """
    3D Navier-Stokes equations.

    Inherits from BaseNavierStokes and implements the get_residuals method
    for the 3D case.
    """

    def __init__(self, nu: float = 0.01) -> None:
        """
        Initialize with kinematic viscosity.

        Args:
            nu (float): Kinematic viscosity.
        """
        super().__init__(nu)

    def get_residuals(self, u: tf.Tensor, v: tf.Tensor, w: tf.Tensor, p: tf.Tensor,
                      x: tf.Tensor, y: tf.Tensor, z: tf.Tensor, tape: tf.GradientTape) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Calculate the residuals of the 3D Navier-Stokes equations.

        Args:
            u (tf.Tensor): Velocity component in the x-direction.
            v (tf.Tensor): Velocity component in the y-direction.
            w (tf.Tensor): Velocity component in the z-direction.
            p (tf.Tensor): Pressure.
            x (tf.Tensor): Spatial coordinate in the x-direction.
            y (tf.Tensor): Spatial coordinate in the y-direction.
            z (tf.Tensor): Spatial coordinate in the z-direction.
            tape (tf.GradientTape): TensorFlow GradientTape for automatic differentiation.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: Residuals for continuity and momentum equations (continuity, momentum_u, momentum_v, momentum_w).
        """
        tape.watch([x, y, z])
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
        momentum_u = u * u_x + v * u_y + w * u_z + p_x - self.nu * (u_xx + u_yy + u_zz)
        momentum_v = u * v_x + v * v_y + w * v_z + p_y - self.nu * (v_xx + v_yy + v_zz)
        momentum_w = u * w_x + v * w_y + w * w_z + p_z - self.nu * (w_xx + w_yy + w_zz)

        return continuity, momentum_u, momentum_v, momentum_w


class NavierStokes2D(BaseNavierStokes):
    """
    2D Navier-Stokes equations.

    Inherits from BaseNavierStokes and implements the get_residuals method
    for the 2D case.
    """

    def __init__(self, nu: float = 0.01) -> None:
        """
        Initialize with kinematic viscosity.

        Args:
            nu (float): Kinematic viscosity.
        """
        super().__init__(nu)

    def _compute_first_derivatives(self, u: tf.Tensor, v: tf.Tensor, p: tf.Tensor,
                                   x: tf.Tensor, y: tf.Tensor, tape: tf.GradientTape) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute first-order derivatives.

        Args:
            u (tf.Tensor): Velocity component in the x-direction.
            v (tf.Tensor): Velocity component in the y-direction.
            p (tf.Tensor): Pressure.
            x (tf.Tensor): Spatial coordinate in the x-direction.
            y (tf.Tensor): Spatial coordinate in the y-direction.
            tape (tf.GradientTape): TensorFlow GradientTape for automatic differentiation.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: First-order derivatives (u_x, u_y, v_x, v_y, p_x, p_y).
        """
        u_x = tape.gradient(u, x)
        u_y = tape.gradient(u, y)
        v_x = tape.gradient(v, x)
        v_y = tape.gradient(v, y)
        p_x = tape.gradient(p, x)
        p_y = tape.gradient(p, y)
        return u_x, u_y, v_x, v_y, p_x, p_y

    def _compute_second_derivatives(self, u_x: tf.Tensor, u_y: tf.Tensor, v_x: tf.Tensor,
                                    v_y: tf.Tensor, x: tf.Tensor, y: tf.Tensor, tape: tf.GradientTape) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute second-order derivatives.

        Args:
            u_x (tf.Tensor): First-order derivative of u with respect to x.
            u_y (tf.Tensor): First-order derivative of u with respect to y.
            v_x (tf.Tensor): First-order derivative of v with respect to x.
            v_y (tf.Tensor): First-order derivative of v with respect to y.
            x (tf.Tensor): Spatial coordinate in the x-direction.
            y (tf.Tensor): Spatial coordinate in the y-direction.
            tape (tf.GradientTape): TensorFlow GradientTape for automatic differentiation.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: Second-order derivatives (u_xx, u_yy, v_xx, v_yy).
        """
        u_xx = tape.gradient(u_x, x)
        u_yy = tape.gradient(u_y, y)
        v_xx = tape.gradient(v_x, x)
        v_yy = tape.gradient(v_y, y)
        return u_xx, u_yy, v_xx, v_yy

    def _compute_continuity(self, u_x: tf.Tensor, v_y: tf.Tensor) -> tf.Tensor:
        """
        Compute continuity equation residual.

        Args:
            u_x (tf.Tensor): First-order derivative of u with respect to x.
            v_y (tf.Tensor): First-order derivative of v with respect to y.

        Returns:
            tf.Tensor: Continuity equation residual.
        """
        return u_x + v_y

    def _compute_momentum(self, u: tf.Tensor, v: tf.Tensor, u_x: tf.Tensor, u_y: tf.Tensor,
                           v_x: tf.Tensor, v_y: tf.Tensor, u_xx: tf.Tensor, u_yy: tf.Tensor,
                           v_xx: tf.Tensor, v_yy: tf.Tensor, p_x: tf.Tensor, p_y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute momentum equation residuals.

        Args:
            u (tf.Tensor): Velocity component in the x-direction.
            v (tf.Tensor): Velocity component in the y-direction.
            u_x (tf.Tensor): First-order derivative of u with respect to x.
            u_y (tf.Tensor): First-order derivative of u with respect to y.
            v_x (tf.Tensor): First-order derivative of v with respect to x.
            v_y (tf.Tensor): First-order derivative of v with respect to y.
            u_xx (tf.Tensor): Second-order derivative of u with respect to x.
            u_yy (tf.Tensor): Second-order derivative of u with respect to y.
            v_xx (tf.Tensor): Second-order derivative of v with respect to x.
            v_yy (tf.Tensor): Second-order derivative of v with respect to y.
            p_x (tf.Tensor): First-order derivative of pressure with respect to x.
            p_y (tf.Tensor): First-order derivative of pressure with respect to y.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Momentum equation residuals (momentum_x, momentum_y).
        """
        momentum_x = (u * u_x + v * u_y) + p_x - self.nu * (u_xx + u_yy)
        momentum_y = (u * v_x + v * v_y) + p_y - self.nu * (v_xx + v_yy)
        return momentum_x, momentum_y

    def get_residuals(self, u: tf.Tensor, v: tf.Tensor, p: tf.Tensor,
                      x: tf.Tensor, y: tf.Tensor, tape: tf.GradientTape) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Calculate the residuals of the 2D Navier-Stokes equations.

        Args:
            u (tf.Tensor): Velocity component in the x-direction.
            v (tf.Tensor): Velocity component in the y-direction.
            p (tf.Tensor): Pressure.
            x (tf.Tensor): Spatial coordinate in the x-direction.
            y (tf.Tensor): Spatial coordinate in the y-direction.
            tape (tf.GradientTape): TensorFlow GradientTape for automatic differentiation.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Residuals for continuity and momentum equations (continuity, momentum_x, momentum_y).
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
