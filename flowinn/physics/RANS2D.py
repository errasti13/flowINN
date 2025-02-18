import tensorflow as tf
from typing import Tuple
from flowinn.physics.steadyNS import NavierStokes
class RANS2D(NavierStokes):
    def __init__(self, rho = 1.0, nu = 0.01):
        self.rho = rho
        self.nu  = nu

    def get_residuals(self, U: tf.Tensor, V: tf.Tensor, P: tf.Tensor,
                    u: tf.Tensor, v: tf.Tensor,
                    x: tf.Tensor, y: tf.Tensor, tape) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Calculate 2D Navier–Stokes residuals for the modified formulation.

        This formulation uses uppercase fields (U, V, P) for the convective,
        pressure, and viscous terms, and additional contributions from the
        lower-case velocities (u, v) through their squared and product terms.

        Equations:
            loss₁ = U·(∂U/∂x) + V·(∂U/∂y) + (1/ρ)·(∂P/∂x) - ν·(∂²U/∂x² + ∂²U/∂y²)
                    + ∂(u²)/∂x + ∂(uv)/∂y

            loss₂ = U·(∂V/∂x) + V·(∂V/∂y) + (1/ρ)·(∂P/∂y) - ν·(∂²V/∂x² + ∂²V/∂y²)
                    + ∂(uv)/∂x + ∂(v²)/∂y

            loss₃ = ∂U/∂x + ∂V/∂y   (Continuity)

        Args:
            U, V: Uppercase velocity components for momentum terms.
            P: Pressure field.
            u, v: Lowercase velocity components used in the extra convective terms.
            x, y: Spatial coordinates.
            tape: Gradient tape for automatic differentiation.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: (continuity, momentum_x, momentum_y) residuals.
        """
        tape.watch([x, y])

        # Compute first derivatives for U, V, and P:
        # Returns [U_x, U_y, V_x, V_y, P_x, P_y]
        [U_x, U_y, V_x, V_y, P_x, P_y] = self._compute_first_derivatives([U, V, P], [x, y], tape)

        # Compute second derivatives for U and V:
        # Returns [U_xx, U_xy, U_yx, U_yy, V_xx, V_xy, V_yx, V_yy]
        [U_xx, U_xy, U_yx, U_yy, V_xx, V_xy, V_yx, V_yy] = self._compute_second_derivatives(
            [U_x, U_y, V_x, V_y], [x, y], tape
        )

        # Extra nonlinear terms computed from lowercase velocities
        u_sq = u ** 2
        uv   = u * v
        v_sq = v ** 2
        # For u², we need only ∂(u²)/∂x
        [u_sq_x, _] = self._compute_first_derivatives([u_sq], [x, y], tape)
        # For uv, we need both ∂(uv)/∂x and ∂(uv)/∂y
        [uv_x, uv_y] = self._compute_first_derivatives([uv], [x, y], tape)
        # For v², we need only ∂(v²)/∂y
        [_, v_sq_y] = self._compute_first_derivatives([v_sq], [x, y], tape)

        # Continuity residual (divergence of U and V)
        continuity = U_x + V_y

        # Momentum residuals with extra nonlinear terms
        momentum_x = U * U_x + V * U_y + (1 / self.rho) * P_x - self.nu * (U_xx + U_yy) \
                    + u_sq_x + uv_y
        momentum_y = U * V_x + V * V_y + (1 / self.rho) * P_y - self.nu * (V_xx + V_yy) \
                    + uv_x + v_sq_y

        return continuity, momentum_x, momentum_y



        

