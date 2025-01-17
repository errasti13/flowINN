import tensorflow as tf


class NavierStokes3D:
    def __init__(self, nu = 0.01):
        self.nu = nu      # dynamic viscosity

    def continuity(self, u, v, w, x, y, z):
        """Continuity equation: ∇·u = 0"""
        du_dx = tf.gradient(u, x)[0]
        dv_dy = tf.gradient(v, y)[0]
        dw_dz = tf.gradient(w, z)[0]
        return du_dx + dv_dy + dw_dz

    def momentum_x(self, u, v, w, p, x, y, z):
        """X-momentum: ρ(∂u/∂t + u∂u/∂x + v∂u/∂y) = -∂p/∂x + μ(∂²u/∂x² + ∂²u/∂y²)"""
        du_dx = tf.gradient(u, x)[0]
        du_dy = tf.gradient(u, y)[0]
        du_dz = tf.gradient(u, z)[0]
        dp_dx = tf.gradient(p, x)[0]
        
        du_dxx = tf.gradient(du_dx, x)[0]
        du_dyy = tf.gradient(du_dy, y)[0]
        du_dzz = tf.gradient(du_dz, z)[0]
        
        convection = self.rho * (u * du_dx + v * du_dy + w * du_dz)
        pressure = dp_dx
        diffusion = self.mu * (du_dxx + du_dyy + du_dzz)
        
        return convection + pressure - diffusion

    def momentum_y(self, u, v, w, p, x, y, z):
        """Y-momentum: ρ(∂v/∂t + u∂v/∂x + v∂v/∂y) = -∂p/∂y + μ(∂²v/∂x² + ∂²v/∂y²)"""
        dv_dx = tf.gradient(v, x)[0]
        dv_dy = tf.gradient(v, y)[0]
        dv_dz = tf.gradient(v, z)[0]
        dp_dy = tf.gradient(p, y)[0]
        
        dv_dxx = tf.gradient(dv_dx, x)[0]
        dv_dyy = tf.gradient(dv_dy, y)[0]
        dv_dzz = tf.gradient(dv_dz, z)[0]
        
        convection = self.rho * (u * dv_dx + v * dv_dy + w * dv_dz)
        pressure = dp_dy
        diffusion = self.mu * (dv_dxx + dv_dyy + dv_dzz)
        
        return convection + pressure - diffusion
    
    def momentum_z(self, u, v, w, p, x, y, z):
        """Y-momentum: ρ(∂v/∂t + u∂v/∂x + v∂v/∂y) = -∂p/∂y + μ(∂²v/∂x² + ∂²v/∂y²)"""
        dw_dx = tf.gradient(w, x)[0]
        dw_dy = tf.gradient(w, y)[0]
        dw_dz = tf.gradient(w, z)[0]
        dp_dz = tf.gradient(p, z)[0]
        
        dw_dxx = tf.gradient(dw_dx, x)[0]
        dw_dyy = tf.gradient(dw_dy, y)[0]
        dw_dzz = tf.gradient(dw_dz, z)[0]
        
        convection = self.rho * (u * dw_dx + v * dw_dy + w * dw_dz)
        pressure = dp_dz
        diffusion = self.mu * (dw_dxx + dw_dyy + dw_dzz)
        
        return convection + pressure - diffusion

    def get_residuals(self, u, v, w, p, x, y, z):
        """Calculate all NS residuals"""
        return {
            'continuity': self.continuity(u, v, w, x, y, z),
            'momentum_x': self.momentum_x(u, v, w, x, y, z),
            'momentum_y': self.momentum_y(u, v, w, x, y, z),
            'momentum_z': self.momentum_z(u, v, w, x, y, z)
        }
    

class NavierStokes2D:
    def __init__(self, nu = 0.01):
        self.nu = nu  # kinematic viscosity

    def get_residuals(self, u, v, p, x, y, tape):
        """Calculate all NS residuals"""

        tape.watch([x, y])
        u_x = tape.gradient(u, x)
        u_y = tape.gradient(u, y)
        v_x = tape.gradient(v, x)
        v_y = tape.gradient(v, y)
        p_x = tape.gradient(p, x)
        p_y = tape.gradient(p, y)

        # Compute second derivatives (second order)
        u_xx = tape.gradient(u_x, x)
        u_yy = tape.gradient(u_y, y)
        v_xx = tape.gradient(v_x, x)
        v_yy = tape.gradient(v_y, y)

        # Continuity equation
        continuity = u_x + v_y

        # Momentum equations
        momentum_u = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        momentum_v = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)

        return continuity, momentum_u, momentum_v
