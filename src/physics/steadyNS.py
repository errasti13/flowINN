import tensorflow as tf


class NavierStokes3D:
    def __init__(self, rho=1.0, mu=0.01):
        self.rho = rho    # density
        self.mu = mu      # dynamic viscosity
        self.nu = mu/rho  # kinematic viscosity

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
    def __init__(self, rho=1.0, mu=0.01):
        self.rho = rho    # density
        self.mu = mu      # dynamic viscosity
        self.nu = mu/rho  # kinematic viscosity

    def continuity(self, u, v, x, y):
        """Continuity equation: ∇·u = 0"""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y])
            # Reshape inputs to ensure proper broadcasting
            u = tf.reshape(u, [-1])
            v = tf.reshape(v, [-1])
            x = tf.reshape(x, [-1])
            y = tf.reshape(y, [-1])
            
            du_dx = tape.gradient(u, x)
            dv_dy = tape.gradient(v, y)
            
        # Convert None gradients to zeros
        du_dx = tf.zeros_like(x) if du_dx is None else du_dx
        dv_dy = tf.zeros_like(y) if dv_dy is None else dv_dy
        
        return du_dx + dv_dy

    def momentum_x(self, u, v, p, x, y):
        """X-momentum equation"""
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y])
            # Reshape inputs
            u = tf.reshape(u, [-1])
            v = tf.reshape(v, [-1])
            p = tf.reshape(p, [-1])
            x = tf.reshape(x, [-1])
            y = tf.reshape(y, [-1])
            
            # First derivatives
            du_dx = tape2.gradient(u, x)
            du_dy = tape2.gradient(u, y)
            dp_dx = tape2.gradient(p, x)
            
            # Handle None gradients
            du_dx = tf.zeros_like(x) if du_dx is None else du_dx
            du_dy = tf.zeros_like(y) if du_dy is None else du_dy
            dp_dx = tf.zeros_like(x) if dp_dx is None else dp_dx
            
        # Second derivatives using another tape
        with tf.GradientTape(persistent=True) as tape3:
            tape3.watch([x, y])
            du_dxx = tape3.gradient(du_dx, x)
            du_dyy = tape3.gradient(du_dy, y)
            
            # Handle None gradients
            du_dxx = tf.zeros_like(x) if du_dxx is None else du_dxx
            du_dyy = tf.zeros_like(y) if du_dyy is None else du_dyy
        
        convection = self.rho * (u * du_dx + v * du_dy)
        pressure = dp_dx
        diffusion = self.mu * (du_dxx + du_dyy)
        
        return convection + pressure - diffusion

    def momentum_y(self, u, v, p, x, y):
        """Y-momentum equation"""
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y])
            # Reshape inputs
            u = tf.reshape(u, [-1])
            v = tf.reshape(v, [-1])
            p = tf.reshape(p, [-1])
            x = tf.reshape(x, [-1])
            y = tf.reshape(y, [-1])
            
            # First derivatives
            dv_dx = tape2.gradient(v, x)
            dv_dy = tape2.gradient(v, y)
            dp_dy = tape2.gradient(p, y)
            
            # Handle None gradients
            dv_dx = tf.zeros_like(x) if dv_dx is None else dv_dx
            dv_dy = tf.zeros_like(y) if dv_dy is None else dv_dy
            dp_dy = tf.zeros_like(y) if dp_dy is None else dp_dy
            
        # Second derivatives using another tape
        with tf.GradientTape(persistent=True) as tape3:
            tape3.watch([x, y])
            dv_dxx = tape3.gradient(dv_dx, x)
            dv_dyy = tape3.gradient(dv_dy, y)
            
            # Handle None gradients
            dv_dxx = tf.zeros_like(x) if dv_dxx is None else dv_dxx
            dv_dyy = tf.zeros_like(y) if dv_dyy is None else dv_dyy
        
        convection = self.rho * (u * dv_dx + v * dv_dy)
        pressure = dp_dy
        diffusion = self.mu * (dv_dxx + dv_dyy)
        
        return convection + pressure - diffusion

    def get_residuals(self, u, v, p, x, y):
        """Calculate all NS residuals"""
        return {
            'continuity': self.continuity(u, v, x, y),
            'momentum_x': self.momentum_x(u, v, p, x, y),
            'momentum_y': self.momentum_y(u, v, p, x, y),
        }