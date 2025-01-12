import tensorflow as tf
from abc import ABC, abstractmethod

class NavierStokesBase(ABC):
    """Abstract base class for Navier-Stokes equations"""
    def __init__(self, rho=1.0, mu=0.01):
        self.rho = rho    # density
        self.mu = mu      # dynamic viscosity
        self.nu = mu/rho  # kinematic viscosity
    
    @abstractmethod
    def continuity(self, *args, **kwargs):
        """Continuity equation: ∇·u = 0"""
        pass
    
    @abstractmethod
    def get_residuals(self, *args, **kwargs):
        """Calculate all NS residuals"""
        pass

    def _safe_gradients(self, y, x):
        """Safely compute gradients with error checking"""
        grads = tf.gradients(y, x)
        if grads[0] is None:
            raise ValueError(f"Failed to compute gradient of {y} with respect to {x}")
        return grads[0]

class NavierStokes2D(NavierStokesBase):
    """2D Navier-Stokes equations implementation"""
    
    def continuity(self, u, v, x, y):
        """Continuity equation: ∇·u = 0"""
        du_dx = self._safe_gradients(u, x)
        dv_dy = self._safe_gradients(v, y)
        return du_dx + dv_dy

    def momentum_x(self, u, v, p, x, y):
        """X-momentum equation"""
        # First derivatives
        du_dx = self._safe_gradients(u, x)
        du_dy = self._safe_gradients(u, y)
        dp_dx = self._safe_gradients(p, x)
        
        # Second derivatives
        du_dxx = self._safe_gradients(du_dx, x)
        du_dyy = self._safe_gradients(du_dy, y)
        
        return self._momentum_terms(
            velocity_terms=(u * du_dx + v * du_dy),
            pressure_grad=dp_dx,
            diffusion_terms=(du_dxx + du_dyy)
        )

    def momentum_y(self, u, v, p, x, y):
        """Y-momentum equation"""
        # First derivatives
        dv_dx = self._safe_gradients(v, x)
        dv_dy = self._safe_gradients(v, y)
        dp_dy = self._safe_gradients(p, y)
        
        # Second derivatives
        dv_dxx = self._safe_gradients(dv_dx, x)
        dv_dyy = self._safe_gradients(dv_dy, y)
        
        return self._momentum_terms(
            velocity_terms=(u * dv_dx + v * dv_dy),
            pressure_grad=dp_dy,
            diffusion_terms=(dv_dxx + dv_dyy)
        )
    
    def _momentum_terms(self, velocity_terms, pressure_grad, diffusion_terms):
        """Calculate momentum equation terms"""
        convection = self.rho * velocity_terms
        pressure = pressure_grad
        diffusion = self.mu * diffusion_terms
        return convection + pressure - diffusion

    def get_residuals(self, u, v, p, x, y):
        """Calculate all 2D NS residuals"""
        return {
            'continuity': self.continuity(u, v, x, y),
            'momentum_x': self.momentum_x(u, v, p, x, y),
            'momentum_y': self.momentum_y(u, v, p, x, y)
        }

class NavierStokes3D(NavierStokesBase):
    """3D Navier-Stokes equations implementation"""
    
    def continuity(self, u, v, w, x, y, z):
        """Continuity equation: ∇·u = 0"""
        du_dx = self._safe_gradients(u, x)
        dv_dy = self._safe_gradients(v, y)
        dw_dz = self._safe_gradients(w, z)
        return du_dx + dv_dy + dw_dz

    def momentum_x(self, u, v, w, p, x, y, z):
        """X-momentum equation"""
        # First derivatives
        du_dx = self._safe_gradients(u, x)
        du_dy = self._safe_gradients(u, y)
        du_dz = self._safe_gradients(u, z)
        dp_dx = self._safe_gradients(p, x)
        
        # Second derivatives
        du_dxx = self._safe_gradients(du_dx, x)
        du_dyy = self._safe_gradients(du_dy, y)
        du_dzz = self._safe_gradients(du_dz, z)
        
        return self._momentum_terms(
            velocity_terms=(u * du_dx + v * du_dy + w * du_dz),
            pressure_grad=dp_dx,
            diffusion_terms=(du_dxx + du_dyy + du_dzz)
        )

    def momentum_y(self, u, v, w, p, x, y, z):
        """Y-momentum equation"""
        # First derivatives
        dv_dx = self._safe_gradients(v, x)
        dv_dy = self._safe_gradients(v, y)
        dv_dz = self._safe_gradients(v, z)
        dp_dy = self._safe_gradients(p, y)
        
        # Second derivatives
        dv_dxx = self._safe_gradients(dv_dx, x)
        dv_dyy = self._safe_gradients(dv_dy, y)
        dv_dzz = self._safe_gradients(dv_dz, z)
        
        return self._momentum_terms(
            velocity_terms=(u * dv_dx + v * dv_dy + w * dv_dz),
            pressure_grad=dp_dy,
            diffusion_terms=(dv_dxx + dv_dyy + dv_dzz)
        )
    
    def momentum_z(self, u, v, w, p, x, y, z):
        """Z-momentum equation"""
        # First derivatives
        dw_dx = self._safe_gradients(w, x)
        dw_dy = self._safe_gradients(w, y)
        dw_dz = self._safe_gradients(w, z)
        dp_dz = self._safe_gradients(p, z)
        
        # Second derivatives
        dw_dxx = self._safe_gradients(dw_dx, x)
        dw_dyy = self._safe_gradients(dw_dy, y)
        dw_dzz = self._safe_gradients(dw_dz, z)
        
        return self._momentum_terms(
            velocity_terms=(u * dw_dx + v * dw_dy + w * dw_dz),
            pressure_grad=dp_dz,
            diffusion_terms=(dw_dxx + dw_dyy + dw_dzz)
        )
    
    def _momentum_terms(self, velocity_terms, pressure_grad, diffusion_terms):
        """Calculate momentum equation terms"""
        convection = self.rho * velocity_terms
        pressure = pressure_grad
        diffusion = self.mu * diffusion_terms
        return convection + pressure - diffusion

    def get_residuals(self, u, v, w, p, x, y, z):
        """Calculate all 3D NS residuals"""
        return {
            'continuity': self.continuity(u, v, w, x, y, z),
            'momentum_x': self.momentum_x(u, v, w, p, x, y, z),
            'momentum_y': self.momentum_y(u, v, w, p, x, y, z),
            'momentum_z': self.momentum_z(u, v, w, p, x, y, z)
        }