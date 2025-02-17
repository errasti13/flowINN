import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional, Tuple

class BoundaryCondition(ABC):
    """Base class for all boundary conditions."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def apply(self, x: tf.Tensor, y: tf.Tensor, values: Dict[str, Any], tape: Optional[tf.GradientTape] = None) -> Dict[str, tf.Tensor]:
        """Apply the boundary condition."""
        pass

class GradientBC(BoundaryCondition):
    """Boundary condition for gradients of variables."""
    def apply(self, x: tf.Tensor, y: tf.Tensor, values: Dict[str, Any], tape: Optional[tf.GradientTape] = None) -> Dict[str, tf.Tensor]:
        if tape is None:
            raise ValueError("GradientTape is required for gradient boundary conditions")
        
        result = {}
        for var_name, grad_info in values.items():
            if grad_info is None:
                result[var_name] = None
                continue

            direction = grad_info.get('direction', 'normal')
            value = grad_info.get('value', 0.0)
            
            if direction == 'normal':
                # Calculate normal direction based on boundary geometry
                nx = grad_info.get('nx', 0.0)
                ny = grad_info.get('ny', 0.0)
                result[var_name] = {'gradient': value, 'direction': (nx, ny)}
            elif direction in ['x', 'y']:
                result[var_name] = {'gradient': value, 'direction': direction}
            
        return result

class DirichletBC(BoundaryCondition):
    """Dirichlet boundary condition with optional gradient constraints."""
    def apply(self, x: tf.Tensor, y: tf.Tensor, values: Dict[str, Any], tape: Optional[tf.GradientTape] = None) -> Dict[str, tf.Tensor]:
        result = {}
        for var_name, var_info in values.items():
            if isinstance(var_info, dict):
                if 'value' in var_info:
                    result[var_name] = {'value': var_info['value']}
                if 'gradient' in var_info and tape is not None:
                    result[f'{var_name}_gradient'] = {'gradient': var_info['gradient'], 'direction': var_info.get('direction', 'normal')}
            else:
                result[var_name] = {'value': var_info} if var_info is not None else None
        return result

# Update existing boundary condition classes
class WallBC(DirichletBC):
    """No-slip wall boundary condition with optional gradient constraints."""
    def apply(self, x: tf.Tensor, y: tf.Tensor, values: Dict[str, Any], tape: Optional[tf.GradientTape] = None) -> Dict[str, tf.Tensor]:
        base_values = {
            'u': {'value': 0.0},
            'v': {'value': 0.0},
            'p': values.get('p', None)  # Pressure can have either value or gradient condition
        }
        # Merge with provided values
        base_values.update(values)
        return super().apply(x, y, base_values, tape)

class InletBC(DirichletBC):
    """Inlet boundary condition with optional gradient constraints."""
    def apply(self, x: tf.Tensor, y: tf.Tensor, values: Dict[str, Any], tape: Optional[tf.GradientTape] = None) -> Dict[str, tf.Tensor]:
        base_values = {
            'u': values.get('u', {'value': 1.0}),
            'v': values.get('v', {'value': 0.0}),
            'p': values.get('p', None)
        }
        return super().apply(x, y, base_values, tape)

class OutletBC(GradientBC):
    """Outlet boundary condition with gradient specifications."""
    def apply(self, x: tf.Tensor, y: tf.Tensor, values: Dict[str, Any], tape: Optional[tf.GradientTape] = None) -> Dict[str, tf.Tensor]:
        result = {}
        
        # Small coefficient to treat gradients as values
        grad_coeff = 1e-6
        
        # Process each variable separately
        for var_name, var_info in values.items():
            if var_info is None:
                continue
            
            if var_name in ['u', 'v']:
                # Apply zero gradient in x-direction
                gradient_value = var_info.get('gradient', 0.0)
                
                # Convert to tensor and ensure correct shape
                if not isinstance(gradient_value, tf.Tensor):
                    gradient_value = tf.convert_to_tensor(gradient_value, dtype=tf.float32)
                if gradient_value.shape != x.shape:
                    gradient_value = tf.zeros_like(x, dtype=tf.float32)
                
                # Treat gradient as a small value
                value = gradient_value * grad_coeff
                result[var_name] = {'value': value}
            
            elif var_name == 'p':
                # Use provided value or default to zero
                value = var_info.get('value', 0.0)
                
                # Convert to tensor and ensure correct shape
                if not isinstance(value, tf.Tensor):
                    value = tf.convert_to_tensor(value, dtype=tf.float32)
                if value.shape != x.shape:
                    value = tf.zeros_like(x, dtype=tf.float32)
                
                result[var_name] = {'value': value}
        
        return result

class MovingWallBC(DirichletBC):
    """Moving wall boundary condition with optional gradient constraints."""
    def apply(self, x: tf.Tensor, y: tf.Tensor, values: Dict[str, Any], tape: Optional[tf.GradientTape] = None) -> Dict[str, tf.Tensor]:
        base_values = {
            'u': values.get('u', {'value': 1.0}),
            'v': {'value': 0.0},
            'p': values.get('p', None)
        }
        return super().apply(x, y, base_values, tape)
