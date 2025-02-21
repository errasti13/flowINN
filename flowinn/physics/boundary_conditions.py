import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union

class BoundaryCondition(ABC):
    """Base class for all boundary conditions."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def apply(self, coords: List[tf.Tensor], values: Dict[str, Any], tape: tf.GradientTape) -> Dict[str, Dict[str, Any]]:
        """
        Apply the boundary condition.
        
        Args:
            coords: List of coordinate tensors [x, y] or [x, y, z]
            values: Dictionary of boundary values and conditions
            tape: GradientTape for automatic differentiation
            
        Returns:
            Dictionary of variable names to their boundary conditions
        """
        pass

class GradientBC(BoundaryCondition):
    """Boundary condition for gradients of variables."""
    def apply(self, coords: List[tf.Tensor], values: Dict[str, Any], tape: tf.GradientTape) -> Dict[str, Dict[str, Any]]:
        result = {}
        n_dims = len(coords)  # Get number of dimensions
        
        for var_name, grad_info in values.items():
            if grad_info is None:
                result[var_name] = None
                continue

            if isinstance(grad_info, dict):
                direction = grad_info.get('direction', 'normal')
                value = grad_info.get('value', 0.0)
                
                if direction == 'normal':
                    # Get normal direction components for actual dimensions
                    normal_components = [
                        grad_info.get(f'n{dim}', 0.0)
                        for dim in ['x', 'y', 'z'][:n_dims]
                    ]
                    direction = tuple(normal_components)
                
                result[var_name] = {
                    'gradient': tf.cast(value, tf.float32),
                    'direction': direction
                }
            
        return result

class DirichletBC(BoundaryCondition):
    """Dirichlet boundary condition."""
    def apply(self, coords: List[tf.Tensor], values: Dict[str, Any], tape: tf.GradientTape) -> Dict[str, Dict[str, Any]]:
        result = {}
        n_dims = len(coords)
        
        for var_name, var_info in values.items():
            if var_info is None:
                result[var_name] = None
                continue

            # Skip 'w' component in 2D
            if var_name == 'w' and n_dims == 2:
                continue

            if isinstance(var_info, dict):
                if 'value' in var_info:
                    result[var_name] = {'value': tf.cast(var_info['value'], tf.float32)}
                if 'gradient' in var_info:
                    direction = var_info.get('direction', 'normal')
                    if direction == 'normal':
                        normal_components = [
                            var_info.get(f'n{dim}', 0.0)
                            for dim in ['x', 'y', 'z'][:n_dims]
                        ]
                        direction = tuple(normal_components)
                    result[var_name] = {
                        'gradient': tf.cast(var_info['gradient'], tf.float32),
                        'direction': direction
                    }
            else:
                result[var_name] = {'value': tf.cast(var_info, tf.float32)}
                
        return result

class WallBC(DirichletBC):
    """No-slip wall boundary condition."""
    def apply(self, coords: List[tf.Tensor], values: Dict[str, Any], tape: tf.GradientTape) -> Dict[str, Dict[str, Any]]:
        base_values = {
            'u': {'value': 0.0},
            'v': {'value': 0.0}
        }
        if len(coords) > 2:  # 3D case
            base_values['w'] = {'value': 0.0}
            
        if 'p' in values:
            base_values['p'] = values['p']
            
        # Override with any provided values
        base_values.update(values)
        return super().apply(coords, base_values, tape)

class InletBC(DirichletBC):
    """Inlet boundary condition."""
    def apply(self, coords: List[tf.Tensor], values: Dict[str, Any], tape: tf.GradientTape) -> Dict[str, Dict[str, Any]]:
        base_values = {
            'u': {'value': 1.0},
            'v': {'value': 0.0}
        }
        if len(coords) > 2:  # 3D case
            base_values['w'] = {'value': 0.0}
            
        # Override with provided values
        base_values.update(values)
        return super().apply(coords, base_values, tape)

class OutletBC(GradientBC):
    """Outlet boundary condition."""
    def apply(self, coords: List[tf.Tensor], values: Dict[str, Any], tape: tf.GradientTape) -> Dict[str, Dict[str, Any]]:
        base_values = {
            'u': {'gradient': 0.0, 'direction': 'x'},
            'v': {'gradient': 0.0, 'direction': 'x'}
        }
        if len(coords) > 2:  # 3D case
            base_values['w'] = {'gradient': 0.0, 'direction': 'x'}
            
        if 'p' in values:
            base_values['p'] = {'value': 0.0}  # Default pressure value at outlet
            
        # Override with provided values
        base_values.update(values)
        return super().apply(coords, base_values, tape)

class MovingWallBC(DirichletBC):
    """Moving wall boundary condition."""
    def apply(self, coords: List[tf.Tensor], values: Dict[str, Any], tape: tf.GradientTape) -> Dict[str, Dict[str, Any]]:
        base_values = {
            'u': {'value': 1.0},
            'v': {'value': 0.0}
        }
        if len(coords) > 2:  # 3D case
            base_values['w'] = {'value': 0.0}
            
        if 'p' in values:
            base_values['p'] = values['p']
            
        # Override with provided values
        base_values.update(values)
        return super().apply(coords, base_values, tape)

class PeriodicBC(BoundaryCondition):
    """Periodic boundary condition."""
    def apply(self, coords: List[tf.Tensor], values: Dict[str, Any], tape: tf.GradientTape) -> Dict[str, Dict[str, Any]]:
        """
        Apply periodic boundary conditions ensuring continuity between coupled boundaries.
        
        Args:
            coords: List of coordinate tensors [x, y] or [x, y, z]
            values: Dictionary containing boundary values and coupled boundary information
            tape: GradientTape for automatic differentiation
            
        Returns:
            Dictionary containing periodic boundary information
        """
        result = {}
        n_dims = len(coords)

        # Define base variables that should be periodic
        variables = ['u', 'v', 'p']
        if n_dims > 2:
            variables.append('w')

        for var_name in variables:
            result[var_name] = {
                'periodic': True,
                'value_match': True,      # Match values across periodic boundary
                'gradient_match': True    # Match gradients across periodic boundary
            }

        # Override with any provided values
        for var_name, var_info in values.items():
            if isinstance(var_info, dict):
                result[var_name].update(var_info)

        return result
