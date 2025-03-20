import tensorflow as tf
from typing import Tuple, List
from abc import ABC, abstractmethod

class NavierStokes(ABC):
    """Base class for Navier-Stokes equations."""
    
    def _compute_first_derivatives(self, variables: list, coords: list, tape) -> list:
        """Compute first-order derivatives for each variable with respect to each coordinate."""
        derivatives = []
        for var in variables:
            var = tf.convert_to_tensor(var, dtype=tf.float32)
            var = tf.reshape(var, [-1])  # Ensure 1D tensor
            for coord in coords:
                coord = tf.convert_to_tensor(coord, dtype=tf.float32)
                coord = tf.reshape(coord, [-1, 1])  # Ensure 2D tensor
                tape.watch(coord)
                grad = tape.gradient(var, coord)
                # Replace None gradients with zeros
                if grad is None:
                    grad = tf.zeros_like(var)
                derivatives.append(tf.reshape(grad, [-1]))
        return derivatives

    def _compute_second_derivatives(self, first_derivatives: list, coordinates: list, tape) -> list:
        """Compute second-order derivatives."""
        derivatives = []
        for d in first_derivatives:
            if d is not None:
                d = tf.convert_to_tensor(d, dtype=tf.float32)
                d = tf.reshape(d, [-1])  # Ensure 1D tensor
                for coord in coordinates:
                    coord = tf.convert_to_tensor(coord, dtype=tf.float32)
                    coord = tf.reshape(coord, [-1, 1])  # Ensure 2D tensor
                    tape.watch(coord)
                    grad = tape.gradient(d, coord)
                    # Replace None gradients with zeros
                    if grad is None:
                        grad = tf.zeros_like(d)
                    derivatives.append(tf.reshape(grad, [-1]))
            else:
                # If first derivative was None, add zeros for second derivative
                derivatives.append(tf.zeros([tf.shape(coordinates[0])[0]]))
        return derivatives

    @abstractmethod
    def get_residuals(self, *args, **kwargs) -> Tuple[tf.Tensor, ...]:
        """Calculate Navier-Stokes residuals."""
        pass
