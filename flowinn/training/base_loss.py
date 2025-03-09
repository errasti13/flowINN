import tensorflow as tf
from abc import ABC, abstractmethod

class NavierStokesBaseLoss(ABC):
    def __init__(self, mesh, model, Re: float = 1000.0, weights=[0.7, 0.3]) -> None:
        self._mesh = mesh
        self._model = model
        self._Re = Re
        
        self.physicsWeight = weights[0]
        self.boundaryWeight = weights[1]
        
        # ReLoBraLo parameters
        self.physics_loss_history = []
        self.boundary_loss_history = []
        self.lookback_window = 50
        self.alpha = 0.1
        self.min_weight = 0.1
        self.max_weight = 0.9

    @property
    def mesh(self):
        return self._mesh
    
    @property
    def model(self):
        return self._model
    
    @property
    def Re(self):
        return self._Re
    
    @abstractmethod
    def compute_physics_loss(self, predictions, coords, tape):
        pass

    @abstractmethod
    def compute_boundary_loss(self, bc_results, vel_pred, p_pred, tape, coords):
        pass

    @abstractmethod
    def loss_function(self, batch_data=None):
        """Abstract method for loss function implementation"""
        pass

    def update_weights(self):
        """Update weights using ReLoBraLo algorithm"""
        if len(self.physics_loss_history) < self.lookback_window:
            return

        lookback = tf.random.uniform([], minval=1, maxval=min(self.lookback_window, len(self.physics_loss_history)), dtype=tf.int32)
        physics_change = (self.physics_loss_history[-1] / (tf.reduce_mean(self.physics_loss_history[-lookback:]) + 1e-10))
        boundary_change = (self.boundary_loss_history[-1] / (tf.reduce_mean(self.boundary_loss_history[-lookback:]) + 1e-10))
        
        weight_update = self.alpha * (physics_change - boundary_change)
        self.physicsWeight = tf.clip_by_value(self.physicsWeight - weight_update, self.min_weight, self.max_weight)
        self.boundaryWeight = 1.0 - self.physicsWeight

    def convert_and_reshape(self, tensor, dtype=tf.float32, shape=(-1, 1)):
        if tensor is not None:
            return tf.reshape(tf.convert_to_tensor(tensor, dtype=dtype), shape)
        return None
