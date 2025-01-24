from src.physics.steadyNS import NavierStokes2D, NavierStokes3D
import tensorflow as tf
from typing import Union, Optional

class NavierStokesLoss:
    def __init__(self, mesh, model, weights = [0.8, 0.2]) -> None:
        self._mesh = mesh
        self._model = model
        self._physics_loss = NavierStokes2D() if mesh.is2D else NavierStokes3D()
        self._loss = None
        self._nu: float = 0.01

        self.physicsWeight  = weights[0]
        self.boundaryWeight = weights[1]

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        if not hasattr(value, 'is2D'):
            raise ValueError("Mesh must have is2D attribute")
        self._mesh = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def physics_loss(self):
        return self._physics_loss

    @physics_loss.setter
    def physics_loss(self, value):
        if not isinstance(value, (NavierStokes2D, NavierStokes3D)):
            raise TypeError("physics_loss must be NavierStokes2D or NavierStokes3D")
        self._physics_loss = value

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, value):
        self._loss = value

    @property
    def nu(self) -> float:
        return self._nu

    @nu.setter
    def nu(self, value: float):
        if not isinstance(value, (int, float)):
            raise TypeError("nu must be a number")
        if value <= 0:
            raise ValueError("nu must be positive")
        self._nu = float(value)

    def loss_function(self):
        if self.mesh.is2D:
            return self.loss_function2D()
        else:
            raise NotImplementedError("Only 2D loss functions are implemented for now.")

    def loss_function2D(self):
        X = tf.reshape(tf.convert_to_tensor(self.mesh.x, dtype=tf.float32), [-1, 1])
        Y = tf.reshape(tf.convert_to_tensor(self.mesh.y, dtype=tf.float32), [-1, 1])

        total_loss = 0

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([X, Y])

            # Compute predictions
            uvp_pred = self.model.model(tf.concat([X, Y], axis=1))
            u_pred = uvp_pred[:, 0]
            v_pred = uvp_pred[:, 1]
            p_pred = uvp_pred[:, 2]

            total_loss += self.physicsWeight * self.compute_physics_loss(u_pred, v_pred, p_pred, X, Y, tape)

        # Compute boundary condition losses
        for boundary_key, boundary_data in self.mesh.boundaries.items():
            xBc = boundary_data['x']
            yBc = boundary_data['y']
            uBc = boundary_data['u']
            vBc = boundary_data['v']
            pBc = boundary_data['p']

            xBc = self.convert_and_reshape(xBc)
            yBc = self.convert_and_reshape(yBc)
            uBc_tensor, vBc_tensor, pBc_tensor = self.imposeBoundaryCondition(uBc, vBc, pBc)

            # Compute boundary losses for each condition
            uBc_loss, vBc_loss, pBc_loss = self.computeBoundaryLoss(self.model.model, xBc, yBc, uBc_tensor, vBc_tensor, pBc_tensor)
            
            total_loss += self.boundaryWeight * (uBc_loss + vBc_loss + pBc_loss)

        return total_loss
    
    def compute_physics_loss(self, u_pred, v_pred, p_pred, X, Y, tape):
        """Compute physics-based loss terms for Navier-Stokes equations.
    
        Args:
            u_pred: Predicted x-velocity component
            v_pred: Predicted y-velocity component
            p_pred: Predicted pressure
            X: X coordinates tensor
            Y: Y coordinates tensor
            tape: GradientTape instance for automatic differentiation
            
        Returns:
            float: Combined physics loss from continuity and momentum equations
        """
            
        continuity, momentum_u, momentum_v = self._physics_loss.get_residuals(u_pred, v_pred, p_pred, X, Y, tape)

        f_loss_u = tf.reduce_mean(tf.square(momentum_u))
        f_loss_v = tf.reduce_mean(tf.square(momentum_v))
        continuity_loss = tf.reduce_mean(tf.square(continuity))

        return f_loss_u + f_loss_v + continuity_loss
         
    def convert_and_reshape(self, tensor, dtype=tf.float32, shape=(-1, 1)):
                        if tensor is not None:
                            return tf.reshape(tf.convert_to_tensor(tensor, dtype=dtype), shape)
                        return None
       
    def imposeBoundaryCondition(self, uBc, vBc, pBc):
        def convert_if_not_none(tensor):
            return tf.convert_to_tensor(tensor, dtype=tf.float32) if tensor is not None else None

        uBc = convert_if_not_none(uBc)
        vBc = convert_if_not_none(vBc)
        pBc = convert_if_not_none(pBc)

        return uBc, vBc, pBc
    
    def computeBoundaryLoss(self, model, xBc, yBc, uBc, vBc, pBc):
        def compute_loss(bc, idx):
            if bc is not None:
                pred = model(tf.concat([tf.cast(xBc, dtype=tf.float32), tf.cast(yBc, dtype=tf.float32)], axis=1))[:, idx]
                return tf.reduce_mean(tf.square(pred - bc))
            else:
                return tf.constant(0.0)

        uBc_loss = compute_loss(uBc, 0)
        vBc_loss = compute_loss(vBc, 1)
        pBc_loss = compute_loss(pBc, 2)

        return uBc_loss, vBc_loss, pBc_loss