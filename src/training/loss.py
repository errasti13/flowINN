from src.physics.steadyNS import NavierStokes2D, NavierStokes3D
import tensorflow as tf
from typing import Union, Optional

class NavierStokesLoss:
    def __init__(self, mesh, model, weights = [0.7, 0.3]) -> None:  # Adjusted weights for physics vs boundary
        self._mesh = mesh
        self._model = model
        self._physics_loss = NavierStokes2D() if mesh.is2D else NavierStokes3D()
        self._loss = None
        self._nu: float = 0.01

        # Update weights to include interior boundary weight
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

    def loss_function(self, batch_data=None):
        """Compute combined physics and boundary condition losses"""
        if self.mesh.is2D:
            return self.loss_function2D(batch_data)
        else:
            return self.loss_function3D(batch_data)

    def loss_function2D(self, batch_data=None):
        """Compute combined physics and boundary losses"""
        if batch_data is None:
            X = tf.reshape(tf.convert_to_tensor(self.mesh.x, dtype=tf.float32), [-1, 1])
            Y = tf.reshape(tf.convert_to_tensor(self.mesh.y, dtype=tf.float32), [-1, 1])
        else:
            X, Y = batch_data
            X = tf.reshape(X, [-1, 1])
            Y = tf.reshape(Y, [-1, 1])

        total_loss = 0.0

        # Compute physics loss
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([X, Y])
            uvp_pred = self.model.model(tf.concat([X, Y], axis=1))
            u_pred = uvp_pred[:, 0]
            v_pred = uvp_pred[:, 1]
            p_pred = uvp_pred[:, 2]

            physics_loss = self.compute_physics_loss(u_pred, v_pred, p_pred, X, Y, tape)
            total_loss += self.physicsWeight * physics_loss

        # Compute boundary losses
        boundary_loss = 0.0
        for boundary_name, boundary_data in self.mesh.boundaries.items():
            try:
                # Get boundary coordinates
                x_bc = tf.reshape(tf.convert_to_tensor(boundary_data['x'], dtype=tf.float32), [-1, 1])
                y_bc = tf.reshape(tf.convert_to_tensor(boundary_data['y'], dtype=tf.float32), [-1, 1])
                
                # Get boundary conditions object and values
                bc_type = boundary_data['bc_type']
                conditions = boundary_data['conditions']

                # Apply boundary conditions using the BC object
                with tf.GradientTape(persistent=True) as bc_tape:
                    bc_tape.watch([x_bc, y_bc])
                    uvp_bc = self.model.model(tf.concat([x_bc, y_bc], axis=1))
                    
                    # Get predicted values at boundary
                    u_bc = uvp_bc[:, 0]
                    v_bc = uvp_bc[:, 1]
                    p_bc = uvp_bc[:, 2]

                    # Apply boundary conditions and get constraints
                    bc_results = bc_type.apply(x_bc, y_bc, conditions, bc_tape)

                    # Compute losses for each variable
                    for var_name, bc_info in bc_results.items():
                        if bc_info is None:
                            continue
                            
                        if 'value' in bc_info:
                            # Handle Dirichlet condition
                            target_value = tf.cast(bc_info['value'], tf.float32)
                            if var_name == 'u':
                                boundary_loss += tf.reduce_mean(tf.square(u_bc - target_value))
                            elif var_name == 'v':
                                boundary_loss += tf.reduce_mean(tf.square(v_bc - target_value))
                            elif var_name == 'p':
                                boundary_loss += tf.reduce_mean(tf.square(p_bc - target_value))
                                
                        if 'gradient' in bc_info:
                            # Handle gradient condition
                            target_gradient = tf.cast(bc_info['gradient'], tf.float32)
                            direction = bc_info['direction']
                            
                            if direction == 'x':
                                grad = bc_tape.gradient(uvp_bc, x_bc)
                            elif direction == 'y':
                                grad = bc_tape.gradient(uvp_bc, y_bc)
                            elif isinstance(direction, tuple):
                                nx, ny = direction
                                grad_x = bc_tape.gradient(uvp_bc, x_bc)
                                grad_y = bc_tape.gradient(uvp_bc, y_bc)
                                grad = nx * grad_x + ny * grad_y
                                
                            if var_name == 'u':
                                boundary_loss += tf.reduce_mean(tf.square(grad[:, 0] - target_gradient))
                            elif var_name == 'v':
                                boundary_loss += tf.reduce_mean(tf.square(grad[:, 1] - target_gradient))
                            elif var_name == 'p':
                                boundary_loss += tf.reduce_mean(tf.square(grad[:, 2] - target_gradient))

            except Exception as e:
                print(f"Warning: Error processing boundary {boundary_name}: {str(e)}")
                continue

        total_loss += self.boundaryWeight * boundary_loss

        # Compute interior boundary losses with higher weight
        interior_loss = 0.0
        interior_weight = 2.0  # Higher weight for interior boundaries
        
        for boundary_name, boundary_data in self.mesh.interiorBoundaries.items():
            try:
                # Get boundary coordinates
                x_int = tf.reshape(tf.convert_to_tensor(boundary_data['x'], dtype=tf.float32), [-1, 1])
                y_int = tf.reshape(tf.convert_to_tensor(boundary_data['y'], dtype=tf.float32), [-1, 1])
                
                # Get boundary conditions object and values
                bc_type = boundary_data['bc_type']
                conditions = boundary_data['conditions']

                with tf.GradientTape(persistent=True) as int_tape:
                    int_tape.watch([x_int, y_int])
                    uvp_int = self.model.model(tf.concat([x_int, y_int], axis=1))
                    
                    u_int = uvp_int[:, 0]
                    v_int = uvp_int[:, 1]
                    p_int = uvp_int[:, 2]

                    # Apply interior boundary conditions
                    bc_results = bc_type.apply(x_int, y_int, conditions, int_tape)
                    
                    # Compute losses for each variable with higher weight
                    for var_name, bc_info in bc_results.items():
                        if bc_info is None:
                            continue
                            
                        if 'value' in bc_info:
                            target_value = tf.cast(bc_info['value'], tf.float32)
                            if var_name == 'u':
                                interior_loss += tf.reduce_mean(tf.square(u_int - target_value))
                            elif var_name == 'v':
                                interior_loss += tf.reduce_mean(tf.square(v_int - target_value))
                            elif var_name == 'p':
                                interior_loss += tf.reduce_mean(tf.square(p_int - target_value))

                        if 'gradient' in bc_info:
                            target_gradient = tf.cast(bc_info['gradient'], tf.float32)
                            direction = bc_info['direction']
                            
                            if direction == 'normal':
                                # Calculate normal direction based on boundary geometry
                                dx = int_tape.gradient(uvp_int, x_int)
                                dy = int_tape.gradient(uvp_int, y_int)
                                if var_name == 'p':
                                    interior_loss += tf.reduce_mean(tf.square(dx[:, 2] - target_gradient))
                                    interior_loss += tf.reduce_mean(tf.square(dy[:, 2] - target_gradient))

            except Exception as e:
                print(f"Warning: Error processing interior boundary {boundary_name}: {str(e)}")
                continue

        # Add interior boundary loss with higher weight
        total_loss += interior_weight * interior_loss

        return total_loss

    def loss_function3D(self, batch_data=None):
        """3D version of the loss function."""
        if batch_data is None:
            X = tf.reshape(tf.convert_to_tensor(self.mesh.x, dtype=tf.float32), [-1, 1])
            Y = tf.reshape(tf.convert_to_tensor(self.mesh.y, dtype=tf.float32), [-1, 1])
            Z = tf.reshape(tf.convert_to_tensor(self.mesh.z, dtype=tf.float32), [-1, 1])
        else:
            X, Y, Z = batch_data
            X = tf.reshape(X, [-1, 1])
            Y = tf.reshape(Y, [-1, 1])
            Z = tf.reshape(Z, [-1, 1])

        total_loss = 0.0

        # Compute physics loss
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([X, Y, Z])
            uvwp_pred = self.model.model(tf.concat([X, Y, Z], axis=1))
            u_pred = uvwp_pred[:, 0]
            v_pred = uvwp_pred[:, 1]
            w_pred = uvwp_pred[:, 2]
            p_pred = uvwp_pred[:, 3]

            physics_loss = self.compute_physics_loss_3d(u_pred, v_pred, w_pred, p_pred, X, Y, Z, tape)
            total_loss += self.physicsWeight * physics_loss

        # Compute exterior boundary losses
        boundary_loss = 0.0
        for boundary_name, boundary_data in self.mesh.boundaries.items():
            try:
                xBc = self.convert_and_reshape(boundary_data['x'])
                yBc = self.convert_and_reshape(boundary_data['y'])
                zBc = self.convert_and_reshape(boundary_data['z'])
                uBc = boundary_data.get('u')
                vBc = boundary_data.get('v')
                wBc = boundary_data.get('w')
                pBc = boundary_data.get('p')

                boundary_loss += self.computeBoundaryLoss3D(
                    self.model.model, xBc, yBc, zBc, uBc, vBc, wBc, pBc
                )
            except Exception as e:
                print(f"Warning: Error processing boundary {boundary_name}: {str(e)}")
                continue

        total_loss += self.boundaryWeight * boundary_loss
        return total_loss

    def compute_physics_loss(self, u_pred, v_pred, p_pred, X, Y, tape):
        """Compute physics-based loss terms for Navier-Stokes equations.
    
        Args:
            u_pred: Predicted x-velocity component
            v_pred: Predicted y-velocity component
            p_pred: Predicted pressure
            X: X coordinates tensor            Y: Y coordinates tensor
            tape: GradientTape instance for automatic differentiation
            
        Returns:
            float: Combined physics loss from continuity and momentum equations
        """
            
        continuity, momentum_u, momentum_v = self._physics_loss.get_residuals(u_pred, v_pred, p_pred, X, Y, tape)

        f_loss_u = tf.reduce_mean(tf.square(momentum_u))
        f_loss_v = tf.reduce_mean(tf.square(momentum_v))
        continuity_loss = tf.reduce_mean(tf.square(continuity))

        return f_loss_u + f_loss_v + continuity_loss

    def compute_physics_loss_3d(self, u_pred, v_pred, w_pred, p_pred, X, Y, Z, tape):
        """Compute physics-based loss terms for 3D Navier-Stokes equations."""
        continuity, momentum_u, momentum_v, momentum_w = self._physics_loss.get_residuals(
            u_pred, v_pred, w_pred, p_pred, X, Y, Z, tape
        )

        f_loss_u = tf.reduce_mean(tf.square(momentum_u))
        f_loss_v = tf.reduce_mean(tf.square(momentum_v))
        f_loss_w = tf.reduce_mean(tf.square(momentum_w))
        continuity_loss = tf.reduce_mean(tf.square(continuity))

        return f_loss_u + f_loss_v + f_loss_w + continuity_loss
         
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

    def computeBoundaryLoss3D(self, model, xBc, yBc, zBc, uBc, vBc, wBc, pBc):
        """Compute boundary loss for 3D case."""
        def compute_component_loss(bc, idx):
            if bc is not None:
                pred = model(tf.concat([
                    tf.cast(xBc, dtype=tf.float32),
                    tf.cast(yBc, dtype=tf.float32),
                    tf.cast(zBc, dtype=tf.float32)
                ], axis=1))[:, idx]
                return tf.reduce_mean(tf.square(pred - tf.cast(bc, dtype=tf.float32)))
            return tf.constant(0.0)

        u_loss = compute_component_loss(uBc, 0)
        v_loss = compute_component_loss(vBc, 1)
        w_loss = compute_component_loss(wBc, 2)
        p_loss = compute_component_loss(pBc, 3)

        return u_loss + v_loss + w_loss + p_loss