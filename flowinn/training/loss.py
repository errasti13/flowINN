from flowinn.physics.steadyNS import NavierStokes2D, NavierStokes3D
import tensorflow as tf

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
        # Get coordinates based on dimension
        if batch_data is None:
            coords = [
                tf.reshape(tf.convert_to_tensor(getattr(self.mesh, coord), dtype=tf.float32), [-1, 1])
                for coord in ['x', 'y', 'z'] if hasattr(self.mesh, coord)
            ]
        else:
            coords = [tf.reshape(x, [-1, 1]) for x in batch_data]

        total_loss = 0.0

        # Compute physics loss
        with tf.GradientTape(persistent=True) as tape:
            for coord in coords:
                tape.watch(coord)
            
            input_tensor = tf.concat(coords, axis=1)
            predictions = self.model.model(input_tensor)
            
            # Extract velocity components and pressure
            velocities = predictions[:, :-1]  # All but last column
            pressure = predictions[:, -1]     # Last column

            physics_loss = self.compute_physics_loss(velocities, pressure, coords, tape)
            total_loss += self.physicsWeight * physics_loss

        # Compute boundary losses
        boundary_loss = 0.0
        for boundary_name, boundary_data in self.mesh.boundaries.items():
            try:
                # Get boundary coordinates
                bc_coords = [
                    tf.reshape(tf.convert_to_tensor(boundary_data[coord], dtype=tf.float32), [-1, 1])
                    for coord in ['x', 'y', 'z'] if coord in boundary_data
                ]
                
                bc_type = boundary_data['bc_type']
                conditions = boundary_data['conditions']

                # Apply boundary conditions
                with tf.GradientTape(persistent=True) as bc_tape:
                    for coord in bc_coords:
                        bc_tape.watch(coord)
                    
                    bc_input = tf.concat(bc_coords, axis=1)
                    bc_pred = self.model.model(bc_input)
                    
                    # Split predictions into velocities and pressure
                    vel_pred = bc_pred[:, :-1]
                    p_pred = bc_pred[:, -1]

                    # Apply boundary conditions and compute loss
                    bc_results = bc_type.apply(bc_coords, conditions, bc_tape)
                    boundary_loss += self.compute_boundary_loss(bc_results, vel_pred, p_pred, bc_tape, bc_coords)

            except Exception as e:
                print(f"Warning: Error processing boundary {boundary_name}: {str(e)}")
                continue

        total_loss += self.boundaryWeight * boundary_loss

        # Handle interior boundaries if they exist
        if hasattr(self.mesh, 'interiorBoundaries'):
            interior_loss = 0.0
            
            for boundary_name, boundary_data in self.mesh.interiorBoundaries.items():
                try:
                    int_coords = [
                        tf.reshape(tf.convert_to_tensor(boundary_data[coord], dtype=tf.float32), [-1, 1])
                        for coord in ['x', 'y', 'z'] if coord in boundary_data
                    ]
                    
                    bc_type = boundary_data['bc_type']
                    conditions = boundary_data['conditions']

                    with tf.GradientTape(persistent=True) as int_tape:
                        for coord in int_coords:
                            int_tape.watch(coord)
                        
                        int_pred = self.model.model(tf.concat(int_coords, axis=1))
                        vel_pred = int_pred[:, :-1]
                        p_pred = int_pred[:, -1]

                        bc_results = bc_type.apply(int_coords, conditions, int_tape)
                        interior_loss += self.compute_boundary_loss(bc_results, vel_pred, p_pred, int_tape, int_coords)

                except Exception as e:
                    print(f"Warning: Error processing interior boundary {boundary_name}: {str(e)}")
                    continue

            total_loss += self.boundaryWeight * interior_loss

        if hasattr(self.mesh, 'periodicBoundaries'):
            periodic_loss = self.compute_periodic_loss()
            total_loss += self.boundaryWeight * periodic_loss

        return total_loss

    def compute_physics_loss(self, velocities, pressure, coords, tape):
        """Compute physics-based loss terms for Navier-Stokes equations.
        
        Args:
            velocities: Tensor of velocity components (u,v) or (u,v,w)
            pressure: Pressure tensor
            coords: List of coordinate tensors [X,Y] or [X,Y,Z]
            tape: GradientTape instance
            
        Returns:
            float: Combined physics loss from continuity and momentum equations
        """
        # Get residuals from physics model
        residuals = self._physics_loss.get_residuals(velocities, pressure, coords, tape)
        
        # Compute loss for each residual separately and then combine
        loss = 0.0
        for residual in residuals:
            # Ensure the residual is properly shaped
            residual = tf.reshape(residual, [-1])
            loss += tf.reduce_mean(tf.square(residual))
            
        return loss

    def compute_boundary_loss(self, bc_results, vel_pred, p_pred, tape, coords):
        """Compute boundary condition losses."""
        loss = 0.0
        n_vel_components = vel_pred.shape[-1]  # Get number of velocity components
        
        for var_name, bc_info in bc_results.items():
            if bc_info is None:
                continue
                
            if 'value' in bc_info:
                # Handle Dirichlet condition
                target_value = tf.cast(bc_info['value'], tf.float32)
                if var_name == 'p':
                    loss += tf.reduce_mean(tf.square(p_pred - target_value))
                else:
                    # Handle velocity components based on dimension
                    component_idx = {'u': 0, 'v': 1, 'w': 2}.get(var_name)
                    if component_idx is not None and component_idx < n_vel_components:
                        loss += tf.reduce_mean(tf.square(vel_pred[:, component_idx] - target_value))
                        
            if 'gradient' in bc_info:
                loss += self.compute_gradient_loss(bc_info, vel_pred, p_pred, tape, coords, var_name, n_vel_components)
                
        return loss

    def compute_gradient_loss(self, bc_info, vel_pred, p_pred, tape, coords, var_name, n_vel_components):
        """Compute gradient-based boundary condition losses."""
        target_gradient = tf.cast(bc_info['gradient'], tf.float32)
        direction = bc_info['direction']
        loss = 0.0
        
        if isinstance(direction, tuple):
            # Handle normal direction gradients
            if var_name == 'p':
                var_tensor = tf.reshape(p_pred, [-1, 1])
            else:
                component_idx = {'u': 0, 'v': 1, 'w': 2}.get(var_name)
                if component_idx is None or component_idx >= n_vel_components:
                    return 0.0
                var_tensor = tf.reshape(vel_pred[:, component_idx], [-1, 1])
            
            # Compute gradients for each coordinate
            grads = []
            for coord, normal_comp in zip(coords, direction[:len(coords)]):
                if normal_comp != 0:
                    grad = tape.gradient(var_tensor, coord)
                    if grad is not None:
                        grads.append(normal_comp * grad)
            
            if grads:
                normal_grad = tf.add_n(grads)
                loss += tf.reduce_mean(tf.square(normal_grad - target_gradient))
                
        else:
            # Handle single direction gradients
            coord_idx = {'x': 0, 'y': 1, 'z': 2}.get(direction)
            if coord_idx is not None and coord_idx < len(coords):
                if var_name == 'p':
                    var_tensor = tf.reshape(p_pred, [-1, 1])
                else:
                    component_idx = {'u': 0, 'v': 1, 'w': 2}.get(var_name)
                    if component_idx is None or component_idx >= n_vel_components:
                        return 0.0
                    var_tensor = tf.reshape(vel_pred[:, component_idx], [-1, 1])
                
                grad = tape.gradient(var_tensor, coords[coord_idx])
                if grad is not None:
                    loss += tf.reduce_mean(tf.square(grad - target_gradient))
                        
        return loss

    def compute_periodic_loss(self):
        """Compute loss for periodic boundary conditions."""
        periodic_loss = 0.0

        for boundary_name, boundary_data in self.mesh.periodicBoundaries.items():
            try:
                coupled_boundary = boundary_data['coupled_boundary']
                coupled_data = self.mesh.boundaries.get(coupled_boundary)

                if coupled_data is None:
                    print(f"Warning: Coupled boundary {coupled_boundary} not found for periodic boundary {boundary_name}")
                    continue

                # Get coordinates and set up gradients
                with tf.GradientTape(persistent=True) as tape:
                    # Get coordinates for base boundary
                    base_coords = [
                        tf.convert_to_tensor(boundary_data[coord], dtype=tf.float32)
                        for coord in ['x', 'y', 'z'] if coord in boundary_data
                    ]
                    base_coords = [tf.reshape(coord, [-1, 1]) for coord in base_coords]
                    
                    # Get coordinates for coupled boundary
                    coupled_coords = [
                        tf.convert_to_tensor(coupled_data[coord], dtype=tf.float32)
                        for coord in ['x', 'y', 'z'] if coord in coupled_data
                    ]
                    coupled_coords = [tf.reshape(coord, [-1, 1]) for coord in coupled_coords]

                    # Watch coordinates for gradient computation
                    for coord in base_coords + coupled_coords:
                        tape.watch(coord)

                    # Get predictions for both boundaries
                    base_input = tf.concat(base_coords, axis=1)
                    coupled_input = tf.concat(coupled_coords, axis=1)
                    
                    base_pred = self.model.model(base_input)
                    coupled_pred = self.model.model(coupled_input)

                    # Match values
                    value_loss = tf.reduce_mean(tf.square(base_pred - coupled_pred))
                    periodic_loss += value_loss

                    # Match gradients
                    base_grads = [tape.gradient(base_pred, coord) for coord in base_coords]
                    coupled_grads = [tape.gradient(coupled_pred, coord) for coord in coupled_coords]

                    for base_grad, coupled_grad in zip(base_grads, coupled_grads):
                        if base_grad is not None and coupled_grad is not None:
                            gradient_loss = tf.reduce_mean(tf.square(base_grad - coupled_grad))
                            periodic_loss += gradient_loss

            except Exception as e:
                print(f"Warning: Error processing periodic boundary {boundary_name}: {str(e)}")
                continue

        return periodic_loss

    def convert_and_reshape(self, tensor, dtype=tf.float32, shape=(-1, 1)):
        if tensor is not None:
            return tf.reshape(tf.convert_to_tensor(tensor, dtype=dtype), shape)
        return None