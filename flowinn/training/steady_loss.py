from flowinn.physics.steady_2D import SteadyNavierStokes2D
from flowinn.physics.steady_3D import SteadyNavierStokes3D
from flowinn.training.base_loss import NavierStokesBaseLoss
import tensorflow as tf

class SteadyNavierStokesLoss(NavierStokesBaseLoss):
    def __init__(self, mesh, model, Re: float = 1000.0, physics_model='NS2D', weights=[0.7, 0.3]) -> None:
        super().__init__(mesh, model, Re, weights)
        
        # Initialize physics model based on type
        if physics_model == 'NS2D':
            self._physics_loss = SteadyNavierStokes2D(Re)
        elif physics_model == 'NS3D':
            self._physics_loss = SteadyNavierStokes3D(Re)
        else:
            raise ValueError(f"Unknown physics model: {physics_model}")

    @property
    def physics_loss(self):
        return self._physics_loss

    def loss_function(self, batch_data=None):
        """Compute combined physics and boundary condition losses"""
        # Get coordinates based on dimension
        if batch_data is None:
            if isinstance(self._physics_loss, SteadyNavierStokes3D):
                coords = [
                    tf.reshape(tf.convert_to_tensor(getattr(self.mesh, coord), dtype=tf.float32), [-1, 1])
                    for coord in ['x', 'y', 'z']
                ]
            else:
                coords = [
                    tf.reshape(tf.convert_to_tensor(getattr(self.mesh, coord), dtype=tf.float32), [-1, 1])
                    for coord in ['x', 'y']
                ]
        else:
            # Fix: Use tf.split instead of Python iteration over tensor
            coords = tf.split(batch_data, num_or_size_splits=batch_data.shape[-1], axis=-1)

        total_loss = 0.0

        # Compute physics loss
        with tf.GradientTape(persistent=True) as tape:
            for coord in coords:
                tape.watch(coord)
            
            input_tensor = tf.concat(coords, axis=1)
            predictions = self.model.model(input_tensor)
            
            physics_loss = self.compute_physics_loss(predictions, coords, tape)
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

        # Store losses for ReLoBraLo
        self.physics_loss_history.append(float(physics_loss))
        self.boundary_loss_history.append(float(boundary_loss))
        
        # Update weights using ReLoBraLo
        self.update_weights()

        # Trim history if too long
        if len(self.physics_loss_history) > self.lookback_window * 2:
            self.physics_loss_history = self.physics_loss_history[-self.lookback_window:]
            self.boundary_loss_history = self.boundary_loss_history[-self.lookback_window:]

        return total_loss

    def compute_physics_loss(self, predictions, coords, tape):
        """Compute physics-based loss terms for flow equations."""
        is_3d = isinstance(self._physics_loss, SteadyNavierStokes3D)
        n_vel_components = 3 if is_3d else 2
        
        # Ensure predictions and coords are proper tensors
        predictions = tf.convert_to_tensor(predictions, dtype=tf.float32)
        coords = [tf.convert_to_tensor(c, dtype=tf.float32) for c in coords]
        
        # Split predictions into velocities and pressure
        velocities = tf.reshape(predictions[:, :n_vel_components], [-1, n_vel_components])
        pressure = tf.reshape(predictions[:, n_vel_components], [-1])
        
        # Use all coordinates for 3D, only x,y for 2D
        coords_to_use = coords if is_3d else coords[:2]
        
        # Watch tensors
        with tape.stop_recording():
            residuals = self._physics_loss.get_residuals(
                velocities, pressure, coords_to_use, tape
            )
        
        # Compute loss for each residual
        loss = 0.0
        for residual in residuals:
            if residual is not None:  # Add null check
                residual = tf.reshape(residual, [-1])
                loss += tf.reduce_mean(tf.square(residual))
            
        return loss

    def compute_boundary_loss(self, bc_results, vel_pred, p_pred, tape, coords):
        """Compute boundary condition losses."""
        loss = 0.0
        is_3d = isinstance(self._physics_loss, SteadyNavierStokes3D)
        n_vel_components = 3 if is_3d else 2
        
        for var_name, bc_info in bc_results.items():
            if bc_info is None:
                continue
                
            if 'value' in bc_info:
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
