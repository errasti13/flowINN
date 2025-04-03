from flowinn.physics.unsteady_2D import UnsteadyNavierStokes2D
from flowinn.training.base_loss import NavierStokesBaseLoss
import tensorflow as tf
import numpy as np

class UnsteadyNavierStokesLoss(NavierStokesBaseLoss):
    def __init__(self, mesh, model, Re: float = 1000.0, physics_model='NS2D', weights=[0.7, 0.3]) -> None:
        super().__init__(mesh, model, Re, weights)
        
        if physics_model == 'NS2D':
            self._physics_loss = UnsteadyNavierStokes2D(Re)
        else:
            raise ValueError(f"Unknown physics model: {physics_model}")

    @property
    def physics_loss(self):
        return self._physics_loss

    def __call__(self, batch_data=None):
        """
        Make the loss function callable directly.
        
        This enables the loss object to be used directly as a function:
        loss_obj(batch_data) instead of loss_obj.loss_function(batch_data)
        
        Args:
            batch_data: Batch of input data for the model
            
        Returns:
            Total loss value
        """
        return self.loss_function(batch_data)

    @tf.function
    def loss_function(self, batch_data=None):
        """Compute combined physics and boundary condition losses"""
        # Get coordinates based on dimension
        if batch_data is None:
            coords = [
                tf.reshape(tf.convert_to_tensor(getattr(self.mesh, coord), dtype=tf.float32), [-1, 1])
                for coord in ['x', 'y', 't']
            ]
        else:
            # Split batch_data into individual coordinate tensors
            x, y, t = tf.split(batch_data, num_or_size_splits=3, axis=-1)
            coords = [x, y, t]

        # Convert all inputs to tensors
        coords = [tf.convert_to_tensor(c, dtype=tf.float32) for c in coords]
        total_loss = 0.0

        # Compute physics loss with a single tape
        with tf.GradientTape(persistent=True) as tape:
            for coord in coords:
                tape.watch(coord)
            
            input_tensor = tf.concat(coords, axis=1)
            predictions = self.model.model(input_tensor, training=True)
            
            # Split predictions for physics loss
            velocities = predictions[:, :2]
            pressure = predictions[:, 2]
            
            # Get residuals outside tape context
            residuals = self._physics_loss.get_residuals(velocities, pressure, coords, tape)
            
            # Compute physics loss
            physics_loss = tf.reduce_mean([tf.reduce_mean(tf.square(r)) for r in residuals])
            total_loss += self.physicsWeight * physics_loss

        # Compute boundary losses with separate tape for efficiency
        boundary_loss = 0.0
        for boundary_name, boundary_data in self.mesh.boundaries.items():
            try:
                # Get boundary coordinates
                bc_coords = [
                    tf.reshape(tf.convert_to_tensor(boundary_data[coord], dtype=tf.float32), [-1, 1])
                    for coord in ['x', 'y', 't'] if coord in boundary_data
                ]

                bc_type = boundary_data['bc_type']
                conditions = boundary_data['conditions']

                # Apply boundary conditions with fresh tape
                with tf.GradientTape(persistent=True) as bc_tape:
                    for coord in bc_coords:
                        bc_tape.watch(coord)
                    
                    bc_input = tf.concat(bc_coords, axis=1)
                    bc_pred = self.model.model(bc_input, training=True)
                    
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
            interior_loss = self.compute_interior_loss()
            total_loss += self.boundaryWeight * interior_loss

        if hasattr(self.mesh, 'periodicBoundaries'):
            periodic_loss = self.compute_periodic_loss()
            total_loss += self.boundaryWeight * periodic_loss

        if hasattr(self.mesh, 'initialConditions'):
            initial_loss = self.compute_initial_loss()
            total_loss += self.boundaryWeight * initial_loss

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
    
    def compute_interior_loss(self):
        interior_loss = 0.0
        
        for boundary_name, boundary_data in self.mesh.interiorBoundaries.items():
            try:
                int_coords = [
                    tf.reshape(tf.convert_to_tensor(boundary_data[coord], dtype=tf.float32), [-1, 1])
                    for coord in ['x', 'y', 't'] if coord in boundary_data
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

        return interior_loss
    
    @tf.autograph.experimental.do_not_convert
    def compute_initial_loss(self):
        """Compute loss for initial conditions."""
        initial_loss = 0.0

        # Ensure initialConditions attribute exists
        if not hasattr(self.mesh, 'initialConditions') or not self.mesh.initialConditions:
            return 0.0 # Return zero loss if no initial conditions defined

        for ic_name, initial_data in self.mesh.initialConditions.items():
            try:
                # Check if all required coordinates exist
                has_all_coords = True
                for coord in ['x', 'y', 't']:
                    if coord not in initial_data:
                        print(f"Warning: Missing '{coord}' coordinate for initial condition '{ic_name}'. Skipping.")
                        has_all_coords = False
                        break
                
                if not has_all_coords:
                    continue
                
                # Handle mesh points with appropriate flattening
                x_data = initial_data['x']
                y_data = initial_data['y']
                t_data = initial_data['t']
                
                # Get shapes of each coordinate and verify consistency
                x_shape = np.asarray(x_data).shape
                y_shape = np.asarray(y_data).shape
                t_shape = np.asarray(t_data).shape
                
                # Check if shapes match
                if x_shape != y_shape or x_shape != t_shape:
                    print(f"Warning: Coordinate shape mismatch in initial condition '{ic_name}':")
                    print(f"  x shape: {x_shape}, y shape: {y_shape}, t shape: {t_shape}")
                    print(f"  Attempting to process anyway by flattening all arrays.")
                
                # Convert to tensors with proper flattening
                x_tensor = tf.reshape(tf.convert_to_tensor(x_data, dtype=tf.float32), [-1, 1])
                y_tensor = tf.reshape(tf.convert_to_tensor(y_data, dtype=tf.float32), [-1, 1])
                t_tensor = tf.reshape(tf.convert_to_tensor(t_data, dtype=tf.float32), [-1, 1])
                
                # Verify all tensors have the same first dimension after reshaping
                x_size = int(x_tensor.shape[0])
                y_size = int(y_tensor.shape[0])
                t_size = int(t_tensor.shape[0])
                
                # Handle dimension mismatches - Use numpy and Python conditionals instead of tensor ops
                if x_size != y_size or x_size != t_size:
                    print(f"Warning: After reshaping, point counts don't match in '{ic_name}':")
                    print(f"  x: {x_size}, y: {y_size}, t: {t_size}")
                    
                    # Try to make them match by repeating smaller tensors to match the largest
                    max_size = max(x_size, y_size, t_size)
                    
                    # Adjust x tensor if needed
                    if x_size == 1 and max_size > 1:
                        x_tensor = tf.repeat(x_tensor, max_size, axis=0)
                    elif x_size != max_size:
                        print(f"  Cannot automatically resize x tensor from {x_size} to {max_size}. Skipping.")
                        continue
                        
                    # Adjust y tensor if needed
                    if y_size == 1 and max_size > 1:
                        y_tensor = tf.repeat(y_tensor, max_size, axis=0)
                    elif y_size != max_size:
                        print(f"  Cannot automatically resize y tensor from {y_size} to {max_size}. Skipping.")
                        continue
                        
                    # Adjust t tensor if needed
                    if t_size == 1 and max_size > 1:
                        t_tensor = tf.repeat(t_tensor, max_size, axis=0)
                    elif t_size != max_size:
                        print(f"  Cannot automatically resize t tensor from {t_size} to {max_size}. Skipping.")
                        continue
                
                # Concatenate coordinate tensors
                input_tensor = tf.concat([x_tensor, y_tensor, t_tensor], axis=1)
                
                # Get model predictions
                predictions = self.model.model(input_tensor, training=True)
                
                # Extract predicted components
                velocities = predictions[:, :2]
                pressure = predictions[:, -1]
                
                pred_u = tf.reshape(velocities[:, 0], [-1])
                pred_v = tf.reshape(velocities[:, 1], [-1])
                pred_p = tf.reshape(pressure, [-1])
                
                # Apply initial conditions and compute loss
                conditions = initial_data.get('conditions', {})
                if not conditions:
                    print(f"Warning: No conditions specified for initial condition '{ic_name}'. Skipping.")
                    continue
                
                # Process each condition
                for var_name, pred_tensor in [('u', pred_u), ('v', pred_v), ('p', pred_p)]:
                    if var_name in conditions and conditions[var_name] is not None:
                        var_cond = conditions[var_name]
                        if 'value' in var_cond:
                            target_val = var_cond['value']
                            try:
                                # Convert target to tensor
                                target_tensor = tf.convert_to_tensor(target_val, dtype=tf.float32)
                                
                                # Handle scalar targets
                                if len(target_tensor.shape) == 0:  # Use len() instead of tf.rank
                                    pred_size = int(pred_tensor.shape[0])
                                    target_tensor = tf.fill([pred_size], target_tensor)
                                else:
                                    target_tensor = tf.reshape(target_tensor, [-1])
                                
                                # Check shapes match - Use Python operations on tensor shapes
                                pred_shape = pred_tensor.shape
                                target_shape = target_tensor.shape
                                
                                if pred_shape[0] == target_shape[0]:
                                    # Compute and add loss
                                    var_loss = tf.reduce_mean(tf.square(pred_tensor - target_tensor))
                                    initial_loss += var_loss
                                else:
                                    print(f"Warning: Shape mismatch for '{var_name}' in initial condition '{ic_name}':")
                                    print(f"  Prediction shape: {pred_shape}, Target shape: {target_shape}")
                            except Exception as e:
                                print(f"Warning: Error processing '{var_name}' in initial condition '{ic_name}': {str(e)}")
                                continue

            except Exception as e:
                print(f"Warning: Error processing initial condition '{ic_name}': {str(e)}")
                import traceback
                traceback.print_exc() # Print detailed traceback for debugging
                continue

        return initial_loss

    def compute_physics_loss(self, predictions, coords, tape):
        """Compute physics-based loss terms for flow equations."""
        n_vel_components = 2
        
        # Split predictions into velocities and pressure
        velocities = predictions[:, :n_vel_components]
        pressure = predictions[:, n_vel_components]
        
        residuals = self._physics_loss.get_residuals(
            velocities, pressure, coords, tape
        )
        
        # Compute loss for each residual
        loss = 0.0
        for residual in residuals:
            residual = tf.reshape(residual, [-1])
            loss += tf.reduce_mean(tf.square(residual))
            
        return loss

    def compute_boundary_loss(self, bc_results, vel_pred, p_pred, tape, coords):
        """Compute boundary condition losses."""
        loss = 0.0
        n_vel_components = 2
        
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
                        for coord in ['x', 'y', 't'] if coord in boundary_data
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
