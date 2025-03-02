import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import tensorflow as tf
import os
from typing import Dict, Any, List, Tuple, Optional
from flowinn.physics.boundary_conditions import InletBC, OutletBC, WallBC

class BoundaryVisualization:
    """
    Class for visualizing boundary condition errors during training.
    
    This class provides methods to evaluate and visualize how well boundary conditions
    are being satisfied during the training of physics-informed neural networks.
    """
    
    def __init__(self, output_dir: str = 'bc_plots'):
        """
        Initialize the boundary visualization class.
        
        Args:
            output_dir: Directory where boundary condition plots will be saved
        """
        self.output_dir = output_dir
        self.detailed_dir = os.path.join(output_dir, 'detailed')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.detailed_dir, exist_ok=True)
        
        # Dictionary to track boundary condition errors over time
        self.bc_error_history = {}
    
    def evaluate_boundary_errors(self, model, mesh, epoch: int) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate boundary condition errors for all boundaries (exterior and interior).
        
        Args:
            model: The neural network model
            mesh: The mesh object containing boundary information
            epoch: Current training epoch
            
        Returns:
            Dictionary containing boundary error data
        """
        # Dictionary to store boundary condition errors and coordinates
        bc_errors = {}
        
        # Process exterior boundaries
        bc_errors.update(self._evaluate_boundary_set(model, mesh.boundaries, "ext", epoch))
        
        # Process interior boundaries if they exist
        if hasattr(mesh, 'interiorBoundaries') and mesh.interiorBoundaries:
            bc_errors.update(self._evaluate_boundary_set(model, mesh.interiorBoundaries, "int", epoch))
            
        return bc_errors
    
    def _evaluate_boundary_set(self, model, boundaries, prefix: str, epoch: int) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate boundary condition errors for a set of boundaries.
        
        Args:
            model: The neural network model
            boundaries: Dictionary of boundaries to evaluate
            prefix: Prefix to add to boundary names (e.g., "ext" or "int")
            epoch: Current training epoch
            
        Returns:
            Dictionary containing boundary error data
        """
        bc_errors = {}
        
        # Evaluate boundary conditions for each boundary
        for boundary_name, boundary_data in boundaries.items():
            # Get boundary coordinates
            bc_coords = [
                tf.reshape(tf.convert_to_tensor(boundary_data[coord], dtype=tf.float32), [-1, 1])
                for coord in ['x', 'y'] if coord in boundary_data
            ]
            
            if not bc_coords:
                continue
                
            bc_type = boundary_data['bc_type']
            conditions = boundary_data['conditions']
            
            # Convert coordinates to numpy for plotting
            x_coords = bc_coords[0].numpy().flatten()
            y_coords = bc_coords[1].numpy().flatten()
            
            # Predict values at boundary points
            bc_input = tf.concat(bc_coords, axis=1)
            bc_pred = model.model(bc_input).numpy()
            
            # Split predictions into velocities and pressure
            vel_pred = bc_pred[:, :-1]
            p_pred = bc_pred[:, -1]
            
            # Calculate errors based on boundary type
            if isinstance(bc_type, WallBC):
                # For wall BC, velocity should be zero
                # Check if the conditions specify a non-zero value
                if 'u' in conditions and isinstance(conditions['u'], dict) and 'value' in conditions['u']:
                    target_u = conditions['u']['value']
                else:
                    target_u = 0.0  # Default for wall BC
                
                if 'v' in conditions and isinstance(conditions['v'], dict) and 'value' in conditions['v']:
                    target_v = conditions['v']['value']
                else:
                    target_v = 0.0  # Default for wall BC
                
                u_error = np.abs(vel_pred[:, 0] - target_u)
                v_error = np.abs(vel_pred[:, 1] - target_v)
                
                bc_errors[f"{prefix}_{boundary_name}_u"] = {
                    'error': u_error,
                    'x': x_coords,
                    'y': y_coords,
                    'description': f'U Velocity Error at {boundary_name}',
                    'boundary_type': 'interior' if prefix == 'int' else 'exterior'
                }
                
                bc_errors[f"{prefix}_{boundary_name}_v"] = {
                    'error': v_error,
                    'x': x_coords,
                    'y': y_coords,
                    'description': f'V Velocity Error at {boundary_name}',
                    'boundary_type': 'interior' if prefix == 'int' else 'exterior'
                }
                
            elif isinstance(bc_type, InletBC):
                # For inlet BC, check against specified values
                if 'u' in conditions:
                    target_u = conditions['u']['value']
                    u_error = np.abs(vel_pred[:, 0] - target_u)
                    bc_errors[f"{prefix}_{boundary_name}_u"] = {
                        'error': u_error,
                        'x': x_coords,
                        'y': y_coords,
                        'description': f'U Velocity Error at {boundary_name}',
                        'boundary_type': 'interior' if prefix == 'int' else 'exterior'
                    }
                    
                if 'v' in conditions:
                    target_v = conditions['v']['value']
                    v_error = np.abs(vel_pred[:, 1] - target_v)
                    bc_errors[f"{prefix}_{boundary_name}_v"] = {
                        'error': v_error,
                        'x': x_coords,
                        'y': y_coords,
                        'description': f'V Velocity Error at {boundary_name}',
                        'boundary_type': 'interior' if prefix == 'int' else 'exterior'
                    }
                    
            elif isinstance(bc_type, OutletBC):
                # For outlet BC, pressure should be constant
                p_error = np.abs(p_pred - np.mean(p_pred))
                bc_errors[f"{prefix}_{boundary_name}_p"] = {
                    'error': p_error,
                    'x': x_coords,
                    'y': y_coords,
                    'description': f'Pressure Error at {boundary_name}',
                    'boundary_type': 'interior' if prefix == 'int' else 'exterior'
                }
        
        # Store mean errors for tracking over time
        for bc_name, error_data in bc_errors.items():
            error_values = error_data['error']
            mean_error = np.mean(error_values)
            max_error = np.max(error_values)
            
            if bc_name not in self.bc_error_history:
                self.bc_error_history[bc_name] = {'epochs': [], 'mean_errors': [], 'max_errors': []}
                
            self.bc_error_history[bc_name]['epochs'].append(epoch)
            self.bc_error_history[bc_name]['mean_errors'].append(mean_error)
            self.bc_error_history[bc_name]['max_errors'].append(max_error)
            
        return bc_errors
    
    def plot_boundary_conditions(self, model, mesh, epoch: int, domain_range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None, 
                                airfoil_coords: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
        """
        Evaluate and plot boundary condition errors.
        
        Args:
            model: The neural network model
            mesh: The mesh object containing boundary information
            epoch: Current training epoch
            domain_range: Optional tuple of ((x_min, x_max), (y_min, y_max)) for domain boundaries
            airfoil_coords: Optional tuple of (x_coords, y_coords) for airfoil or other interior boundary
        """
        # Evaluate boundary errors
        bc_errors = self.evaluate_boundary_errors(model, mesh, epoch)
        
        if not bc_errors:
            return
            
        # Create summary plot showing spatial distribution of errors
        self._plot_spatial_error_distribution(bc_errors, epoch, domain_range, airfoil_coords)
        
        # Create detailed plots for each boundary
        self._plot_detailed_boundary_errors(bc_errors, epoch)
        
        # Create separate plots for interior and exterior boundaries
        self._plot_boundary_type_comparison(bc_errors, epoch)
        
        print(f"Boundary condition plots saved for epoch {epoch}")
    
    def _plot_spatial_error_distribution(self, bc_errors: Dict[str, Dict[str, Any]], epoch: int, 
                                        domain_range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                                        airfoil_coords: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
        """
        Create a summary plot showing the spatial distribution of boundary condition errors.
        
        Args:
            bc_errors: Dictionary containing boundary error data
            epoch: Current training epoch
            domain_range: Optional tuple of ((x_min, x_max), (y_min, y_max)) for domain boundaries
            airfoil_coords: Optional tuple of (x_coords, y_coords) for airfoil or other interior boundary
        """
        plt.figure(figsize=(15, 10))
        
        # Plot domain outline if provided
        if domain_range is not None:
            x_range, y_range = domain_range
            plt.plot([x_range[0], x_range[1], x_range[1], x_range[0], x_range[0]],
                     [y_range[0], y_range[0], y_range[1], y_range[1], y_range[0]],
                     'k-', linewidth=0.5, alpha=0.5)
        
        # Plot airfoil outline if provided
        if airfoil_coords is not None:
            x_airfoil, y_airfoil = airfoil_coords
            plt.plot(x_airfoil, y_airfoil, 'k-', linewidth=1.0)
            
        # Set up the colormap and normalization
        cmap = cm.viridis
        max_error = max([np.max(data['error']) for data in bc_errors.values()])
        norm = Normalize(vmin=0, vmax=max_error)
        
        # Plot errors for each boundary
        for bc_name, data in bc_errors.items():
            # Use different marker styles for interior vs exterior boundaries
            marker = 'o' if data.get('boundary_type', 'exterior') == 'exterior' else '^'
            
            scatter = plt.scatter(data['x'], data['y'], 
                        c=data['error'],
                        cmap=cmap, 
                        norm=norm,
                        s=30, 
                        alpha=0.7,
                        marker=marker,
                        label=bc_name)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Error Magnitude')
        
        # Customize plot
        plt.title(f'Spatial Distribution of Boundary Condition Errors - Epoch {epoch}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Create a more compact legend with two columns
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', 
                  ncol=2 if len(handles) > 6 else 1, fontsize='small')
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f"{self.output_dir}/spatial_error_distribution_epoch_{epoch}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_detailed_boundary_errors(self, bc_errors: Dict[str, Dict[str, Any]], epoch: int) -> None:
        """
        Create detailed plots for each boundary condition error.
        
        Args:
            bc_errors: Dictionary containing boundary error data
            epoch: Current training epoch
        """
        for bc_name, data in bc_errors.items():
            error_values = data['error']
            x_coords = data['x']
            y_coords = data['y']
            description = data['description']
            boundary_type = data.get('boundary_type', 'exterior')
            
            # Create a figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Add boundary type to the figure title
            fig.suptitle(f"{description} ({boundary_type.capitalize()} Boundary)", fontsize=16)
            
            # 1. Plot histogram with distribution
            ax1.hist(error_values, bins=30, color='royalblue', alpha=0.7, edgecolor='black', linewidth=0.5)
            ax1.set_title(f'Error Distribution')
            ax1.set_xlabel('Error Magnitude')
            ax1.set_ylabel('Count')
            ax1.grid(True, linestyle='--', alpha=0.5)
            
            # Add mean and max error statistics
            mean_error = np.mean(error_values)
            max_error = np.max(error_values)
            stats_text = f"Mean Error: {mean_error:.6f}\nMax Error: {max_error:.6f}"
            ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # 2. Plot spatial distribution along the boundary
            # For clarity, we'll plot error vs. distance along the boundary
            # Calculate cumulative distance along the boundary
            if len(x_coords) > 1:
                dx = np.diff(x_coords)
                dy = np.diff(y_coords)
                distances = np.sqrt(dx**2 + dy**2)
                cumulative_distance = np.concatenate(([0], np.cumsum(distances)))
                
                # Plot error vs. distance along boundary
                scatter = ax2.scatter(cumulative_distance, error_values, c=error_values, 
                          cmap=cm.viridis, s=30, alpha=0.7)
                ax2.plot(cumulative_distance, error_values, 'k-', alpha=0.3)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax2)
                cbar.set_label('Error Magnitude')
                
                ax2.set_title(f'Error along Boundary')
                ax2.set_xlabel('Distance along Boundary')
                ax2.set_ylabel('Error Magnitude')
                ax2.grid(True, linestyle='--', alpha=0.5)
            else:
                ax2.text(0.5, 0.5, "Not enough points to plot", ha='center', va='center')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
            plt.savefig(f"{self.detailed_dir}/{bc_name}_error_epoch_{epoch}.png", dpi=300)
            plt.close()
    
    def _plot_boundary_type_comparison(self, bc_errors: Dict[str, Dict[str, Any]], epoch: int) -> None:
        """
        Create comparison plots for interior vs exterior boundary errors.
        
        Args:
            bc_errors: Dictionary containing boundary error data
            epoch: Current training epoch
        """
        # Separate interior and exterior boundaries
        interior_errors = {k: v for k, v in bc_errors.items() 
                          if v.get('boundary_type', 'exterior') == 'interior'}
        exterior_errors = {k: v for k, v in bc_errors.items() 
                          if v.get('boundary_type', 'exterior') == 'exterior'}
        
        # Skip if either category is empty
        if not interior_errors or not exterior_errors:
            return
            
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Interior vs Exterior Boundary Errors - Epoch {epoch}", fontsize=16)
        
        # Plot interior boundary errors
        self._plot_error_boxplot(ax1, interior_errors, "Interior Boundaries")
        
        # Plot exterior boundary errors
        self._plot_error_boxplot(ax2, exterior_errors, "Exterior Boundaries")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
        plt.savefig(f"{self.output_dir}/boundary_type_comparison_epoch_{epoch}.png", dpi=300)
        plt.close()
    
    def _plot_error_boxplot(self, ax, errors_dict, title):
        """Helper method to create boxplots of errors."""
        if not errors_dict:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title(title)
            return
            
        # Prepare data for boxplot
        data = []
        labels = []
        
        for name, error_data in errors_dict.items():
            data.append(error_data['error'])
            # Simplify the label by removing the prefix
            simple_name = name.split('_', 1)[1] if '_' in name else name
            labels.append(simple_name)
        
        # Create boxplot
        ax.boxplot(data, labels=labels, vert=True, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.8),
                  medianprops=dict(color='red'),
                  flierprops=dict(marker='o', markerfacecolor='red', markersize=3))
        
        # Customize plot
        ax.set_title(title)
        ax.set_ylabel('Error Magnitude')
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    def plot_error_evolution(self) -> None:
        """
        Plot the evolution of boundary condition errors over training epochs.
        """
        if not self.bc_error_history:
            print("No boundary condition error history to plot.")
            return
        
        # Separate interior and exterior boundaries
        interior_history = {k: v for k, v in self.bc_error_history.items() if 'int_' in k}
        exterior_history = {k: v for k, v in self.bc_error_history.items() if 'ext_' in k}
        
        # Plot overall error evolution
        self._plot_overall_error_evolution()
        
        # Plot separate error evolution for interior and exterior if both exist
        if interior_history and exterior_history:
            self._plot_boundary_type_error_evolution(interior_history, exterior_history)
    
    def _plot_overall_error_evolution(self) -> None:
        """Plot the overall evolution of boundary condition errors."""
        plt.figure(figsize=(15, 10))
        
        # Plot mean errors
        plt.subplot(2, 1, 1)
        for bc_name, history in self.bc_error_history.items():
            # Use different line styles for interior vs exterior boundaries
            linestyle = '--' if 'int_' in bc_name else '-'
            plt.semilogy(history['epochs'], history['mean_errors'], 'o', 
                        linestyle=linestyle, label=bc_name)
        
        plt.title('Evolution of Mean Boundary Condition Errors')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Error (log scale)')
        plt.grid(True, which="both", ls="-")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        # Plot max errors
        plt.subplot(2, 1, 2)
        for bc_name, history in self.bc_error_history.items():
            # Use different line styles for interior vs exterior boundaries
            linestyle = '--' if 'int_' in bc_name else '-'
            plt.semilogy(history['epochs'], history['max_errors'], 'o', 
                        linestyle=linestyle, label=bc_name)
        
        plt.title('Evolution of Maximum Boundary Condition Errors')
        plt.xlabel('Epoch')
        plt.ylabel('Max Error (log scale)')
        plt.grid(True, which="both", ls="-")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/bc_error_evolution.png", dpi=300)
        plt.close()
    
    def _plot_boundary_type_error_evolution(self, interior_history, exterior_history) -> None:
        """Plot separate error evolution for interior and exterior boundaries."""
        plt.figure(figsize=(15, 10))
        
        # Plot mean errors for interior boundaries
        plt.subplot(2, 2, 1)
        for bc_name, history in interior_history.items():
            plt.semilogy(history['epochs'], history['mean_errors'], 'o-', label=bc_name)
        
        plt.title('Mean Error - Interior Boundaries')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Error (log scale)')
        plt.grid(True, which="both", ls="-")
        plt.legend(fontsize='x-small')
        
        # Plot mean errors for exterior boundaries
        plt.subplot(2, 2, 2)
        for bc_name, history in exterior_history.items():
            plt.semilogy(history['epochs'], history['mean_errors'], 'o-', label=bc_name)
        
        plt.title('Mean Error - Exterior Boundaries')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Error (log scale)')
        plt.grid(True, which="both", ls="-")
        plt.legend(fontsize='x-small')
        
        # Plot max errors for interior boundaries
        plt.subplot(2, 2, 3)
        for bc_name, history in interior_history.items():
            plt.semilogy(history['epochs'], history['max_errors'], 'o-', label=bc_name)
        
        plt.title('Max Error - Interior Boundaries')
        plt.xlabel('Epoch')
        plt.ylabel('Max Error (log scale)')
        plt.grid(True, which="both", ls="-")
        plt.legend(fontsize='x-small')
        
        # Plot max errors for exterior boundaries
        plt.subplot(2, 2, 4)
        for bc_name, history in exterior_history.items():
            plt.semilogy(history['epochs'], history['max_errors'], 'o-', label=bc_name)
        
        plt.title('Max Error - Exterior Boundaries')
        plt.xlabel('Epoch')
        plt.ylabel('Max Error (log scale)')
        plt.grid(True, which="both", ls="-")
        plt.legend(fontsize='x-small')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/bc_error_evolution_by_type.png", dpi=300)
        plt.close() 