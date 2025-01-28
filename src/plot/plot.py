import matplotlib.pyplot as plt
import numpy as np
from src.plot.postprocess import Postprocess

class Plot:
    def __init__(self, mesh):
        self._mesh = mesh
        self._postprocessor: Postprocess = None

    @property
    def mesh(self) -> None:
        return self._mesh

    @mesh.setter
    def Z(self, value: np.ndarray) -> None:
        self._mesh = value

    @property
    def postprocessor(self) -> Postprocess:
        return self._postprocessor

    @postprocessor.setter
    def postprocessor(self, value: Postprocess) -> None:
        if not isinstance(value, Postprocess):
            raise TypeError("postprocessor must be a Postprocess instance")
        self._postprocessor = value

    def plot(self, solkey, streamlines):
        from scipy.interpolate import griddata

        if solkey == 'vMag' and 'vMag' not in self.mesh.solutions:
            self.postprocessor.compute_velocity_magnitude()

        if solkey not in self.mesh.solutions:
            raise KeyError(
                f"The solution key '{solkey}' was not found in the available solutions. "
                f"Available keys are: {list(self.mesh.solutions.keys())}."
            )

        x = self.mesh.x
        y = self.mesh.y
        sol = self.mesh.solutions[solkey]

        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))

        # Interpolate the scattered data to the grid using griddata
        grid_sol = griddata((x, y), sol, (grid_x, grid_y), method='cubic')

        plt.figure(figsize=(8, 6))
        plt.title(f'Solution Field {solkey}')
        
        # Plot contourf
        cp = plt.contourf(grid_x, grid_y, grid_sol, cmap='jet', levels=50)
        plt.colorbar(cp)

        if streamlines:
            if 'u' not in self.mesh.solutions or 'v' not in self.mesh.solutions:
                raise KeyError("Streamline plotting requires 'u' and 'v' velocity components in solutions.")
            
            # Interpolate velocity components for streamlines
            u = self.mesh.solutions['u']
            v = self.mesh.solutions['v']
            grid_u = griddata((x, y), u, (grid_x, grid_y), method='cubic')
            grid_v = griddata((x, y), v, (grid_x, grid_y), method='cubic')
            
            # Plot streamlines
            plt.streamplot(grid_x, grid_y, grid_u, grid_v, color='k', linewidth=1)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def scatterPlot(self, solkey):
        """
        Visualize the solution field using scatter plot with boundaries.
        """
        x = self.mesh.x.flatten()
        y = self.mesh.y.flatten()       

        sol = self.mesh.solutions[solkey]
        plt.figure(figsize=(8, 6))
        plt.title(f'Solution Field: {solkey}')  # Changed title
        
        # Plot mesh points
        scatter = plt.scatter(x, y, c=sol, s=5, cmap='jet')  # Removed label from scatter
        plt.colorbar(scatter, label=solkey)  # Added label to colorbar
        
        # Plot exterior boundaries
        for boundary_data in self.mesh.boundaries.values():
            x_boundary = boundary_data['x']
            y_boundary = boundary_data['y']
            plt.plot(x_boundary, y_boundary, 'b-', linewidth=2, label='Exterior Boundary')
        
        # Plot interior boundaries if they exist
        if self.mesh.interiorBoundaries:
            for boundary_data in self.mesh.interiorBoundaries.values():
                x_boundary = boundary_data['x']
                y_boundary = boundary_data['y']
                plt.plot(x_boundary, y_boundary, 'r-', linewidth=2, label='Interior Boundary')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        
        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        plt.show()

