import matplotlib.pyplot as plt
import numpy as np

from src.plot.postprocess import Postprocess

class Plot:

    def __init__(self, mesh):
        self.X = mesh.x
        self.Y = mesh.y

        if mesh.is2D == False:
            self.Z = mesh.Z

        self.solutions = mesh.solutions

        self.postprocessor = Postprocess(self)

    def plot(self, solkey, streamlines):
        from scipy.interpolate import griddata

        if solkey == 'vMag':
            self.postprocessor.compute_velocity_magnitude()

        if solkey not in self.solutions:
            raise KeyError(
                f"The solution key '{solkey}' was not found in the available solutions. "
                f"Available keys are: {list(self.solutions.keys())}."
            )

        x = self.X
        y = self.Y
        sol = self.solutions[solkey]

        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))

        # Interpolate the scattered data to the grid using griddata
        grid_sol = griddata((x, y), sol, (grid_x, grid_y), method='cubic')

        plt.figure(figsize=(8, 6))
        plt.title(f'Solution Field {solkey}')
        
        # Plot contourf
        cp = plt.contourf(grid_x, grid_y, grid_sol, cmap='jet', levels=50)
        plt.colorbar(cp)

        if streamlines:
            if 'u' not in self.solutions or 'v' not in self.solutions:
                raise KeyError("Streamline plotting requires 'u' and 'v' velocity components in solutions.")
            
            # Interpolate velocity components for streamlines
            u = self.solutions['u']
            v = self.solutions['v']
            grid_u = griddata((x, y), u, (grid_x, grid_y), method='cubic')
            grid_v = griddata((x, y), v, (grid_x, grid_y), method='cubic')
            
            # Plot streamlines
            plt.streamplot(grid_x, grid_y, grid_u, grid_v, color='k', linewidth=1)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


