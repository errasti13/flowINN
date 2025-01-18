import matplotlib.pyplot as plt
import numpy as np

from src.plot.postprocess import Postprocess

class Plot:

    def __init__(self, mesh):
        self.X = mesh.X
        self.Y = mesh.Y

        if mesh.is2D == False:
            self.Z = mesh.Z

        self.solutions = mesh.solutions

    def plot(self, solkey):
        from scipy.interpolate import griddata

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

        # Plot the result using contourf
        plt.figure(figsize=(8, 6))
        plt.title(f'Solution Field {solkey}')
        cp = plt.contourf(grid_x, grid_y, grid_sol, cmap='jet', levels=50)
        plt.colorbar(cp)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

