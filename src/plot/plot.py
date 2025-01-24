import matplotlib.pyplot as plt
import numpy as np
from src.plot.postprocess import Postprocess

class Plot:
    def __init__(self, mesh):
        self._X: np.ndarray = mesh.x
        self._Y: np.ndarray = mesh.y
        self._Z: np.ndarray = None
        self._solutions: dict = mesh.solutions
        self._postprocessor: Postprocess = None

        if not mesh.is2D:
            self._Z = mesh.Z

        self._postprocessor = Postprocess(self)

    @property
    def X(self) -> np.ndarray:
        return self._X

    @X.setter
    def X(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("X must be a numpy array")
        self._X = value

    @property
    def Y(self) -> np.ndarray:
        return self._Y

    @Y.setter
    def Y(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("Y must be a numpy array")
        self._Y = value

    @property
    def Z(self) -> np.ndarray:
        return self._Z

    @Z.setter
    def Z(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("Z must be a numpy array")
        self._Z = value

    @property
    def solutions(self) -> dict:
        return self._solutions

    @solutions.setter
    def solutions(self, value: dict) -> None:
        if not isinstance(value, dict):
            raise TypeError("solutions must be a dictionary")
        self._solutions = value

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

    def scatterPlot(self, solkey):
        """
        Create a scatter plot of the solution field.

        Parameters:
        - solkey: str, key of the solution to plot

        Raises:
        - KeyError: If solution key doesn't exist
        - ValueError: If solution data is invalid
        """
        try:
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

            # Validate data
            if len(x) != len(y) or len(x) != len(sol):
                raise ValueError("Coordinate and solution arrays must have the same length")

            plt.figure(figsize=(8, 6))
            plt.title(f'Solution Field {solkey}')
            
            # Fixed scatter plot parameters
            scatter = plt.scatter(x, y, s=5, c=sol, cmap='jet')
            plt.colorbar(scatter, label=solkey)
            
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.axis('equal')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            plt.close()  # Close figure in case of error
            raise type(e)(f"Error in scatter plot: {str(e)}")


