import numpy as np
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from src.plot.plot import Plot


class Postprocess:
    """
    A class for postprocessing simulation results, such as computing derived quantities.

    Attributes:
        plot (Plot): The Plot object associated with this postprocessor.
        solutions (Dict[str, np.ndarray]): The dictionary containing solution fields.
    """

    def __init__(self, plot_obj: 'Plot') -> None:
        """
        Initializes a new Postprocess object.

        Args:
            plot_obj (Plot): The Plot object associated with this postprocessor.
        """
        from src.plot.plot import Plot
        if not isinstance(plot_obj, Plot):
            raise TypeError("plot_obj must be a Plot instance")

        self._plot: 'Plot' = plot_obj
        self._solutions: Dict[str, np.ndarray] = plot_obj.mesh.solutions

    @property
    def plot(self) -> 'Plot':
        """
        Returns the Plot object.
        """
        return self._plot

    @plot.setter
    def plot(self, value: 'Plot') -> None:
        """
        Sets the Plot object.

        Args:
            value (Plot): The new Plot object.

        Raises:
            TypeError: If value is not a Plot instance.
        """
        from src.plot.plot import Plot
        if not isinstance(value, Plot):
            raise TypeError("plot must be a Plot instance")
        self._plot = value

    @property
    def solutions(self) -> Dict[str, np.ndarray]:
        """
        Returns the solutions dictionary.
        """
        return self._solutions

    @solutions.setter
    def solutions(self, value: Dict[str, np.ndarray]) -> None:
        """
        Sets the solutions dictionary.

        Args:
            value (Dict[str, np.ndarray]): The new solutions dictionary.

        Raises:
            TypeError: If value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("solutions must be a dictionary")
        self._solutions = value

    def compute_velocity_magnitude(self) -> None:
        """
        Computes the velocity magnitude from the velocity components (u, v, and optionally w).

        The computed velocity magnitude is stored in the solutions dictionary under the key 'vMag'.
        """
        u: np.ndarray = self.solutions['u']
        v: np.ndarray = self.solutions['v']

        if self._plot.mesh.z is not None:
            w: np.ndarray = self.solutions['w']
            magnitude: np.ndarray = np.sqrt(u**2 + v**2 + w**2)
        else:
            magnitude: np.ndarray = np.sqrt(u**2 + v**2)

        self.solutions['vMag'] = magnitude

    def compute_vorticity(self) -> None:
        """
        Computes the vorticity (to be implemented based on your mesh structure).

        This method is a placeholder and should be implemented based on the specific
        mesh structure and numerical scheme used in the simulation.
        """
        pass

    def compute_pressure_coefficient(self, rho_inf: float = 1.0, v_inf: float = 1.0) -> Optional[np.ndarray]:
        """
        Computes the pressure coefficient: Cp = (p - p_inf)/(0.5 * rho_inf * v_inf²)

        Args:
            rho_inf (float): Free stream density. Defaults to 1.0.
            v_inf (float): Free stream velocity. Defaults to 1.0.

        Returns:
            Optional[np.ndarray]: The computed pressure coefficient, or None if 'p' is not in solutions.
        """
        if 'p' not in self.solutions:
            return None
        p: np.ndarray = self.solutions['p']
        p_inf: float = 0.0  # You might want to make this a parameter

        cp: np.ndarray = (p - p_inf) / (0.5 * rho_inf * v_inf**2)
        self.solutions['pressure_coefficient'] = cp
        return cp