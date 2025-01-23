import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.plot.plot import Plot

class Postprocess:
    def __init__(self, plot_obj: 'Plot') -> None:
        """Initialize postprocessing with plot object."""
        self._plot: 'Plot' = plot_obj
        self._solutions: dict = plot_obj.solutions

    @property
    def plot(self) -> 'Plot':
        return self._plot

    @plot.setter
    def plot(self, value: 'Plot') -> None:
        from src.plot.plot import Plot
        if not isinstance(value, Plot):
            raise TypeError("plot must be a Plot instance")
        self._plot = value

    @property
    def solutions(self) -> dict:
        return self._solutions

    @solutions.setter
    def solutions(self, value: dict) -> None:
        if not isinstance(value, dict):
            raise TypeError("solutions must be a dictionary")
        self._solutions = value

    def compute_velocity_magnitude(self) -> None:
        """Compute velocity magnitude from velocity components."""
        u = self.solutions['u']
        v = self.solutions['v']
        
        if self.plot.Z is not None:
            w = self.solutions['w']
            magnitude = np.sqrt(u**2 + v**2 + w**2)
        else:
            magnitude = np.sqrt(u**2 + v**2)
            
        self.solutions['vMag'] = magnitude
    
    def compute_vorticity(self):
        """
        Compute vorticity (to be implemented based on your mesh structure)
        This is just a placeholder showing how you could extend the postprocessor
        """
        pass
    
    def compute_pressure_coefficient(self, rho_inf=1.0, v_inf=1.0):
        """
        Compute pressure coefficient: Cp = (p - p_inf)/(0.5 * rho_inf * v_inf²)
        """
        p = self.solutions['p']
        p_inf = 0  # You might want to make this a parameter
        
        cp = (p - p_inf) / (0.5 * rho_inf * v_inf**2)
        self.solutions['pressure_coefficient'] = cp
        return cp