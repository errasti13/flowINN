import numpy as np

class Postprocess:
    def __init__(self, plot_obj):
        """
        Initialize postprocessor with a Plot object
        
        Args:
            plot_obj (Plot): Plot object containing solution data
        """
        self.plot = plot_obj
        self.solutions = plot_obj.solutions
    
    def compute_velocity_magnitude(self):
        """
        Compute velocity magnitude from u and v components.
        For 2D flows: magnitude = sqrt(u² + v²)
        For 3D flows: magnitude = sqrt(u² + v² + w²)
        """
        u = self.solutions['u']
        v = self.solutions['v']
        
        if hasattr(self.plot, 'Z'):  # Check if it's a 3D problem
            w = self.solutions['w']
            magnitude = np.sqrt(u**2 + v**2 + w**2)
        else:
            magnitude = np.sqrt(u**2 + v**2)
            
        self.solutions['velocity_magnitude'] = magnitude
        return magnitude
    
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