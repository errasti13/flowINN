import numpy as np
from typing import List, Optional
from src.mesh.mesh import Mesh

class MeshIO(Mesh):
    def __init__(self, is2D: bool = True) -> None:
        """
        Initialize MeshIO class.
        
        Args:
            is2D: Whether the mesh is 2D (True) or 3D (False)
        """
        super().__init__(is2D)
        self._variables: List[str] = ["X", "Y", "U", "V", "P"]

    def write_tecplot(self, filename: str, variables: Optional[List[str]] = None) -> None:
        """Write solution in a Paraview-compatible CSV format."""
        if variables:
            self._variables = variables

        try:
            # Prepare data arrays
            x = self.x.flatten()
            y = self.y.flatten()
            data_dict = {}
            
            # Add coordinates first
            data_dict['x'] = x
            data_dict['y'] = y
            
            # Add solution variables
            for var in ['u', 'v', 'p']:  # Fixed order of variables
                if var in self.solutions:
                    data_dict[var] = self.solutions[var].flatten()
                else:
                    data_dict[var] = np.zeros_like(x)  # Fill with zeros if missing
            
            # Convert to structured array for clean CSV output
            dtype = [(name, 'float64') for name in data_dict.keys()]
            structured_data = np.zeros(len(x), dtype=dtype)
            for name in data_dict.keys():
                structured_data[name] = data_dict[name]

            # Write to CSV with proper header
            header = ','.join(data_dict.keys())  # Simple comma-separated header
            np.savetxt(filename, 
                      structured_data,
                      delimiter=',',
                      header=header,
                      comments='',  # No # in header
                      fmt='%.8e')   # Scientific notation with 8 decimal places

        except Exception as e:
            raise IOError(f"Error writing data file: {str(e)}")

    def write_solution(self, filename: str, variables: Optional[List[str]] = None) -> None:
        """
        Write solution to file in CSV format.
        
        Args:
            filename: Path to output file
            variables: Optional list of variable names to write
        
        Raises:
            IOError: If writing fails
            ValueError: If required solution variables are missing
        """
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        try:
            self.write_tecplot(filename, variables)
        except Exception as e:
            raise IOError(f"Failed to write solution: {str(e)}")

    def set_variables(self, variables: List[str]) -> None:
        """
        Set the variables to be written to file.
        
        Args:
            variables: List of variable names
            
        Raises:
            ValueError: If variables list is empty or contains invalid names
        """
        if not variables:
            raise ValueError("Variables list cannot be empty")
        if not all(isinstance(v, str) for v in variables):
            raise ValueError("All variables must be strings")
        self._variables = variables
