import numpy as np
from typing import List, Optional, Dict
# Remove the import of the Mesh class to break the circular dependency
#from src.mesh.mesh import Mesh  # Import Mesh class


class MeshIO:
    """
    A class for handling mesh input/output operations.

    Attributes:
        mesh (Mesh): The mesh object to which this MeshIO instance is associated.
        variables (List[str]): List of variable names to be written to file.
    """

    def __init__(self, mesh) -> None:
        """
        Initializes a new MeshIO object.

        Args:
            mesh (Mesh): The mesh object to which this MeshIO instance is associated.
        """
        
        self._mesh = mesh
        self._variables: List[str] = ["X", "Y", "U", "V", "P"]

    @property
    def x(self) -> np.ndarray:
        """
        Returns the x-coordinates of the mesh points.
        """
        return self._mesh.x

    @property
    def y(self) -> np.ndarray:
        """
        Returns the y-coordinates of the mesh points.
        """
        return self._mesh.y

    @property
    def z(self) -> Optional[np.ndarray]:
        """
        Returns the z-coordinates of the mesh points.
        """
        return self._mesh.z

    @property
    def solutions(self) -> Dict[str, np.ndarray]:
        """
        Returns the solutions dictionary from the mesh.
        """
        return self._mesh.solutions

    @solutions.setter
    def solutions(self, value: Dict[str, np.ndarray]) -> None:
        """
        Sets the solutions dictionary in the mesh.

        Args:
            value (Dict[str, np.ndarray]): A dictionary containing solution data.

        Raises:
            TypeError: If value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("solutions must be a dictionary")
        self._mesh.solutions = value

    @property
    def boundaries(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Returns the boundaries dictionary from the mesh.
        """
        return self._mesh.boundaries

    @boundaries.setter
    def boundaries(self, value: Dict[str, Dict[str, np.ndarray]]) -> None:
        """
        Sets the boundaries dictionary in the mesh.

        Args:
            value (Dict[str, Dict[str, np.ndarray]]): A dictionary containing boundary data.

        Raises:
            TypeError: If value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("boundaries must be a dictionary")
        self._mesh.boundaries = value

    @property
    def interiorBoundaries(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Returns the interior boundaries dictionary from the mesh.
        """
        return self._mesh.interiorBoundaries

    @interiorBoundaries.setter
    def interiorBoundaries(self, value: Dict[str, Dict[str, np.ndarray]]) -> None:
        """
        Sets the interior boundaries dictionary in the mesh.

        Args:
            value (Dict[str, Dict[str, np.ndarray]]): A dictionary containing interior boundary data.

        Raises:
            TypeError: If value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("interiorBoundaries must be a dictionary")
        self._mesh.interiorBoundaries = value

    @property
    def is2D(self) -> bool:
        """
        Returns the is2D flag from the mesh.
        """
        return self._mesh.is2D

    @is2D.setter
    def is2D(self, value: bool) -> None:
        """
        Sets the is2D flag in the mesh.

        Args:
            value (bool): A boolean value indicating if the mesh is 2D.

        Raises:
            TypeError: If value is not a boolean.
        """
        if not isinstance(value, bool):
            raise TypeError("is2D must be a boolean")
        self._mesh.is2D = value

    def write_tecplot(self, filename: str, variables: Optional[List[str]] = None) -> None:
        """
        Writes the solution to a Tecplot file.

        Args:
            filename (str): The name of the Tecplot file to write.
            variables (Optional[List[str]]): Optional list of variable names to write.
                                             If None, all variables in solutions will be used.

        Raises:
            IOError: If writing to the file fails.
        """
        try:
            x = self.x.flatten()
            y = self.y.flatten()
            data_dict: Dict[str, np.ndarray] = {}

            # Add coordinates
            data_dict['x'] = x
            data_dict['y'] = y
            if not self.is2D and self.z is not None:
                data_dict['z'] = self.z.flatten()

            # Add all solution variables
            for var_name, var_data in self.solutions.items():
                data_dict[var_name] = var_data.flatten()

            if variables:  # If specific variables are requested, filter data_dict
                filtered_dict = {k: data_dict[k] for k in variables if k in data_dict}
                if filtered_dict:
                    data_dict = filtered_dict
                else:
                    raise ValueError("None of the specified variables found in the solution data")

            dtype = [(name, 'float64') for name in data_dict.keys()]
            structured_data = np.zeros(len(x), dtype=dtype)
            for name in data_dict.keys():
                structured_data[name] = data_dict[name]

            header = ','.join(data_dict.keys())
            np.savetxt(filename,
                      structured_data,
                      delimiter=',',
                      header=header,
                      comments='',
                      fmt='%.8e')

        except Exception as e:
            raise IOError(f"Error writing data file: {str(e)}")

    def write_solution(self, filename: str, variables: Optional[List[str]] = None) -> None:
        """
        Writes the solution to a file in CSV format.

        Args:
            filename (str): Path to the output file.
            variables (Optional[List[str]]): Optional list of variable names to write.
                                             If None, the default variables will be used.

        Raises:
            IOError: If writing fails.
        """
        if not filename.endswith('.csv'):
            filename += '.csv'

        try:
            self.write_tecplot(filename, variables)
        except Exception as e:
            raise IOError(f"Failed to write solution: {str(e)}")

    def set_variables(self, variables: List[str]) -> None:
        """
        Sets the variables to be written to file.

        Args:
            variables (List[str]): List of variable names.

        Raises:
            ValueError: If the variables list is empty or contains invalid names.
            TypeError: If not all variables are strings.
        """
        if not variables:
            raise ValueError("Variables list cannot be empty")
        if not all(isinstance(v, str) for v in variables):
            raise TypeError("All variables must be strings")
        self._variables = variables
